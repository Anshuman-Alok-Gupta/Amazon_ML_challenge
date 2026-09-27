"""Residual edge GNN over the candidate graph (S1 entities <-> S2/S3 records).

Nodes are S1 entities and target records, edges are candidate pairs. An edge starts from its
stack inputs (level 0 + level-1 logits + relational features + level-2 logit + degrees) and, over
a few rounds, reads what the *other* edges say at its S1, at its target record and at its
(S1, source) group:
  - the leave-one-out mean of their states
  - the max of their states, relative to its own
  - a competition term: its score minus the log-sum-exp of theirs (who else claims this record?)
The output is the level-2 logit plus a learned correction whose last layer starts at zero, so an
untrained GNN is exactly the stack, and early stopping at epoch 0 falls back to it.

Why a complete graph: the stack's training frame is a sample of S1, so a target's competing S1
are mostly missing there. Target-side competition only looks like test on a frame holding every
S1 of its regions, with the same hidden share of S1 as test (pipeline graph-frame). Training is
transductive: every edge is visible, and the loss uses only the training folds' labels (labels
are never inputs), as on test, where every edge is visible and none is labelled.
"""
from __future__ import annotations

import copy
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch import nn

import model as mdl
import stack
from config import SEED, GNNConfig, PostConfig
from postprocess import MacroScorer, select_with

warnings.filterwarnings("ignore", message="index_reduce")  # beta API notice


# ------------------------------------------------------------------ inputs
def range_inputs(fdir: Path, a: int, b: int, P: pd.DataFrame, spec: dict, models: dict, target_paths):
    """Stack features of rows [a, b) of a frame (one country): level-2 input Z + level-2 logit."""
    X = stack.read_level0(fdir, a, b, spec)
    _, Lm = stack.level1(P, X, spec, models)
    rec = stack.Records(target_paths, P["cand_id"])
    Z, p2 = stack.level2(P, X, Lm, spec, models, rec)
    Z = Z[spec["l2_features"]].astype(np.float32)
    Z["l2"] = stack.logit(p2)
    return Z


def graph_keys(P: pd.DataFrame):
    """Integer node keys of every edge: S1, target record, (S1, source) group."""
    s1 = pd.factorize(P["s1_id"].to_numpy(object))[0].astype(np.int64)
    tgt = pd.factorize(P["cand_id"].to_numpy(object))[0].astype(np.int64)
    src3 = (P["source"].to_numpy() == 3).astype(np.int64)
    return s1, tgt, s1 * 2 + src3, src3


def add_graph_features(Z: pd.DataFrame, s1, tgt, grp, src3) -> pd.DataFrame:
    for name, k in (("s1", s1), ("tgt", tgt), ("grp", grp)):
        Z[f"g_deg_{name}"] = np.log1p(np.bincount(k)[k]).astype(np.float32)
    Z["g_src3"] = src3.astype(np.float32)
    return Z


class Scaler:
    """Standardise, clip, and add a missing-value indicator for columns that had NaNs at fit."""

    def __init__(self, state: dict | None = None):
        self.state = state

    def fit(self, Z: pd.DataFrame) -> "Scaler":
        v = Z.to_numpy(np.float64)
        v[~np.isfinite(v)] = np.nan
        with np.errstate(all="ignore"):
            mean, std = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
        mean, std = np.nan_to_num(mean), np.where(np.nan_to_num(std) > 1e-6, std, 1.0)
        self.state = {"cols": list(Z.columns), "mean": mean.tolist(), "std": std.tolist(),
                      "nan_cols": [c for c, n in zip(Z.columns, np.isnan(v).any(axis=0)) if n]}
        return self

    def transform(self, Z: pd.DataFrame) -> np.ndarray:
        s = self.state
        v = Z[s["cols"]].to_numpy(np.float32)
        v[~np.isfinite(v)] = np.nan
        miss = np.isnan(Z[s["nan_cols"]].to_numpy(np.float32)).astype(np.float32)
        v = np.clip((v - np.float32(s["mean"])) / np.float32(s["std"]), -5, 5)
        return np.hstack([np.nan_to_num(v, nan=0.0), miss]).astype(np.float32)


# ------------------------------------------------------------------ graph batches
def edge_components(s1: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Connected component of every edge in the bipartite S1 / target graph."""
    n1, n2 = int(s1.max()) + 1, int(tgt.max()) + 1
    A = coo_matrix((np.ones(len(s1), np.int8), (s1, n1 + tgt)), shape=(n1 + n2, n1 + n2))
    return connected_components(A, directed=False)[1][s1]


def component_batches(comp: np.ndarray, max_edges: int) -> list[np.ndarray]:
    """Edge index arrays holding whole components, grouped up to about max_edges each."""
    order = np.argsort(comp, kind="stable")
    cs = comp[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    cuts = [0]
    for s in starts[1:]:
        if s - cuts[-1] >= max_edges:
            cuts.append(int(s))
    cuts.append(len(cs))
    return [order[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


class Batch:
    """Edges of whole components: inputs, base logit, and node indices local to the batch."""

    def __init__(self, edges: np.ndarray, x: np.ndarray, base: np.ndarray, keys, y: np.ndarray | None = None):
        self.edges = edges
        self.x = torch.from_numpy(x[edges])
        self.base = torch.from_numpy(base[edges].astype(np.float32))
        self.y = torch.from_numpy(y[edges].astype(np.float32)) if y is not None else None
        self.groups = []
        for k in keys:
            u, inv = np.unique(k[edges], return_inverse=True)
            idx = torch.from_numpy(inv.astype(np.int64))
            cnt = torch.from_numpy(np.bincount(inv, minlength=len(u)).astype(np.float32))
            self.groups.append((idx, len(u), cnt))


def make_batches(Z: pd.DataFrame, scaler: Scaler, keys, y, max_edges: int) -> list[Batch]:
    x = scaler.transform(Z)
    base = Z["l2"].to_numpy(np.float32)
    comp = edge_components(keys[0], keys[1])
    out = [Batch(e, x, base, keys[:3], y) for e in component_batches(comp, max_edges)]
    print(f"  graph: {len(Z):,} edges, {comp.max() + 1:,} components, {len(out)} batches, "
          f"{x.shape[1]} inputs", flush=True)
    return out


# ------------------------------------------------------------------ model
def seg_stats(m: torch.Tensor, s: torch.Tensor, idx: torch.Tensor, n: int, cnt: torch.Tensor):
    """What the other edges of each edge's node say: LOO mean, max gap, competition, alone flag."""
    d = m.shape[1]
    c = cnt[idx].unsqueeze(1)
    tot = m.new_zeros(n, d).index_add_(0, idx, m)
    loo = (tot[idx] - m) / (c - 1).clamp(min=1)
    mx = m.new_full((n, d), -1e4).index_reduce_(0, idx, m, "amax", include_self=True)
    gap = mx[idx] - m
    smax = s.detach().new_full((n,), -1e4).index_reduce_(0, idx, s.detach(), "amax", include_self=True)
    ex = torch.exp(s - smax[idx])
    other = (s.new_zeros(n).index_add_(0, idx, ex)[idx] - ex).clamp(min=0)
    comp = (s - smax[idx] - torch.log(other + 1e-4)).clamp(-10, 10).unsqueeze(1)
    return [loo, gap, comp, (c <= 1).float()]


class Layer(nn.Module):
    def __init__(self, d: int, hidden: int, n_groups: int = 3):
        super().__init__()
        self.msg, self.score = nn.Linear(d, d), nn.Linear(d, 1)
        self.upd = nn.Sequential(nn.Linear(d + n_groups * (2 * d + 2), hidden), nn.GELU(), nn.Linear(hidden, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, e: torch.Tensor, b: Batch) -> torch.Tensor:
        m, s = self.msg(e), self.score(e).squeeze(1)
        parts = [e]
        for idx, n, cnt in b.groups:
            parts += seg_stats(m, s, idx, n, cnt)
        return self.norm(e + self.upd(torch.cat(parts, 1)))


class EdgeGNN(nn.Module):
    def __init__(self, f_in: int, cfg: GNNConfig):
        super().__init__()
        d, h = cfg.dim, cfg.hidden
        self.enc = nn.Sequential(nn.Linear(f_in, h), nn.GELU(), nn.Linear(h, d), nn.LayerNorm(d))
        self.layers = nn.ModuleList(Layer(d, h) for _ in range(cfg.layers))
        self.head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.head[-1].weight)  # starts as the stack: logit = level-2 logit
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, b: Batch) -> torch.Tensor:
        e = self.enc(b.x)
        for layer in self.layers:
            e = layer(e, b)
        return b.base + self.head(e).squeeze(1)


def infer(net: EdgeGNN, batches: list[Batch], n: int) -> np.ndarray:
    net.eval()
    out = np.zeros(n, np.float32)
    with torch.no_grad():
        for b in batches:
            out[b.edges] = net(b).numpy()
    return out


def bce(z: np.ndarray, y: np.ndarray) -> float:
    z = z.astype(np.float64)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_fold(batches: list[Batch], train: np.ndarray, val: np.ndarray, y: np.ndarray, f_in: int, cfg: GNNConfig):
    """Train on the edges flagged `train`, early-stop on the BCE of `val` (epoch 0 = the stack)."""
    torch.manual_seed(SEED)
    net = EdgeGNN(f_in, cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    masks = [torch.from_numpy(train[b.edges]) for b in batches]
    rng = np.random.default_rng(SEED)
    n = len(y)
    z = infer(net, batches, n)
    best, best_ep, best_state, best_z = bce(z[val], y[val]), 0, copy.deepcopy(net.state_dict()), z
    print(f"    epoch 0 (stack): val BCE {best:.5f}", flush=True)
    for ep in range(1, cfg.max_epochs + 1):
        t = time.perf_counter()
        net.train()
        for i in rng.permutation(len(batches)):
            m = masks[i]
            if not m.any():
                continue
            loss = F_.binary_cross_entropy_with_logits(net(batches[i])[m], batches[i].y[m])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        z = infer(net, batches, n)
        v = bce(z[val], y[val])
        print(f"    epoch {ep}: val BCE {v:.5f}  train BCE {bce(z[train], y[train]):.5f}  "
              f"({time.perf_counter() - t:.0f}s)", flush=True)
        if v < best - 1e-5:
            best, best_ep, best_state, best_z = v, ep, copy.deepcopy(net.state_dict()), z
        elif ep - best_ep >= cfg.patience:
            break
    net.load_state_dict(best_state)
    return net, best_ep, best_z


# ------------------------------------------------------------------ train / predict
def frame_inputs(fdir: Path, target_paths, spec: dict, models: dict):
    pairs = pd.read_parquet(fdir / "pairs.parquet")
    parts = []
    for a, b in stack.check_ranges(pairs):
        t = time.perf_counter()
        parts.append(range_inputs(fdir, a, b, pairs.iloc[a:b].reset_index(drop=True), spec, models, target_paths))
        print(f"  [{pairs['country'].iat[a]}] {b - a:,} edges: stack inputs in {time.perf_counter() - t:.0f}s",
              flush=True)
    return pairs, pd.concat(parts, ignore_index=True)


def train(fdir: Path, target_paths, gt: dict, art: Path, tune_decoder, cfg: GNNConfig | None = None):
    """Fold-wise GNN on the graph frame; bootstrap against the stack's level 2 on the same S1."""
    cfg, pcfg = cfg or GNNConfig(), PostConfig()
    torch.set_num_threads(os.cpu_count() or 1)
    art = Path(art)
    spec = json.loads((art / "stack.json").read_text())
    models = stack.load_models(art, spec)
    pairs, Z = frame_inputs(fdir, target_paths, spec, models)
    s1 = pd.read_parquet(fdir / "s1.parquet")
    ids, country = s1["s1_id"].tolist(), s1["country"].to_numpy(object)
    y = pairs["y"].to_numpy(np.float32)
    keys = graph_keys(pairs)
    Z = add_graph_features(Z, *keys)
    scaler = Scaler().fit(Z)
    batches = make_batches(Z, scaler, keys, y, cfg.batch_edges)
    f_in = batches[0].x.shape[1]
    folds = mdl.assign_folds(pairs["s1_id"].to_numpy(object), cfg.n_folds)
    oof = np.zeros(len(pairs), np.float32)
    gdir = art / "gnn"
    gdir.mkdir(parents=True, exist_ok=True)
    epochs = []
    for k in range(cfg.n_folds):
        t = time.perf_counter()
        print(f"  fold {k}", flush=True)
        net, ep, z = fit_fold(batches, folds != k, folds == k, y, f_in, cfg)
        oof[folds == k] = z[folds == k]
        epochs.append(ep)
        torch.save(net.state_dict(), gdir / f"fold{k}.pt")
        print(f"  fold {k}: best epoch {ep} ({time.perf_counter() - t:.0f}s)", flush=True)

    # evaluation: same S1, same decoder tuning; the stack's own (frame-B) decoder for reference
    p_l2 = stack.sigmoid(Z["l2"].to_numpy())
    scored = pairs[["s1_id", "cand_id"]].assign(prob=p_l2)
    kept = select_with(scored, spec["threshold"], spec["exclusive"], spec["decoder"])
    f_b = MacroScorer(scored, gt, ids).score(kept.index)
    print(f"\n  L2 with the stack's decoder ({spec['decoder']} @ {spec['threshold']}): OOF F0.5 {f_b:.4f}")
    r_l2 = stack._decode_eval("L2 stack, decoder tuned on the graph frame", pairs, p_l2, gt, ids, pcfg, tune_decoder)
    r_gnn = stack._decode_eval("GNN (OOF)", pairs, stack.sigmoid(oof), gt, ids, pcfg, tune_decoder)
    lo = stack._compare(r_l2, r_gnn, country)
    for r in (r_l2, r_gnn):
        print(f"  {r['f05']:.4f}  {r['name']}  ({r['decoder']} @ {r['threshold']})")
        for c in pd.unique(country):
            print(f"    [{c}] {r['per'][country == c].mean():.4f}")
    meta = {"cfg": cfg.__dict__, "f_in": f_in, "scaler": scaler.state, "epochs": epochs,
            "folds": [k for k, e in enumerate(epochs) if e > 0], "stack_decoder_f05": f_b, "bootstrap_lo": lo,
            "gnn": {k: r_gnn[k] for k in ("name", "decoder", "threshold", "f05")},
            "l2g": {k: r_l2[k] for k in ("name", "decoder", "threshold", "f05")}}
    (gdir / "gnn.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({"s1_id": ids, "country": country, "f05_l2": r_l2["per"], "f05_gnn": r_gnn["per"]}).to_parquet(
        gdir / "oof_entity_f05_graph.parquet", index=False)
    print(f"saved GNN to {gdir} (fold epochs {epochs}; bootstrap lower bound {lo:+.4f})")
    return meta


def predict(fdir: Path, target_paths, art: Path, use_gnn: bool = True) -> tuple[pd.DataFrame, dict]:
    """GNN probabilities (mean logit of the fold models) for a (test) frame, or level 2 alone."""
    art = Path(art)
    torch.set_num_threads(os.cpu_count() or 1)
    spec = json.loads((art / "stack.json").read_text())
    meta = json.loads((art / "gnn" / "gnn.json").read_text())
    models = stack.load_models(art, spec)
    cfg = GNNConfig(**meta["cfg"])
    scaler = Scaler(meta["scaler"])
    nets = []
    if use_gnn:
        for k in meta["folds"]:  # folds that stopped at epoch 0 are the stack itself
            net = EdgeGNN(meta["f_in"], cfg)
            net.load_state_dict(torch.load(art / "gnn" / f"fold{k}.pt"))
            nets.append(net)
        print(f"  GNN: averaging {len(nets)} fold models (of {len(meta['epochs'])})")
    pairs = pd.read_parquet(fdir / "pairs.parquet", columns=["s1_id", "cand_id", "source", "country", "prob"])
    out = []
    for a, b in stack.check_ranges(pairs):
        t = time.perf_counter()
        P = pairs.iloc[a:b].reset_index(drop=True)
        Z = range_inputs(fdir, a, b, P, spec, models, target_paths)
        z = Z["l2"].to_numpy(np.float32)
        if nets:
            keys = graph_keys(P)
            Z = add_graph_features(Z, *keys)
            batches = make_batches(Z, scaler, keys, None, cfg.batch_edges)
            z = np.mean([infer(net, batches, len(P)) for net in nets], axis=0)
            del batches
        out.append(P[["s1_id", "cand_id"]].assign(prob=stack.sigmoid(z)))
        print(f"  [{P['country'].iat[0]}] {len(P):,} pairs scored in {time.perf_counter() - t:.0f}s", flush=True)
        del Z, P
    dec = dict(meta["gnn"] if use_gnn else meta["l2g"], exclusive=spec["exclusive"])
    return pd.concat(out, ignore_index=True), dec

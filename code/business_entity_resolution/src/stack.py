"""Stacked matcher on the cached frames (see pipeline.frame_dir).

    level 0  the pair features of features.py (+ the cross-encoder logit, when scored)
    level 1  LightGBM, XGBoost (and CatBoost) on level 0, identical GroupKFold folds -> OOF logits
    level 2  LightGBM on level 0 + level-1 logits + relational features built from the level-1
             probabilities of the *other* candidates of the same S1 entity

Relational features use S1-side groups only. The training frame holds a sample of S1 entities,
so target-side competition (other S1s claiming the same record) would look different on the
full test set; blocking.target_best already covers that side with retrieval scores over all S1.

Every level is scored on OOF macro F0.5 with its own tuned decoder, and kept only when the
paired bootstrap of per-entity F0.5 against the level it extends has a lower 95% bound above 0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import model as mdl
from config import ModelConfig, PostConfig, StackConfig
from metrics import paired_bootstrap
from postprocess import MacroScorer, select_with

REC_COLS = ["entity_id", "name_core", "addr_clean", "addr_numbers"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-np.asarray(z, np.float64)))).astype(np.float32)


# ------------------------------------------------------------------ level-1 learners
def _xgb():
    import xgboost as xgb
    return xgb


def xgb_params(cfg: StackConfig, gpu: bool) -> dict:
    return {**cfg.xgb_params, "device": "cuda" if gpu else "cpu"}


def xgb_cv(X, y, folds, cfg: StackConfig, gpu: bool):
    xgb = _xgb()
    oof, iters = np.zeros(len(X)), []
    for k in range(int(folds.max()) + 1):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        dtr = xgb.QuantileDMatrix(X.iloc[tr], y[tr])
        dva = xgb.DMatrix(X.iloc[va], y[va])
        b = xgb.train(xgb_params(cfg, gpu), dtr, cfg.num_boost_round, evals=[(dva, "va")],
                      early_stopping_rounds=cfg.early_stopping_rounds, verbose_eval=False)
        oof[va] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
        iters.append(b.best_iteration + 1)
        print(f"  xgb fold {k}: best_iter={iters[-1]}", flush=True)
        del dtr, dva, b
    return oof, iters


def xgb_fit(X, y, cfg: StackConfig, gpu: bool, rounds: int):
    xgb = _xgb()
    return xgb.train(xgb_params(cfg, gpu), xgb.QuantileDMatrix(X, y), max(rounds, 1))


def xgb_predict(booster, X) -> np.ndarray:
    xgb = _xgb()
    return booster.predict(xgb.DMatrix(X))


def cat_cv(X, y, folds, cfg: StackConfig, gpu: bool):
    from catboost import CatBoostClassifier
    oof, iters = np.zeros(len(X)), []
    for k in range(int(folds.max()) + 1):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        m = CatBoostClassifier(**cfg.cat_params, iterations=cfg.num_boost_round, task_type="GPU" if gpu else "CPU",
                               early_stopping_rounds=cfg.early_stopping_rounds, verbose=False)
        m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        iters.append(m.get_best_iteration() + 1)
        print(f"  cat fold {k}: best_iter={iters[-1]}", flush=True)
    return oof, iters


def cat_fit(X, y, cfg: StackConfig, gpu: bool, rounds: int):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(**cfg.cat_params, iterations=max(rounds, 1), task_type="GPU" if gpu else "CPU",
                           verbose=False)
    return m.fit(X, y)


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ------------------------------------------------------------------ relational features
def _group_order(key: np.ndarray, p: np.ndarray):
    """Rows sorted by (key, p desc) and, per sorted row, the position of its group's first row."""
    order = np.lexsort((-p, key))
    ks = key[order]
    n = len(order)
    start = np.r_[True, ks[1:] != ks[:-1]] if n else np.zeros(0, bool)
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0)) if n else np.zeros(0, np.int64)
    return order, ks, start, first


def _top_two(key: np.ndarray, p: np.ndarray):
    """Per row: rank within its key group (1 = best), best row, second-best row (-1 if none)."""
    n = len(key)
    order, ks, start, first = _group_order(key, p)
    pos = np.arange(n)
    rank = np.empty(n, np.int32)
    rank[order] = pos - first + 1
    best = np.empty(n, np.int64)
    best[order] = order[first]
    sp = np.minimum(first + 1, max(n - 1, 0))
    has2 = (first + 1 < n) & (ks[sp] == ks)
    second = np.empty(n, np.int64)
    second[order] = np.where(has2, order[sp], -1)
    heads = order[start]
    return rank, best, second, ks[start], heads


class Records:
    """name_core / addr_clean / first house number of target records, looked up by entity id."""

    def __init__(self, paths, ids):
        need = pd.Index(pd.unique(np.asarray(ids, dtype=object)))
        parts = []
        for path in paths:
            df = pd.read_parquet(path, columns=REC_COLS)
            parts.append(df[df["entity_id"].isin(need)])
        df = pd.concat(parts, ignore_index=True)
        self.index = pd.Index(df["entity_id"].to_numpy(object))
        self.name = df["name_core"].to_numpy(object)
        self.addr = df["addr_clean"].to_numpy(object)
        self.num = df["addr_numbers"].str.split(" ", n=1).str[0].to_numpy(object)

    def rows(self, ids) -> np.ndarray:
        r = self.index.get_indexer(np.asarray(ids, dtype=object))
        if (r < 0).any():
            raise ValueError(f"{(r < 0).sum()} candidate ids missing from the record caches")
        return r


def _pair_sims(rec: Records, ra: np.ndarray, rb: np.ndarray, valid: np.ndarray, prefix: str, F: dict):
    n = len(ra)
    for k in ("name", "addr", "num_eq"):
        F[f"{prefix}_{k}"] = np.full(n, np.nan, np.float32)
    v = np.flatnonzero(valid)
    if len(v) == 0:
        return
    a, b = ra[v], rb[v]
    F[f"{prefix}_name"][v] = cpdist(rec.name[a], rec.name[b], scorer=fuzz.token_set_ratio, workers=-1,
                                    dtype=np.float32)
    F[f"{prefix}_addr"][v] = cpdist(rec.addr[a], rec.addr[b], scorer=fuzz.token_set_ratio, workers=-1,
                                    dtype=np.float32)
    na, nb = rec.num[a], rec.num[b]
    F[f"{prefix}_num_eq"][v] = np.where((na != "") & (nb != ""), (na == nb).astype(np.float32), np.nan)


def relational(pairs: pd.DataFrame, p: np.ndarray, first_num_equal: np.ndarray, rec: Records,
               prefix: str = "r") -> pd.DataFrame:
    """Features of a pair given the level-1 probabilities of its S1's other candidates.

    - where the pair stands among its S1's candidates (rank, gap to the best, how many are likely)
    - twin: the best *other* candidate of the same source -- a near-identical sibling with a
      higher probability is the typical false merge (same name, house number changed)
    - anchor: the S1's best candidate in the *other* source -- a true match from S2 usually looks
      like the true match from S3, which rescues garbled names / empty addresses
    """
    s1 = pd.factorize(pairs["s1_id"].to_numpy(object))[0].astype(np.int64)
    src3 = (pairs["source"].to_numpy() == 3).astype(np.int64)
    p = np.asarray(p, np.float64)
    n = len(p)
    F = {}

    rank_s1, best_s1, second_s1, _, _ = _top_two(s1, p)
    ps = pd.Series(p)
    F["r_rank_s1"] = rank_s1.astype(np.float32)
    F["r_pmax_s1"] = p[best_s1].astype(np.float32)
    F["r_p2_s1"] = np.where(second_s1 >= 0, p[np.maximum(second_s1, 0)], 0).astype(np.float32)
    F["r_psum_s1"] = ps.groupby(s1).transform("sum").to_numpy(np.float32)
    F["r_nhi_s1"] = (ps > 0.5).groupby(s1).transform("sum").to_numpy(np.float32)
    F["r_gap_s1"] = (F["r_pmax_s1"] - p).astype(np.float32)

    key = s1 * 2 + src3
    rank_g, best_g, second_g, heads_k, heads_row = _top_two(key, p)
    F["r_rank_src"] = rank_g.astype(np.float32)
    F["r_gap_src"] = (p[best_g] - p).astype(np.float32)
    F["r_n_src"] = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy(np.float32)

    rows = rec.rows(pairs["cand_id"].to_numpy(object))
    # twin: best other candidate of the same (S1, source)
    twin = np.where(rank_g == 1, second_g, best_g)
    has_twin = twin >= 0
    tw = np.maximum(twin, 0)
    F["r_twin_p"] = np.where(has_twin, p[tw], np.nan).astype(np.float32)
    F["r_twin_higher"] = np.where(has_twin, (p[tw] > p).astype(np.float32), np.nan)
    _pair_sims(rec, rows, rows[tw], has_twin, "r_twin", F)
    F["r_twin_numeq_s1"] = np.where(has_twin, first_num_equal[tw], np.nan).astype(np.float32)

    # anchor: best candidate of the other source
    other = key ^ 1
    i = np.searchsorted(heads_k, other)
    ic = np.minimum(i, max(len(heads_k) - 1, 0))
    has_anc = (i < len(heads_k)) & (heads_k[ic] == other) if len(heads_k) else np.zeros(n, bool)
    anc = np.where(has_anc, heads_row[ic], 0)
    F["r_anc_p"] = np.where(has_anc, p[anc], np.nan).astype(np.float32)
    _pair_sims(rec, rows, rows[anc], has_anc, "r_anc", F)
    F["r_anc_x_name"] = F["r_anc_p"] * F["r_anc_name"] / 100
    F["r_anc_x_addr"] = F["r_anc_p"] * F["r_anc_addr"] / 100
    return pd.DataFrame({prefix + k[1:]: v for k, v in F.items()})


def level3_input(Z: pd.DataFrame, p2: np.ndarray, pairs: pd.DataFrame, X: pd.DataFrame, rec: Records) -> pd.DataFrame:
    R3 = relational(pairs, p2, X["first_num_equal"].to_numpy(np.float32), rec, prefix="r3")
    return pd.concat([Z, pd.DataFrame({"l2": logit(p2)}), R3], axis=1)


# ------------------------------------------------------------------ training
def _decode_eval(name, pairs, prob, gt, ids, pcfg, tune_decoder):
    scored = pairs[["s1_id", "cand_id"]].assign(prob=prob)
    print(f"\n== {name}", flush=True)
    decoder, thr, f = tune_decoder(scored, gt, ids, pcfg)
    kept = select_with(scored, thr, pcfg.exclusive_candidates, decoder)
    return {"name": name, "decoder": decoder, "threshold": thr, "f05": f,
            "per": MacroScorer(scored, gt, ids).per_entity(kept.index)}


def _compare(a: dict, b: dict, country: np.ndarray) -> float:
    d, lo, hi = paired_bootstrap(a["per"], b["per"])
    print(f"  {b['name']} vs {a['name']}: dF0.5 = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  -> {'KEEP' if lo > 0 else 'not significant'}", flush=True)
    worst = 0.0
    for c in pd.unique(country):
        m = country == c
        dc = float((b["per"][m] - a["per"][m]).mean())
        worst = min(worst, dc)
        print(f"    [{c}] dF0.5 = {dc:+.4f} (n={m.sum():,})")
    return lo if worst >= 0 else -1.0


def train(fdir: Path, target_paths, gt: dict, art: Path, tune_decoder, scfg: StackConfig | None = None):
    """Fit all levels on the training frame, print the ablation, save models + stack.json."""
    scfg, mcfg, pcfg = scfg or StackConfig(), ModelConfig(), PostConfig()
    gpu = has_gpu()
    art = Path(art)
    art.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(fdir / "pairs.parquet")
    X = pd.read_parquet(fdir / "X.parquet")
    s1 = pd.read_parquet(fdir / "s1.parquet")
    ids, country = s1["s1_id"].tolist(), s1["country"].to_numpy(object)
    y, folds = pairs["y"].to_numpy(np.int8), pairs["fold"].to_numpy()
    has_ce = (fdir / "ce.parquet").exists()
    if has_ce:
        X["ce_logit"] = pd.read_parquet(fdir / "ce.parquet")["ce_logit"].to_numpy(np.float32)
    print(f"stack: {len(pairs):,} pairs, {X.shape[1]} level-0 features (cross-encoder: {has_ce}), "
          f"gpu={gpu}", flush=True)

    rows = [_decode_eval("L0 LightGBM (run-1 model, OOF)", pairs, pairs["prob"].to_numpy(), gt, ids, pcfg,
                         tune_decoder)]
    spec = {"features": list(X.columns), "has_ce": has_ce}

    # level 1. Without the cross-encoder, LightGBM on level 0 *is* the run-1 model: its OOF
    # probabilities are in the frame, and its refit predictions are in the test frame.
    l1_oof, l1_models = {}, {}
    t = time.perf_counter()
    if has_ce:
        oof, it = mdl.train_cv(X, y, folds, mcfg)
        l1_oof["lgb"] = oof
        l1_models["lgb"] = mdl.fit(X, y, mcfg, int(np.mean(it) * 1.1))
        rows.append(_decode_eval("L1 LightGBM + cross-encoder", pairs, oof, gt, ids, pcfg, tune_decoder))
    else:
        l1_oof["lgb"] = pairs["prob"].to_numpy()
    if scfg.use_xgb:
        oof, xit = xgb_cv(X, y, folds, scfg, gpu)
        l1_oof["xgb"] = oof
        l1_models["xgb"] = xgb_fit(X, y, scfg, gpu, int(np.mean(xit) * 1.1))
    if scfg.use_cat:
        oof, cit = cat_cv(X, y, folds, scfg, gpu)
        l1_oof["cat"] = oof
        l1_models["cat"] = cat_fit(X, y, scfg, gpu, int(np.mean(cit) * 1.1))
    print(f"  level 1 done in {time.perf_counter() - t:.0f}s", flush=True)
    names = list(l1_oof)
    L = np.column_stack([logit(l1_oof[k]) for k in names])
    p_mean = sigmoid(L.mean(axis=1))
    if len(names) > 1:
        rows.append(_decode_eval(f"L1 mean of {'+'.join(names)}", pairs, p_mean, gt, ids, pcfg, tune_decoder))

    # level 2
    rec = Records(target_paths, pairs["cand_id"])
    R = relational(pairs, p_mean, X["first_num_equal"].to_numpy(np.float32), rec)
    Z = pd.concat([X, pd.DataFrame(L, columns=[f"l1_{k}" for k in names]), R], axis=1)
    oof2, it2 = mdl.train_cv(Z, y, folds, mcfg)
    rows.append(_decode_eval("L2 stack (+ level-1 logits + relational)", pairs, oof2, gt, ids, pcfg, tune_decoder))
    l2 = mdl.fit(Z, y, mcfg, int(np.mean(it2) * 1.1))

    # level 3: one more relational round, from the (sharper) level-2 probabilities
    Z3 = level3_input(Z, oof2, pairs, X, rec)
    oof3, it3 = mdl.train_cv(Z3, y, folds, mcfg)
    rows.append(_decode_eval("L3 second relational round", pairs, oof3, gt, ids, pcfg, tune_decoder))
    l3 = mdl.fit(Z3, y, mcfg, int(np.mean(it3) * 1.1))

    # choose: walk down the ablation, keep a level only if it beats the one kept so far
    print("\n== ablation (OOF macro F0.5)")
    for r in rows:
        print(f"  {r['f05']:.4f}  {r['name']}  ({r['decoder']} @ {r['threshold']})")
    chosen = 0
    for i in range(1, len(rows)):
        if _compare(rows[chosen], rows[i], country) > 0:
            chosen = i
    final = rows[chosen]
    print(f"\n  chosen: {final['name']}  OOF F0.5 {final['f05']:.4f}", flush=True)
    for c in pd.unique(country):
        print(f"    [{c}] {final['per'][country == c].mean():.4f}")

    # persist
    name = final["name"]
    level = ("l0" if chosen == 0 else "l3" if name.startswith("L3") else "l2" if name.startswith("L2")
             else "l1_mean" if "mean" in name else "l1_lgb")
    if l1_models.get("lgb") is not None:
        l1_models["lgb"].save_model(str(art / "stack_l1_lgb.txt"))
    if "xgb" in l1_models:
        l1_models["xgb"].save_model(str(art / "stack_l1_xgb.json"))
    if "cat" in l1_models:
        l1_models["cat"].save_model(str(art / "stack_l1_cat.cbm"))
    l2.save_model(str(art / "stack_l2.txt"))
    l3.save_model(str(art / "stack_l3.txt"))
    spec.update({"level": level, "l1": names, "l2_features": list(Z.columns), "l3_features": list(Z3.columns),
                 "decoder": final["decoder"],
                 "threshold": final["threshold"], "exclusive": pcfg.exclusive_candidates,
                 "oof_macro_f05": final["f05"],
                 "ablation": [{k: r[k] for k in ("name", "decoder", "threshold", "f05")} for r in rows]})
    (art / "stack.json").write_text(json.dumps(spec, indent=2))
    pd.DataFrame({"s1_id": ids, "country": country, "f05": final["per"]}).to_parquet(
        art / "oof_entity_f05_stack.parquet", index=False)
    print(f"saved stack ({level}) to {art}")
    return spec


# ------------------------------------------------------------------ prediction
def read_rows(path: Path, a: int, b: int, columns=None) -> pd.DataFrame:
    """Rows [a, b) of a parquet file, reading only the row groups that overlap them."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    tables, off, first = [], 0, None
    for g in range(f.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        if off + n > a and off < b:
            first = off if first is None else first
            tables.append(f.read_row_group(g, columns=columns))
        off += n
    t = pa.concat_tables(tables).slice(a - first, b - a)
    return t.to_pandas().reset_index(drop=True)


def country_ranges(country: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous row ranges per country (frames are generated one country at a time, and an S1
    entity's candidates never leave its country, so each range holds whole S1 groups)."""
    c = pd.Series(country).astype(str).str.strip().str.lower().to_numpy(object)
    cuts = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True]) if len(c) else np.array([0])
    return list(zip(cuts[:-1].tolist(), cuts[1:].tolist()))


def predict(fdir: Path, target_paths, art: Path) -> tuple[pd.DataFrame, dict]:
    """Probabilities of the chosen stack level for every pair of a (test) frame."""
    art = Path(art)
    spec = json.loads((art / "stack.json").read_text())
    level = spec["level"]
    pairs = pd.read_parquet(fdir / "pairs.parquet", columns=["s1_id", "cand_id", "source", "country", "prob"])
    if level == "l0":
        return pairs[["s1_id", "cand_id", "prob"]], spec
    lgb_m = lgb.Booster(model_file=str(art / "stack_l1_lgb.txt")) if spec["has_ce"] else None
    xgb_m = None
    if "xgb" in spec["l1"]:
        xgb_m = _xgb().Booster()
        xgb_m.load_model(str(art / "stack_l1_xgb.json"))
    cat_m = None
    if "cat" in spec["l1"]:
        from catboost import CatBoostClassifier
        cat_m = CatBoostClassifier().load_model(str(art / "stack_l1_cat.cbm"))
    l2 = lgb.Booster(model_file=str(art / "stack_l2.txt")) if level in ("l2", "l3") else None
    l3 = lgb.Booster(model_file=str(art / "stack_l3.txt")) if level == "l3" else None

    ranges = country_ranges(pairs["country"].to_numpy(object))
    rid = np.repeat(np.arange(len(ranges)), [b - a for a, b in ranges])
    if pd.Series(rid).groupby(pairs["s1_id"].to_numpy(object)).nunique().max() > 1:
        raise ValueError("an S1 entity's candidates span several country ranges")
    out = []
    for a, b in ranges:
        t = time.perf_counter()
        P = pairs.iloc[a:b].reset_index(drop=True)
        X = read_rows(fdir / "X.parquet", a, b)
        if spec["has_ce"]:
            X["ce_logit"] = read_rows(fdir / "ce.parquet", a, b)["ce_logit"].to_numpy(np.float32)
        X = X[spec["features"]]
        L = {"lgb": logit(lgb_m.predict(X)) if lgb_m is not None else logit(P["prob"].to_numpy())}
        if xgb_m is not None:
            L["xgb"] = logit(xgb_predict(xgb_m, X))
        if cat_m is not None:
            L["cat"] = logit(cat_m.predict_proba(X)[:, 1])
        Lm = np.column_stack([L[k] for k in spec["l1"]])
        if level == "l1_lgb":
            prob = sigmoid(L["lgb"])
        elif level == "l1_mean":
            prob = sigmoid(Lm.mean(axis=1))
        else:
            rec = Records(target_paths, P["cand_id"])
            R = relational(P, sigmoid(Lm.mean(axis=1)), X["first_num_equal"].to_numpy(np.float32), rec)
            Z = pd.concat([X, pd.DataFrame(Lm, columns=[f"l1_{k}" for k in spec["l1"]]), R], axis=1)
            prob = l2.predict(Z[spec["l2_features"]])
            if level == "l3":
                prob = l3.predict(level3_input(Z, prob, P, X, rec)[spec["l3_features"]])
            del rec, R, Z
        out.append(P[["s1_id", "cand_id"]].assign(prob=np.asarray(prob, np.float32)))
        print(f"  [{P['country'].iat[0]}] {len(P):,} pairs scored ({level}) in {time.perf_counter() - t:.0f}s",
              flush=True)
        del X, L, Lm, P
    return pd.concat(out, ignore_index=True), spec

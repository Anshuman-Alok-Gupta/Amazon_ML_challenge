"""End-to-end CLI.

    python src/pipeline.py eda        # dataset statistics
    python src/pipeline.py train      # sampled S1 -> blocking report, CV F0.5, threshold, final model
    python src/pipeline.py validate   # leave-one-country-out robustness check
    python src/pipeline.py predict    # full test set -> output/*.tsv
    python src/pipeline.py all        # train + predict
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import threading
import time
from pathlib import Path

# Arrow's default allocator keeps freed memory cached; the system allocator returns it to the OS,
# which matters when each country's data is loaded and dropped in turn on a 16 GB machine.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import lightgbm as lgb
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda  # noqa: E402
import model as mdl  # noqa: E402
import normalize  # noqa: E402
from blocking import (BlockText, TokenIndex, keep_top, make_blocks, make_target, pool,  # noqa: E402
                      pool_features, stage1_features, target_best)
from config import (DEFAULT_ARTIFACT_DIR, DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR,  # noqa: E402
                    DEFAULT_OUTPUT_DIR, SEED, BlockingConfig, ModelConfig, PostConfig, StackConfig,
                    TrainConfig)
from features import build_features  # noqa: E402
from io_utils import (CAND_HEADER, MATCH_HEADER, check_pairs, gt_dict, load_split,  # noqa: E402
                      read_ground_truth, read_source, write_pairs)  # noqa: E402
from metrics import blocking_report, macro_f05, paired_bootstrap, per_entity_f05  # noqa: E402
from postprocess import select_with, to_lists, tune_expected_f, tune_threshold  # noqa: E402
from regions import drop_regions, infer_regions, rare_regions, region_freq, resolve_regions  # noqa: E402

PAIR_COLS = ["s1_id", "cand_id", "source", "country"]


def release_memory():
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"[{self.label}] ...", flush=True)

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.perf_counter() - self.t:.1f}s", flush=True)


# ------------------------------------------------------------------ data
FEATURE_COLS = ["entity_id", "country", "name_clean", "name_core", "name_legal", "name_nonlatin", "name_alt",
                "addr_clean", "addr_numbers", "postal", "addr_landmark", "addr_nonlatin", "region"]


def _norm_tag() -> str:
    """Cache key: normalised data is rebuilt whenever normalize.py changes."""
    return hashlib.md5(Path(normalize.__file__).read_bytes()).hexdigest()[:8]


class SplitData:
    """Normalised records of one split, cached as parquet and loaded one country at a time.

    Holding a whole split (~12M records) in memory at once does not fit on a 16 GB machine;
    blocking never crosses countries, so a country is the natural unit of work.
    """

    def __init__(self, data_dir: Path, split: str, cache_dir: Path, subset: float | None = None):
        """subset: keep only a seeded random fraction of each country's regions (dev mode).

        All S1 and target records of the kept regions are kept (so competition inside a region
        stays realistic), plus every region-less target (they may match any kept S1).
        """
        self.split, tag = split, _norm_tag()
        d = Path(data_dir) / split
        self.paths = {}
        for k in ("s1", "s2", "s3"):
            path = Path(cache_dir) / f"{split}_{k}_{tag}.parquet"
            if not path.exists():
                with Timer(f"normalize {split} {k}"):
                    n = normalize.normalize_tsv(d / f"{split}_source{k[1]}.tsv", path)
                print(f"  {n:,} rows -> {path.name}", flush=True)
                release_memory()
            self.paths[k] = path
        gt_path = d / f"{split}_ground_truth.tsv"
        self.gt = read_ground_truth(gt_path) if gt_path.exists() else None
        s1 = pd.read_parquet(self.paths["s1"], columns=["entity_id", "country", "country_norm", "region"])
        # data-driven, per country label (see regions.py): ambiguous region candidates are resolved
        # by the country's own region frequencies, and regions too rare to be real are dropped
        self.freq = region_freq(s1["country_norm"], s1["region"])
        s1["region"] = self._resolve(s1)
        self.rare = rare_regions(s1["country_norm"], s1["region"])
        s1["region"] = s1["region"].where(
            ~pd.Series([r in self.rare.get(c, ()) for c, r in zip(s1["country_norm"], s1["region"])],
                       index=s1.index), "")
        self.keep_regions = None
        if subset:
            rng = np.random.default_rng(SEED)
            self.keep_regions = {}
            for c, regs in s1.groupby("country_norm")["region"]:
                pool = sorted(set(regs) - {""})
                n = max(1, int(round(subset * len(pool))))
                self.keep_regions[c] = set(rng.choice(pool, n, replace=False)) if pool else set()
            s1 = s1[self._mask(s1, is_target=False)]
            print(f"  subset {subset}: {len(s1):,} S1 in "
                  f"{sum(len(v) for v in self.keep_regions.values())} regions", flush=True)
        self.s1_ids = s1["entity_id"].to_numpy()
        self.s1_country = s1["country"].to_numpy()
        self.countries = sorted(s1["country_norm"].unique())
        self.hidden: set = set()

    def hide(self, ids):
        """Drop these S1 entities from every later step (their targets become orphans)."""
        self.hidden = set(ids)
        keep = ~pd.Index(self.s1_ids).isin(self.hidden)
        self.s1_ids, self.s1_country = self.s1_ids[keep], self.s1_country[keep]

    def _resolve(self, df: pd.DataFrame, country: str | None = None) -> pd.Series:
        """Region column with multi-candidate values resolved (per row country, or `country`)."""
        if country is not None:
            return resolve_regions(df["region"], self.freq.get(country, {}))
        out = df["region"].copy()
        for c, idx in df.groupby("country_norm").groups.items():
            out.loc[idx] = resolve_regions(df.loc[idx, "region"], self.freq.get(c, {}))
        return out

    def _mask(self, df: pd.DataFrame, is_target: bool) -> np.ndarray:
        """Rows kept by the dev subset (all rows when no subset is set)."""
        if self.keep_regions is None:
            return np.ones(len(df), bool)
        reg = df["region"].to_numpy()
        cty = df["country_norm"].to_numpy() if "country_norm" in df else None
        keep = np.zeros(len(df), bool)
        for c, regs in self.keep_regions.items():
            m = np.isin(reg, list(regs))
            keep |= m if cty is None else (m & (cty == c))
        if is_target:
            keep |= reg == ""
        return keep

    def target_ids(self) -> set:
        cols = ["entity_id", "country_norm", "region"]
        out = []
        for k in ("s2", "s3"):
            t = pd.read_parquet(self.paths[k], columns=cols)
            if self.keep_regions is not None:  # dev subset: match regions the way load() does
                t["region"] = self._resolve(t)
            out.append(t["entity_id"].to_numpy()[self._mask(t, is_target=True)])
        return set(np.concatenate(out))

    def load(self, country: str, cols=FEATURE_COLS):
        f = [("country_norm", "==", country)]
        s1 = pd.read_parquet(self.paths["s1"], columns=cols, filters=f)
        if self.hidden:
            s1 = s1[~s1["entity_id"].isin(self.hidden)].reset_index(drop=True)
        tgt = make_target(pd.read_parquet(self.paths["s2"], columns=cols, filters=f),
                          pd.read_parquet(self.paths["s3"], columns=cols, filters=f))
        if "region" in s1.columns:
            s1["region"], tgt["region"] = self._resolve(s1, country), self._resolve(tgt, country)
            bad = self.rare.get(country, set())
            s1, tgt = drop_regions(s1, bad), drop_regions(tgt, bad)
            n_empty = int((tgt["region"] == "").sum())
            s1, tgt, n_filled = infer_regions(s1, tgt)
            reg = s1["region"].value_counts()
            print(f"  {country}: {len(reg)} regions ({', '.join(f'{r}={n:,}' for r, n in reg.items() if r)}); "
                  f"dropped rare {sorted(bad) if bad else '-'}; region-less targets "
                  f"{n_empty:,} -> {int((tgt['region'] == '').sum()):,} (inferred {n_filled:,} records)",
                  flush=True)
        if self.keep_regions is not None:
            regs = list(self.keep_regions.get(country, ()))
            s1 = s1[np.isin(s1["region"].to_numpy(), regs)].reset_index(drop=True)
            t_reg = tgt["region"].to_numpy()
            tgt = tgt[np.isin(t_reg, regs) | (t_reg == "")].reset_index(drop=True)
        return s1, tgt

    def load_all(self, cols=None):
        return pd.read_parquet(self.paths["s1"], columns=cols), make_target(
            pd.read_parquet(self.paths["s2"], columns=cols), pd.read_parquet(self.paths["s3"], columns=cols))


# ------------------------------------------------------------------ candidates + features
STAGE1_PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 100,
                 "feature_fraction": 0.9, "verbose": -1, "seed": SEED, "num_threads": 0}


def _s1_chunks(l1: np.ndarray, chunk_pairs: int):
    """Chunk boundaries that never split one S1's candidates (pairs are sorted by l1)."""
    starts = np.flatnonzero(np.r_[True, l1[1:] != l1[:-1]])
    cuts = [0]
    for s in starts:
        if s - cuts[-1] >= chunk_pairs:
            cuts.append(s)
    cuts.append(len(l1))
    return list(zip(cuts[:-1], cuts[1:]))


def _blocks(data: SplitData, bcfg: BlockingConfig, query_ids):
    """Yield (block name, s1 frame, target frame, queried S1 rows, TokenIndex, BlockText, name_freq).

    A country's records are loaded once and split into region blocks (see make_blocks); each
    block gets its own small index, so memory stays bounded and decoys from other regions never
    enter the candidate pools.
    """
    for c in data.countries:
        s1_all, tgt_all = data.load(c)
        name_freq = (s1_all["name_core"].value_counts(), tgt_all["name_core"].value_counts())
        queried = (np.ones(len(s1_all), bool) if query_ids is None
                   else pd.Index(s1_all["entity_id"].array).isin(query_ids))
        blocks = make_blocks(s1_all["region"].to_numpy(), tgt_all["region"].to_numpy())
        print(f"  {c}: s1={len(s1_all):,} tgt={len(tgt_all):,} queried={queried.sum():,} "
              f"blocks={len(blocks)}", flush=True)
        for name, i1, it in blocks:
            q = np.flatnonzero(queried[i1])
            if len(q) == 0:
                continue
            s1_b = s1_all.iloc[i1].reset_index(drop=True)
            tgt_b = tgt_all.iloc[it].reset_index(drop=True)
            idx = TokenIndex(s1_b, tgt_b, bcfg)
            yield f"{c}/{name}", s1_b, tgt_b, q, idx, BlockText(s1_b, tgt_b), name_freq
            del idx, s1_b, tgt_b
            gc.collect()
        del s1_all, tgt_all
        release_memory()


def stage1_pools(data: SplitData, bcfg: BlockingConfig, query_ids, gt: dict):
    """Retrieval pools with stage-1 features and labels, used to train the re-ranker."""
    feats, ys = [], []
    with Timer("stage-1 pools"):
        for name, s1_b, tgt_b, q, idx, text, _ in _blocks(data, bcfg, query_ids):
            src = tgt_b["source"].to_numpy()
            for s in range(0, len(q), bcfg.query_chunk):
                p = pool_features(pool(idx, src, q[s:s + bcfg.query_chunk], bcfg), idx, text)
                sid = s1_b["entity_id"].to_numpy()[p["l1"].to_numpy()]
                cid = tgt_b["entity_id"].to_numpy()[p["lt"].to_numpy()]
                feats.append(stage1_features(p))
                ys.append(np.array([t in gt[a] for a, t in zip(sid, cid)], dtype=np.int8))
    return pd.concat(feats, ignore_index=True), np.concatenate(ys)


def generate(data: SplitData, bcfg: BlockingConfig, query_ids, stage1, on_chunk, chunk_pairs: int = 1_000_000):
    """Retrieval -> stage-1 re-rank -> candidates -> features, one block at a time.

    Calls on_chunk(pairs, X) per S1-aligned chunk. query_ids: S1 ids to produce candidates for
    (None = all). Retrieval searches the block's full target set, and target-side competition
    (t_best) uses all S1 records of the block, whether queried or not.
    """
    total = n_pool = n_q = 0
    t0 = time.perf_counter()
    for name, s1_b, tgt_b, q, idx, text, name_freq in _blocks(data, bcfg, query_ids):
        src = tgt_b["source"].to_numpy()
        kept = []
        for s in range(0, len(q), bcfg.query_chunk):
            p = pool_features(pool(idx, src, q[s:s + bcfg.query_chunk], bcfg), idx, text)
            n_pool += len(p)
            if len(p):  # tiny blocks can retrieve nothing
                kept.append(keep_top(p, stage1.predict(stage1_features(p)), bcfg))
            del p
        if not kept:
            continue
        pairs = pd.concat(kept, ignore_index=True)
        del kept
        if len(pairs) == 0:
            continue
        ut, inv = np.unique(pairs["lt"].to_numpy(), return_inverse=True)
        best, second = target_best(idx, ut, bcfg)
        pairs["t_best"], pairs["t_second"] = best[inv], second[inv]
        idx.drop_views()
        pairs["s1_id"] = s1_b["entity_id"].iloc[pairs["l1"].to_numpy()].reset_index(drop=True)
        pairs["cand_id"] = tgt_b["entity_id"].iloc[pairs["lt"].to_numpy()].reset_index(drop=True)
        pairs["country"] = s1_b["country"].iloc[pairs["l1"].to_numpy()].reset_index(drop=True)
        for a, b in _s1_chunks(pairs["l1"].to_numpy(), chunk_pairs):
            chunk = pairs.iloc[a:b].reset_index(drop=True)
            on_chunk(chunk[PAIR_COLS], build_features(chunk, s1_b, tgt_b, idx, name_freq))
        total += len(pairs)
        n_q += len(q)
        if len(q) >= 20_000 or name.endswith("~residual"):
            print(f"  [{name}] s1={len(s1_b):,} tgt={len(tgt_b):,} queried={len(q):,} -> "
                  f"{len(pairs):,} candidates  (total {total:,}, {time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  candidates: {total:,} for {n_q:,} S1 queries (pool {n_pool:,})", flush=True)
    return total


def frame_dir(args, name: str) -> Path:
    """Cached model-input frames (train = sample B, ce = sample C, test) live under the cache dir."""
    return Path(args.cache_dir) / "frames" / name


class FrameWriter:
    """Streams (pairs, X) chunks of a split into pairs.parquet / X.parquet (same row order)."""

    def __init__(self, d: Path):
        self.d, self.w = Path(d), {}

    def write(self, pairs: pd.DataFrame, X: pd.DataFrame):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for name, df in (("pairs", pairs), ("X", X)):
            t = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
            if name not in self.w:
                self.w[name] = (pq.ParquetWriter(self.d / f"{name}.parquet", t.schema), t.schema)
            self.w[name][0].write_table(t.cast(self.w[name][1]))

    def close(self):
        for w, _ in self.w.values():
            w.close()


def labels(pairs: pd.DataFrame, gt: dict) -> np.ndarray:
    return np.array([c in gt.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])], dtype=np.int8)


def id_lists(pairs: pd.DataFrame) -> dict[str, list[str]]:
    return pairs.groupby("s1_id", sort=False)["cand_id"].apply(list).to_dict()


def hide_fraction(args) -> float:
    """--hide-s1: fraction of train S1 to hide so train has test's targets-per-S1 density.

    'auto' measures it from the row counts: h = 1 - (train targets/S1) / (test targets/S1).
    """
    v = str(getattr(args, "hide_s1", "0"))
    if v != "auto":
        return float(v)
    SplitData(args.data_dir, "test", args.cache_dir)  # builds the test caches if missing
    rate = {}
    for split in ("train", "test"):
        n = {k: pq.ParquetFile(p).metadata.num_rows for k, p in cache_paths(args, split).items()}
        rate[split] = (n["s2"] + n["s3"]) / n["s1"]
    h = float(np.clip(1 - rate["train"] / rate["test"], 0.0, 0.5))
    print(f"  hide-s1 auto: targets per S1 train {rate['train']:.3f}, test {rate['test']:.3f} -> h={h:.4f}")
    return h


def train_frame(args):
    """Stage 1 on sample A; candidates, features and labels for a disjoint sample B.

    Also returns the candidate pairs (with labels) of a third disjoint sample C, the training set
    of the cross-encoder: its scores on B are then out-of-sample, like stage 1's.
    """
    bcfg, tcfg = BlockingConfig(), TrainConfig()
    data = SplitData(args.data_dir, "train", args.cache_dir, subset=args.subset)
    n_b = args.sample or tcfg.n_s1_sample
    n_c = tcfg.n_ce_sample if args.ce_sample is None else args.ce_sample
    n_a = min(tcfg.n_stage1_sample, max(len(data.s1_ids) - n_b - n_c, len(data.s1_ids) // 4))
    perm = np.random.default_rng(SEED).permutation(data.s1_ids)
    ids_a, ids_b = np.sort(perm[:n_a]), np.sort(perm[n_a:n_a + n_b])
    ids_c = np.sort(perm[n_a + n_b:n_a + n_b + n_c])
    h = hide_fraction(args)
    if h > 0:
        # hidden S1 come from outside A/B/C, so the samples (and the trained CE) are unchanged
        start = n_a + n_b + n_c
        hidden = perm[start:start + int(round(h * len(perm)))]
        n_s1 = len(perm)
        data.hide(hidden)
        n_tgt = sum(pq.ParquetFile(v).metadata.num_rows for k, v in data.paths.items() if k != "s1")
        print(f"  hide-s1: h={h:.4f}, hid {len(hidden):,} of {n_s1:,} train S1 -> "
              f"{n_tgt / len(data.s1_ids):.2f} targets per S1 (was {n_tgt / n_s1:.2f})", flush=True)
    gt = gt_dict(data.gt, np.concatenate([ids_a, ids_b, ids_c]))
    n_targets = len(data.target_ids())
    countries = pd.Series(data.s1_country, index=data.s1_ids).loc[ids_b]
    del data.gt

    # stage 1: re-ranker trained on sample A's retrieval pools (samples are seeded, so a saved
    # stage-1 model can be reused with --reuse-stage1 when only stage 2 changes)
    s1_path = Path(args.artifact_dir) / "stage1.txt"
    if getattr(args, "reuse_stage1", False) and s1_path.exists():
        stage1 = lgb.Booster(model_file=str(s1_path))
        print(f"  reusing stage-1 model {s1_path}")
    else:
        F1, y1 = stage1_pools(data, bcfg, set(ids_a), gt)
        true_a = sum(len(gt[s]) for s in ids_a)
        print(f"  stage-1 sample A: {len(ids_a):,} S1, pool {len(F1) / len(ids_a):.1f} per S1, "
              f"pool recall {y1.sum() / max(true_a, 1):.4f}")
        with Timer("stage-1 fit"):
            stage1 = lgb.train(STAGE1_PARAMS, lgb.Dataset(F1, y1), tcfg.stage1_rounds)
        del F1, y1
        gc.collect()
        s1_path.parent.mkdir(parents=True, exist_ok=True)
        stage1.save_model(str(s1_path))

    # stage 2 data: sample B, candidates chosen by the (out-of-sample) stage-1 model; sample C's
    # candidates come from the same pass (features are per pair / per S1, so B is unaffected)
    set_c = set(ids_c)
    parts_p, parts_x, parts_c = [], [], []

    def keep(p, x):
        in_c = p["s1_id"].isin(set_c).to_numpy()
        if in_c.any():
            parts_c.append(p[in_c])
        if not in_c.all():
            parts_p.append(p[~in_c])
            parts_x.append(x[~in_c])

    generate(data, bcfg, set(ids_b) | set_c, stage1, keep)
    pairs, X = pd.concat(parts_p, ignore_index=True), pd.concat(parts_x, ignore_index=True)
    del parts_p, parts_x
    pairs_c = (pd.concat(parts_c, ignore_index=True) if parts_c
               else pd.DataFrame(columns=PAIR_COLS))
    pairs_c["y"] = labels(pairs_c, {s: gt[s] for s in ids_c})
    print(f"  sample C (cross-encoder): {len(ids_c):,} S1, {len(pairs_c):,} pairs, "
          f"positives {pairs_c['y'].mean() if len(pairs_c) else 0:.3f}")
    gt_b = {s: gt[s] for s in ids_b}
    y = labels(pairs, gt_b)
    rep = blocking_report(id_lists(pairs), gt_b, n_targets, ids_b)
    print("  blocking (sample B): " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                for k, v in rep.items()))
    return stage1, pairs, X, y, gt_b, countries, rep, pairs_c


def tune_decoder(scored, gt, s1_ids, pcfg: PostConfig, prob_col: str = "prob"):
    """Tune both decoders on OOF probabilities; keep the better one: (decoder, threshold, f05)."""
    thr, f = tune_threshold(scored, gt, s1_ids, pcfg.exclusive_candidates, prob_col)
    print(f"  OOF macro F0.5 = {f:.4f} @ threshold {thr} (exclusive={pcfg.exclusive_candidates})")
    floor, f_e = tune_expected_f(scored, gt, s1_ids, pcfg.exclusive_candidates, prob_col)
    print(f"  OOF macro F0.5 = {f_e:.4f} with expected-F decoding, floor {floor}")
    decoder, thr, f = ("expected_f", floor, f_e) if f_e > f else ("threshold", thr, f)
    print(f"  decoder: {decoder} @ {thr}")
    return decoder, thr, f


def fit_with_cv(pairs, X, y, gt, s1_ids, mcfg: ModelConfig, pcfg: PostConfig, folds=None):
    """CV -> out-of-fold probs -> tuned decoder -> refit on everything.

    Two decoders are compared on the OOF probabilities: the global threshold, and per-S1
    expected-F0.5 decoding (see postprocess.select_expected_f). Each has exactly one tuned
    number (threshold / floor), picked by the same plateau-middle rule; the decoder with the
    higher OOF score is kept. Returns (booster, scored, decoder, threshold, oof_f05); `scored`
    carries the fold of every pair (GroupKFold by S1 unless `folds` is given).
    """
    folds = mdl.assign_folds(pairs["s1_id"].to_numpy(), mcfg.n_folds) if folds is None else folds
    with Timer("cv"):
        oof, iters = mdl.train_cv(X, y, folds, mcfg)
    scored = pairs[["s1_id", "cand_id"]].assign(prob=oof, fold=folds)
    decoder, thr, f = tune_decoder(scored, gt, s1_ids, pcfg)
    rounds = int(np.mean(iters) * 1.1)
    with Timer(f"refit ({rounds} rounds)"):
        booster = mdl.fit(X, y, mcfg, rounds)
    return booster, scored, decoder, thr, f


# ------------------------------------------------------------------ commands
def cmd_eda(args):
    for split in ("train", "test"):
        print(f"\n######## {split}")
        data = SplitData(args.data_dir, split, args.cache_dir)
        s1, tgt = data.load_all(["entity_id", "business_name", "business_address", "country",
                                 "name_core", "name_nonlatin", "addr_clean", "postal"])
        eda.report(s1, tgt, data.gt)
        del s1, tgt
        gc.collect()


def cmd_train(args):
    mcfg, pcfg = ModelConfig(), PostConfig()
    stage1, pairs, X, y, gt, countries, brep, pairs_c = train_frame(args)
    print(f"  train pairs {len(pairs):,}, positives {y.mean():.3f}, features {X.shape[1]}")
    ids = countries.index.tolist()
    booster, scored, decoder, thr, f = fit_with_cv(pairs, X, y, gt, ids, mcfg, pcfg)
    with Timer("save frames"):  # inputs of the cross-encoder and stack commands
        fdir, cdir = frame_dir(args, "train"), frame_dir(args, "ce")
        fdir.mkdir(parents=True, exist_ok=True)
        cdir.mkdir(parents=True, exist_ok=True)
        pairs[PAIR_COLS].assign(fold=scored["fold"].to_numpy(), y=y, prob=scored["prob"].to_numpy()).to_parquet(
            fdir / "pairs.parquet", index=False)
        X.to_parquet(fdir / "X.parquet", index=False)
        pd.DataFrame({"s1_id": ids, "country": countries.loc[ids].to_numpy()}).to_parquet(
            fdir / "s1.parquet", index=False)
        pairs_c.to_parquet(cdir / "pairs.parquet", index=False)
    pred = to_lists(select_with(scored, thr, pcfg.exclusive_candidates, decoder))
    for c, cid in countries.groupby(countries).groups.items():
        print(f"  OOF F0.5 [{c}] = {macro_f05(pred, gt, cid):.4f}  (n={len(cid)})")
    per = pd.DataFrame({"s1_id": ids, "country": countries.loc[ids].to_numpy(),
                        "f05": per_entity_f05(pred, gt, ids)})
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    per.to_parquet(Path(args.artifact_dir) / "oof_entity_f05.parquet", index=False)
    if args.baseline:
        compare_to_baseline(per, Path(args.baseline))
    if args.dump_errors:
        dump_errors(scored, pred, gt, ids, args)
    mdl.save(booster, {"features": list(X.columns), "threshold": thr, "decoder": decoder,
                       "exclusive": pcfg.exclusive_candidates, "oof_macro_f05": f, "blocking": brep},
             args.artifact_dir)
    stage1.save_model(str(Path(args.artifact_dir) / "stage1.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("  top features (gain):", ", ".join(imp.index[:20]))
    print(f"saved model to {args.artifact_dir}")


def cmd_validate(args):
    """Leave-one-country-out: simulates the unseen-country (France) shift of the test set."""
    mcfg, pcfg = ModelConfig(), PostConfig()
    _, pairs, X, y, gt, countries, _, _ = train_frame(args)
    pc = pairs["country"].values
    for c in sorted(countries.unique()):
        tr, te = pc != c, pc == c
        if not tr.any() or not te.any():
            continue
        print(f"\n== hold out country {c!r}")
        ids_tr = countries.index[countries != c].tolist()
        ids_te = countries.index[countries == c].tolist()
        booster, _, decoder, thr, _ = fit_with_cv(pairs[tr].reset_index(drop=True), X[tr].reset_index(drop=True),
                                                  y[tr], gt, ids_tr, mcfg, pcfg)
        scored = pairs.loc[te, ["s1_id", "cand_id"]].assign(prob=booster.predict(X[te]))
        pred = to_lists(select_with(scored, thr, pcfg.exclusive_candidates, decoder))
        print(f"  unseen-country F0.5 [{c}] = {macro_f05(pred, gt, ids_te):.4f} ({decoder} @ {thr})")


def cmd_predict(args):
    bcfg = BlockingConfig()
    booster, meta = mdl.load(args.artifact_dir)
    stage1 = lgb.Booster(model_file=str(Path(args.artifact_dir) / "stage1.txt"))
    data = SplitData(args.data_dir, "test", args.cache_dir)
    parts = []
    fdir = frame_dir(args, "test")
    fdir.mkdir(parents=True, exist_ok=True)
    writer = FrameWriter(fdir)

    def score(p, X):
        prob = booster.predict(X[meta["features"]]).astype(np.float32)
        parts.append(p[["s1_id", "cand_id"]].assign(prob=prob))
        writer.write(p.assign(prob=prob), X)

    generate(data, bcfg, None, stage1, score)
    writer.close()
    pd.DataFrame({"s1_id": data.s1_ids, "country": data.s1_country}).to_parquet(fdir / "s1.parquet", index=False)
    scored = pd.concat(parts, ignore_index=True)
    del parts
    matches = select_with(scored, meta["threshold"], meta["exclusive"], meta.get("decoder", "threshold"))

    problems = check_pairs(matches, scored, data.s1_ids, data.target_ids())
    if problems:
        raise SystemExit("submission check failed:\n  " + "\n  ".join(problems[:20]))
    out = Path(args.output_dir)
    write_pairs(out / "matching_results.tsv", MATCH_HEADER, data.s1_ids, matches)
    write_pairs(out / "candidate_pairs.tsv", CAND_HEADER, data.s1_ids, scored)

    n_matched = matches.groupby("s1_id").size().reindex(data.s1_ids).fillna(0).to_numpy()
    summary = pd.DataFrame({"country": data.s1_country, "n": n_matched}).groupby("country")["n"].agg(
        entities="size", with_match=lambda v: (v > 0).mean(), avg_matches="mean")
    print(summary.to_string())
    print(f"candidates: {len(scored):,} pairs ({len(scored) / len(data.s1_ids):.1f} per S1); "
          f"matches: {len(matches):,}; wrote {out}")


def cache_paths(args, split: str) -> dict[str, Path]:
    """Normalised parquet caches of a split (written by SplitData on first use)."""
    tag = _norm_tag()
    paths = {k: Path(args.cache_dir) / f"{split}_{k}_{tag}.parquet" for k in ("s1", "s2", "s3")}
    missing = [str(v) for v in paths.values() if not v.exists()]
    if missing:
        raise SystemExit(f"normalised caches missing (run train / predict first): {missing}")
    return paths


def ce_config(args):
    from config import CEConfig
    cfg = CEConfig()
    if args.ce_full:  # GPU budget: all of sample C, every pair scored
        cfg.max_train_pairs, cfg.band = None, None
    if args.ce_band:
        cfg.band = tuple(args.ce_band)
    return cfg


def cmd_ce_train(args):
    import json
    import cross_encoder as ce
    cfg = ce_config(args)
    pairs = pd.read_parquet(frame_dir(args, "ce") / "pairs.parquet", columns=["s1_id", "cand_id", "y"])
    with Timer("cross-encoder texts"):
        texts = ce.load_texts(cache_paths(args, "train").values(), np.r_[pairs["s1_id"], pairs["cand_id"]])
        a, b = ce.pair_texts(pairs, texts)
    with Timer("cross-encoder train"):
        rep = ce.train(a, b, pairs["y"].to_numpy(np.float32), Path(args.artifact_dir) / "ce_model", cfg)
    (Path(args.artifact_dir) / "ce_report.json").write_text(json.dumps({**rep, "model": cfg.model}, indent=2))


def cmd_ce_score(args):
    """Cross-encoder logit for the train (sample B) and test frames -> frames/<split>/ce.parquet.

    With a band (CPU budget) only pairs whose level-0 probability is in it are scored; the band
    is applied to the OOF probabilities on train and to the refit model's on test. Scores already
    in ce.parquet are kept, so widening the band only scores the new pairs.
    """
    import cross_encoder as ce
    cfg = ce_config(args)
    for split in ("train", "test"):
        fdir = frame_dir(args, split)
        pairs = pd.read_parquet(fdir / "pairs.parquet", columns=["s1_id", "cand_id", "prob"])
        out = np.full(len(pairs), np.nan, np.float32)
        if (fdir / "ce.parquet").exists():
            prev = pd.read_parquet(fdir / "ce.parquet")["ce_logit"].to_numpy(np.float32)
            if len(prev) == len(pairs):
                out = prev.copy()  # parquet-backed arrays are read-only
        m = ce.band_mask(pairs["prob"].to_numpy(), cfg.band) & np.isnan(out)
        print(f"  {split}: scoring {m.sum():,} of {len(pairs):,} pairs (band {cfg.band}; "
              f"{(~np.isnan(out)).sum():,} already scored)", flush=True)
        sub = pairs[m]
        with Timer(f"cross-encoder score {split}"):
            texts = ce.load_texts(cache_paths(args, split).values(), np.r_[sub["s1_id"], sub["cand_id"]])
            a, b = ce.pair_texts(sub, texts)
            del texts
            out[m] = ce.score(Path(args.artifact_dir) / "ce_model", a, b, cfg)
        pd.DataFrame({"ce_logit": out}).to_parquet(fdir / "ce.parquet", index=False)
        del pairs, sub, a, b
        release_memory()


def cmd_stack(args):
    import stack
    fdir = frame_dir(args, "train")
    ids = pd.read_parquet(fdir / "s1.parquet")["s1_id"].tolist()
    gt = gt_dict(read_ground_truth(Path(args.data_dir) / "train" / "train_ground_truth.tsv"), ids)
    paths = cache_paths(args, "train")
    scfg = StackConfig(use_cat=True) if args.use_cat else None
    stack.train(fdir, [paths["s2"], paths["s3"]], gt, Path(args.artifact_dir), tune_decoder, scfg)
    if args.baseline:
        per = pd.read_parquet(Path(args.artifact_dir) / "oof_entity_f05_stack.parquet")
        compare_to_baseline(per, Path(args.baseline))


def cmd_stack_predict(args):
    import stack
    fdir = frame_dir(args, "test")
    paths = cache_paths(args, "test")
    with Timer("stack predict"):
        scored, spec = stack.predict(fdir, [paths["s2"], paths["s3"]], Path(args.artifact_dir))
    data = SplitData(args.data_dir, "test", args.cache_dir)
    matches = select_with(scored, spec["threshold"], spec["exclusive"], spec["decoder"])
    problems = check_pairs(matches, scored, data.s1_ids, data.target_ids())
    if problems:
        raise SystemExit("submission check failed:\n  " + "\n  ".join(problems[:20]))
    out = Path(args.output_dir)
    write_pairs(out / "matching_results.tsv", MATCH_HEADER, data.s1_ids, matches)
    write_pairs(out / "candidate_pairs.tsv", CAND_HEADER, data.s1_ids, scored)
    n_matched = matches.groupby("s1_id").size().reindex(data.s1_ids).fillna(0).to_numpy()
    summary = pd.DataFrame({"country": data.s1_country, "n": n_matched}).groupby("country")["n"].agg(
        entities="size", with_match=lambda v: (v > 0).mean(), avg_matches="mean")
    print(summary.to_string())
    print(f"stack level {spec['level']} ({spec['decoder']} @ {spec['threshold']}): candidates {len(scored):,}, "
          f"matches {len(matches):,}; wrote {out}")


def compare_to_baseline(per: pd.DataFrame, baseline: Path):
    """Paired bootstrap of per-entity OOF F0.5 against an earlier run's oof_entity_f05.parquet.

    Only S1 entities scored in both runs are compared. A change is worth keeping when the lower
    95% bound of the gain is above zero overall and no country's mean gain is negative.
    """
    if baseline.is_dir():
        baseline = baseline / "oof_entity_f05.parquet"
    base = pd.read_parquet(baseline)
    m = per.merge(base[["s1_id", "f05"]], on="s1_id", suffixes=("", "_base"))
    if m.empty:
        print(f"  baseline {baseline}: no common S1 entities")
        return
    d, lo, hi = paired_bootstrap(m["f05_base"].to_numpy(), m["f05"].to_numpy())
    print(f"  vs baseline ({len(m):,} common S1): dF0.5 = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  -> {'KEEP' if lo > 0 else 'not significant'}")
    for c, g in m.groupby("country"):
        d, lo, hi = paired_bootstrap(g["f05_base"].to_numpy(), g["f05"].to_numpy())
        print(f"    [{c}] dF0.5 = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={len(g):,})")


def dump_errors(scored, pred, gt, ids, args):
    """False positives / negatives from OOF predictions, for error analysis."""
    data = load_split(args.data_dir, "train")
    info = pd.concat([data["s1"], data["s2"], data["s3"]]).set_index("entity_id")
    rows = []
    for s in ids[:20000]:
        p, true = set(pred.get(s, ())), gt.get(s, set())
        rows += [(s, c, "FP") for c in p - true] + [(s, c, "FN") for c in true - p]
    err = pd.DataFrame(rows, columns=["s1_id", "cand_id", "type"])
    err = err.merge(scored, on=["s1_id", "cand_id"], how="left")  # prob NaN => lost in blocking
    err = err.join(info.add_prefix("s1_"), on="s1_id").join(info.add_prefix("c_"), on="cand_id")
    path = Path(args.artifact_dir) / "oof_errors.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    err.to_csv(path, sep="\t", index=False)
    print(f"  wrote {len(err)} OOF errors to {path}")


def start_memory_guard(min_free_gb: float):
    """Abort (exit code 3) if available RAM drops below min_free_gb.

    On a 16 GB machine a run that overshoots RAM makes Windows grow the page file on C: rather
    than fail; stopping early is the better outcome.
    """
    try:
        import psutil
    except ImportError:
        return

    def watch():
        while True:
            if psutil.virtual_memory().available < min_free_gb * 2**30:
                me = psutil.Process()
                rss = me.memory_info().rss / 2**30
                print(f"!! available RAM below {min_free_gb} GB (process {rss:.1f} GB) - aborting", flush=True)
                for child in me.children(recursive=True):  # e.g. normalisation pool workers
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["eda", "train", "validate", "predict", "all",
                                        "ce-train", "ce-score", "stack", "stack-predict"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--sample", type=int, default=None, help="number of train S1 entities to use")
    ap.add_argument("--ce-sample", type=int, default=None,
                    help="train S1 entities (disjoint sample C) whose candidates train the cross-encoder")
    ap.add_argument("--dump-errors", action="store_true", help="write OOF FP/FN pairs for analysis")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="artifact dir (or oof_entity_f05.parquet) of an earlier run: paired bootstrap vs it")
    ap.add_argument("--reuse-stage1", action="store_true", help="reuse artifacts/stage1.txt instead of refitting")
    ap.add_argument("--ce-full", action="store_true",
                    help="cross-encoder: train on all of sample C and score every pair (GPU budget)")
    ap.add_argument("--ce-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="cross-encoder: score only pairs whose level-0 probability is in [LO, HI]")
    ap.add_argument("--subset", type=float, default=None,
                    help="dev mode: train on a seeded fraction of regions (e.g. 0.15)")
    ap.add_argument("--hide-s1", default="0",
                    help="train: hide this fraction of train S1 ('auto' = match test's targets per S1)")
    ap.add_argument("--use-cat", action="store_true",
                    help="stack: add CatBoost as a level-1 learner (slow on CPU)")
    ap.add_argument("--min-free-gb", type=float, default=1.5,
                    help="abort if available RAM falls below this (0 disables)")
    args = ap.parse_args(argv)
    if args.min_free_gb > 0:
        start_memory_guard(args.min_free_gb)
    cmds = {"eda": [cmd_eda], "train": [cmd_train], "validate": [cmd_validate],
            "predict": [cmd_predict], "all": [cmd_train, cmd_predict], "ce-train": [cmd_ce_train],
            "ce-score": [cmd_ce_score], "stack": [cmd_stack], "stack-predict": [cmd_stack_predict]}
    for fn in cmds[args.command]:
        fn(args)


if __name__ == "__main__":
    main()

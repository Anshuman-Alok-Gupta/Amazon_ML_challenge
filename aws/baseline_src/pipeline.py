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
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda  # noqa: E402
import model as mdl  # noqa: E402
import normalize  # noqa: E402
from blocking import TokenIndex, keep_top, make_blocks, make_target, pool, stage1_features, target_best  # noqa: E402
from config import (DEFAULT_ARTIFACT_DIR, DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR,  # noqa: E402
                    DEFAULT_OUTPUT_DIR, SEED, BlockingConfig, ModelConfig, PostConfig, TrainConfig)
from features import build_features  # noqa: E402
from io_utils import (CAND_HEADER, MATCH_HEADER, check_pairs, gt_dict, load_split,  # noqa: E402
                      read_ground_truth, read_source, write_pairs)  # noqa: E402
from metrics import blocking_report, macro_f05  # noqa: E402
from postprocess import select, select_pairs, tune_threshold  # noqa: E402

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
FEATURE_COLS = ["entity_id", "country", "name_clean", "name_core", "name_legal", "name_nonlatin",
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
            out.append(t["entity_id"].to_numpy()[self._mask(t, is_target=True)])
        return set(np.concatenate(out))

    def load(self, country: str, cols=FEATURE_COLS):
        f = [("country_norm", "==", country)]
        s1 = pd.read_parquet(self.paths["s1"], columns=cols, filters=f)
        tgt = make_target(pd.read_parquet(self.paths["s2"], columns=cols, filters=f),
                          pd.read_parquet(self.paths["s3"], columns=cols, filters=f))
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
    """Yield (block name, s1 frame, target frame, queried S1 rows, TokenIndex, name_freq) per block.

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
            yield f"{c}/{name}", s1_b, tgt_b, q, idx, name_freq
            del idx, s1_b, tgt_b
            gc.collect()
        del s1_all, tgt_all
        release_memory()


def stage1_pools(data: SplitData, bcfg: BlockingConfig, query_ids, gt: dict):
    """Retrieval pools with stage-1 features and labels, used to train the re-ranker."""
    feats, ys = [], []
    with Timer("stage-1 pools"):
        for name, s1_b, tgt_b, q, idx, _ in _blocks(data, bcfg, query_ids):
            src = tgt_b["source"].to_numpy()
            for s in range(0, len(q), bcfg.query_chunk):
                p = pool(idx, src, q[s:s + bcfg.query_chunk], bcfg)
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
    for name, s1_b, tgt_b, q, idx, name_freq in _blocks(data, bcfg, query_ids):
        src = tgt_b["source"].to_numpy()
        kept = []
        for s in range(0, len(q), bcfg.query_chunk):
            p = pool(idx, src, q[s:s + bcfg.query_chunk], bcfg)
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


def collect(data: SplitData, bcfg, query_ids, stage1):
    parts_p, parts_x = [], []
    generate(data, bcfg, query_ids, stage1, lambda p, x: (parts_p.append(p), parts_x.append(x)))
    return pd.concat(parts_p, ignore_index=True), pd.concat(parts_x, ignore_index=True)


def labels(pairs: pd.DataFrame, gt: dict) -> np.ndarray:
    return np.array([c in gt.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])], dtype=np.int8)


def id_lists(pairs: pd.DataFrame) -> dict[str, list[str]]:
    return pairs.groupby("s1_id", sort=False)["cand_id"].apply(list).to_dict()


def train_frame(args):
    """Stage 1 on sample A; candidates, features and labels for a disjoint sample B."""
    bcfg, tcfg = BlockingConfig(), TrainConfig()
    data = SplitData(args.data_dir, "train", args.cache_dir, subset=args.subset)
    n_b = args.sample or tcfg.n_s1_sample
    n_a = min(tcfg.n_stage1_sample, max(len(data.s1_ids) - n_b, len(data.s1_ids) // 4))
    perm = np.random.default_rng(SEED).permutation(data.s1_ids)
    ids_a, ids_b = np.sort(perm[:n_a]), np.sort(perm[n_a:n_a + n_b])
    gt = gt_dict(data.gt, np.concatenate([ids_a, ids_b]))
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

    # stage 2 data: sample B, candidates chosen by the (out-of-sample) stage-1 model
    pairs, X = collect(data, bcfg, set(ids_b), stage1)
    gt_b = {s: gt[s] for s in ids_b}
    y = labels(pairs, gt_b)
    rep = blocking_report(id_lists(pairs), gt_b, n_targets, ids_b)
    print("  blocking (sample B): " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                for k, v in rep.items()))
    return stage1, pairs, X, y, gt_b, countries, rep


def fit_with_cv(pairs, X, y, gt, s1_ids, mcfg: ModelConfig, pcfg: PostConfig):
    """CV -> out-of-fold probs -> tuned threshold -> refit on everything."""
    with Timer("cv"):
        oof, iters = mdl.train_cv(X, y, pairs["s1_id"].values, mcfg)
    scored = pairs[["s1_id", "cand_id"]].assign(prob=oof)
    thr, f = tune_threshold(scored, gt, s1_ids, pcfg.exclusive_candidates)
    print(f"  OOF macro F0.5 = {f:.4f} @ threshold {thr} (exclusive={pcfg.exclusive_candidates})")
    rounds = int(np.mean(iters) * 1.1)
    with Timer(f"refit ({rounds} rounds)"):
        booster = mdl.fit(X, y, mcfg, rounds)
    return booster, scored, thr, f


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
    stage1, pairs, X, y, gt, countries, brep = train_frame(args)
    print(f"  train pairs {len(pairs):,}, positives {y.mean():.3f}, features {X.shape[1]}")
    ids = countries.index.tolist()
    booster, scored, thr, f = fit_with_cv(pairs, X, y, gt, ids, mcfg, pcfg)
    pred = select(scored, thr, pcfg.exclusive_candidates)
    for c, cid in countries.groupby(countries).groups.items():
        print(f"  OOF F0.5 [{c}] = {macro_f05(pred, gt, cid):.4f}  (n={len(cid)})")
    # [bench patch: reporting only] per-entity OOF F0.5 for the paired bootstrap vs the new code
    from metrics import f05
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"s1_id": ids, "country": countries.loc[ids].to_numpy(),
                  "f05": [f05(set(pred.get(s, ())), gt.get(s, set())) for s in ids]}
                 ).to_parquet(Path(args.artifact_dir) / "oof_entity_f05.parquet", index=False)
    if args.dump_errors:
        dump_errors(scored, pred, gt, ids, args)
    mdl.save(booster, {"features": list(X.columns), "threshold": thr,
                       "exclusive": pcfg.exclusive_candidates, "oof_macro_f05": f, "blocking": brep},
             args.artifact_dir)
    stage1.save_model(str(Path(args.artifact_dir) / "stage1.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("  top features (gain):", ", ".join(imp.index[:20]))
    print(f"saved model to {args.artifact_dir}")


def cmd_validate(args):
    """Leave-one-country-out: simulates the unseen-country (France) shift of the test set."""
    mcfg, pcfg = ModelConfig(), PostConfig()
    _, pairs, X, y, gt, countries, _ = train_frame(args)
    pc = pairs["country"].values
    for c in sorted(countries.unique()):
        tr, te = pc != c, pc == c
        if not tr.any() or not te.any():
            continue
        print(f"\n== hold out country {c!r}")
        ids_tr = countries.index[countries != c].tolist()
        ids_te = countries.index[countries == c].tolist()
        booster, _, thr, _ = fit_with_cv(pairs[tr].reset_index(drop=True), X[tr].reset_index(drop=True),
                                         y[tr], gt, ids_tr, mcfg, pcfg)
        scored = pairs.loc[te, ["s1_id", "cand_id"]].assign(prob=booster.predict(X[te]))
        pred = select(scored, thr, pcfg.exclusive_candidates)
        print(f"  unseen-country F0.5 [{c}] = {macro_f05(pred, gt, ids_te):.4f} @ threshold {thr}")


def cmd_predict(args):
    bcfg = BlockingConfig()
    booster, meta = mdl.load(args.artifact_dir)
    stage1 = lgb.Booster(model_file=str(Path(args.artifact_dir) / "stage1.txt"))
    data = SplitData(args.data_dir, "test", args.cache_dir)
    parts = []

    def score(p, X):
        parts.append(p[["s1_id", "cand_id"]].assign(prob=booster.predict(X[meta["features"]]).astype(np.float32)))

    generate(data, bcfg, None, stage1, score)
    scored = pd.concat(parts, ignore_index=True)
    del parts
    matches = select_pairs(scored, meta["threshold"], meta["exclusive"])

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
    ap.add_argument("command", choices=["eda", "train", "validate", "predict", "all"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--sample", type=int, default=None, help="number of train S1 entities to use")
    ap.add_argument("--dump-errors", action="store_true", help="write OOF FP/FN pairs for analysis")
    ap.add_argument("--reuse-stage1", action="store_true", help="reuse artifacts/stage1.txt instead of refitting")
    ap.add_argument("--subset", type=float, default=None,
                    help="dev mode: train on a seeded fraction of regions (e.g. 0.15)")
    ap.add_argument("--min-free-gb", type=float, default=1.5,
                    help="abort if available RAM falls below this (0 disables)")
    args = ap.parse_args(argv)
    if args.min_free_gb > 0:
        start_memory_guard(args.min_free_gb)
    cmds = {"eda": [cmd_eda], "train": [cmd_train], "validate": [cmd_validate],
            "predict": [cmd_predict], "all": [cmd_train, cmd_predict]}
    for fn in cmds[args.command]:
        fn(args)


if __name__ == "__main__":
    main()

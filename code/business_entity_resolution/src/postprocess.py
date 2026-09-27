"""Turn pair probabilities into final match lists; tune the threshold for macro F0.5."""
from __future__ import annotations

import numpy as np
import pandas as pd


def exclusive(pairs: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    """Keep, for every S2/S3 record, only its highest-probability S1 entity."""
    idx = pairs.groupby("cand_id")[prob_col].idxmax()
    return pairs.loc[idx]


def select_pairs(pairs: pd.DataFrame, threshold: float, use_exclusive: bool,
                 prob_col: str = "prob") -> pd.DataFrame:
    """Final (s1_id, cand_id, prob) matches: optional exclusive assignment, then the cut-off."""
    p = pairs[pairs[prob_col] >= threshold]  # the per-target argmax is always above any lower cut
    if use_exclusive:
        # argmax over the rows that pass the cut equals the global argmax whenever it passes
        p = exclusive(pairs[pairs["cand_id"].isin(p["cand_id"])], prob_col)
        p = p[p[prob_col] >= threshold]
    return p.sort_values(["s1_id", prob_col], ascending=[True, False])


def select(pairs: pd.DataFrame, threshold: float, use_exclusive: bool,
           prob_col: str = "prob") -> dict[str, list[str]]:
    p = select_pairs(pairs, threshold, use_exclusive, prob_col)
    return p.groupby("s1_id", sort=False)["cand_id"].apply(list).to_dict()


def select_expected_f(pairs: pd.DataFrame, floor: float, use_exclusive: bool,
                      prob_col: str = "prob") -> pd.DataFrame:
    """Per-S1 decoding that maximises the (approximate) expected F0.5 of each S1's list.

    The metric is averaged per S1 entity, so the right cut is per entity, not global. With the
    candidates of one S1 sorted by probability p1 >= p2 >= ..., predicting the top k scores
        E[F0.5 | k] ~= 1.25 * (p1 + ... + pk) / (0.25 * (p1 + ... + pn) + k)      (k >= 1)
        E[F0.5 | 0]  = prod(1 - pi)          (the entity is a singleton: empty list scores 1)
    and the best k is kept. No parameter is fitted: `floor` is the tuned global threshold, used
    only as a safety net (a candidate below it is never predicted), so this can only prune
    lists relative to the threshold rule, or keep them identical.
    """
    p = exclusive(pairs, prob_col) if use_exclusive else pairs
    p = p[["s1_id", "cand_id", prob_col]].sort_values(["s1_id", prob_col], ascending=[True, False])
    pr = p[prob_col].to_numpy(dtype=np.float64).clip(1e-6, 1 - 1e-6)
    g = p.groupby("s1_id", sort=False)
    k = g.cumcount().to_numpy() + 1
    cum = g[prob_col].cumsum().to_numpy()
    tot = g[prob_col].transform("sum").to_numpy()
    p_empty = np.exp(pd.Series(np.log1p(-pr), index=p.index).groupby(p["s1_id"], sort=False).transform("sum").to_numpy())
    score = 1.25 * cum / (0.25 * tot + k)
    best = pd.Series(score, index=p.index).groupby(p["s1_id"], sort=False).transform("max").to_numpy()
    # best k per S1 = rank of the first row reaching the group max
    first_best = pd.Series(np.where(score >= best - 1e-12, k, np.inf), index=p.index)
    k_best = first_best.groupby(p["s1_id"], sort=False).transform("min").to_numpy()
    keep = (best > p_empty) & (k <= k_best) & (pr >= floor)
    return p[keep]


def select_with(pairs: pd.DataFrame, threshold: float, use_exclusive: bool, decoder: str = "threshold",
                prob_col: str = "prob") -> pd.DataFrame:
    if decoder == "expected_f":
        return select_expected_f(pairs, threshold, use_exclusive, prob_col)
    return select_pairs(pairs, threshold, use_exclusive, prob_col)


def to_lists(p: pd.DataFrame) -> dict[str, list[str]]:
    return p.groupby("s1_id", sort=False)["cand_id"].apply(list).to_dict()


class MacroScorer:
    """macro_f05 of many candidate selections from one pair frame, vectorised.

    Per S1: F0.5 = 1.25 tp / (0.25 n_true + n_pred) (0 when tp = 0), and 1.0 for a singleton
    with an empty list -- the same numbers as metrics.macro_f05, via bincount instead of Python
    sets, so a threshold grid over 400k entities takes seconds instead of many minutes.
    """

    def __init__(self, base: pd.DataFrame, gt: dict, s1_ids):
        ids = pd.Index(list(s1_ids))
        self.n = len(ids)
        self.n_true = np.array([len(gt.get(s, ())) for s in ids], dtype=np.float64)
        code = ids.get_indexer(base["s1_id"].to_numpy(object))
        keep = code >= 0  # pairs of S1 entities outside s1_ids do not count
        self.code = pd.Series(np.where(keep, code, 0), index=base.index)
        self.y = pd.Series(keep & np.fromiter((c in gt.get(a, ()) for a, c in zip(base["s1_id"], base["cand_id"])),
                                              dtype=bool, count=len(base)), index=base.index)
        self.valid = pd.Series(keep, index=base.index)

    def per_entity(self, index) -> np.ndarray:
        """F0.5 of every S1 (in s1_ids order) when exactly the base rows with these labels are predicted."""
        v = self.valid.loc[index].to_numpy()
        code = self.code.loc[index].to_numpy()[v]
        tp = np.bincount(code, weights=self.y.loc[index].to_numpy()[v], minlength=self.n)
        npred = np.bincount(code, minlength=self.n).astype(np.float64)
        return np.where(self.n_true == 0, (npred == 0).astype(np.float64),
                        1.25 * tp / np.maximum(0.25 * self.n_true + npred, 1e-12))

    def score(self, index) -> float:
        """Macro F0.5 when exactly the base rows with these index labels are predicted."""
        return float(self.per_entity(index).mean()) if self.n else float("nan")


def _plateau_middle(grid, scores) -> tuple[float, float]:
    # Middle of the best plateau: more robust to calibration shift (e.g. an unseen country).
    best = np.flatnonzero(scores >= scores.max() - 1e-6)
    i = best[len(best) // 2]
    return float(grid[i]), float(scores[i])


def tune_threshold(pairs: pd.DataFrame, gt: dict, s1_ids, use_exclusive: bool,
                   prob_col: str = "prob", grid=None) -> tuple[float, float]:
    """Grid-search the probability cut-off that maximises macro F0.5 (singletons included)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    base = exclusive(pairs, prob_col) if use_exclusive else pairs
    base = base[base[prob_col] >= grid.min()]
    sc = MacroScorer(base, gt, s1_ids)
    prob = base[prob_col].to_numpy()
    scores = np.array([sc.score(base.index[prob >= t]) for t in grid])
    return _plateau_middle(grid, scores)


def tune_expected_f(pairs: pd.DataFrame, gt: dict, s1_ids, use_exclusive: bool,
                    prob_col: str = "prob", grid=None) -> tuple[float, float]:
    """Floor for select_expected_f, chosen the same way as the threshold (plateau middle)."""
    grid = np.round(np.arange(0.30, 0.91, 0.05), 2) if grid is None else grid
    base = exclusive(pairs, prob_col) if use_exclusive else pairs
    sc = MacroScorer(base, gt, s1_ids)
    scores = np.array([sc.score(select_expected_f(base, t, False, prob_col).index) for t in grid])
    return _plateau_middle(grid, scores)

"""Turn pair probabilities into final match lists; tune the threshold for macro F0.5."""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import macro_f05


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


def tune_threshold(pairs: pd.DataFrame, gt: dict, s1_ids, use_exclusive: bool,
                   prob_col: str = "prob", grid=None) -> tuple[float, float]:
    """Grid-search the probability cut-off that maximises macro F0.5 (singletons included)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    base = exclusive(pairs, prob_col) if use_exclusive else pairs
    base = base[base[prob_col] >= grid.min()]
    scores = np.array([macro_f05(select(base, t, use_exclusive=False, prob_col=prob_col), gt, s1_ids)
                       for t in grid])
    # Middle of the best plateau: more robust to calibration shift (e.g. an unseen country).
    best = np.flatnonzero(scores >= scores.max() - 1e-6)
    i = best[len(best) // 2]
    return float(grid[i]), float(scores[i])

"""Challenge metric (macro F0.5 incl. singletons) and blocking-quality metrics."""
from __future__ import annotations

import numpy as np

BETA2 = 0.25  # beta = 0.5


def f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: dict, gt: dict, s1_ids=None) -> float:
    """Average per-S1 F0.5 over `s1_ids` (defaults to all ground-truth S1 ids)."""
    ids = list(gt) if s1_ids is None else list(s1_ids)
    if not ids:
        return float("nan")
    return float(np.mean([f05(set(pred.get(s, ())), gt.get(s, set())) for s in ids]))


def blocking_report(cands: dict, gt: dict, n_targets: int, s1_ids=None) -> dict:
    """Recall ceiling of the candidate set and its size."""
    ids = list(gt) if s1_ids is None else list(s1_ids)
    true_pairs = sum(len(gt.get(s, ())) for s in ids)
    hit = sum(len(set(cands.get(s, ())) & gt.get(s, set())) for s in ids)
    n_pairs = sum(len(cands.get(s, ())) for s in ids)
    # Best achievable score if the matcher were perfect on this candidate set.
    oracle = {s: set(cands.get(s, ())) & gt.get(s, set()) for s in ids}
    return {
        "pair_recall": hit / true_pairs if true_pairs else float("nan"),
        "f05_ceiling": macro_f05(oracle, gt, ids),
        "avg_candidates": n_pairs / max(len(ids), 1),
        "n_pairs": n_pairs,
        "reduction_ratio": 1 - n_pairs / max(len(ids) * n_targets, 1),
    }

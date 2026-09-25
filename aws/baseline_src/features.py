"""Pairwise features for (S1 record, candidate record) pairs.

All features are country-agnostic and vectorised: rapidfuzz `cpdist` (multithreaded C++) for
string similarities, sparse row-wise products on the blocking token incidence matrix for
IDF-weighted token / number overlap, bit operations for legal-form agreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from scipy import sparse

SCORERS = {
    "ratio": fuzz.ratio,
    "partial": fuzz.partial_ratio,
    "tsort": fuzz.token_sort_ratio,
    "tset": fuzz.token_set_ratio,
    "jw": JaroWinkler.normalized_similarity,
}


def _cp(a, b, scorer) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _overlap(idx, l1, lt, F, chunk: int = 1_000_000):
    """IDF-weighted overlap of name / address tokens and shared-number counts.

    One sparse product per chunk: P = X[s1] * X[target] marks the shared tokens, and
    P @ w_kind sums their weights for each kind (name, address, number).
    """
    lt_g = lt + idx.n1  # target rows follow S1 rows in the index
    n = len(l1)
    shared = {k: np.empty(n, np.float32) for k in idx.w}
    mx = {k: np.empty(n, np.float32) for k in ("name", "addr")}
    for s in range(0, n, chunk):
        P = idx.X[l1[s:s + chunk]].multiply(idx.X[lt_g[s:s + chunk]]).tocsr()
        for k, w in idx.w.items():
            shared[k][s:s + chunk] = P @ w
        for k in mx:
            mx[k][s:s + chunk] = (P @ sparse.diags(idx.w[k])).max(axis=1).toarray().ravel()
    for k in ("name", "addr"):
        sh, ma, mb = shared[k], idx.mass[k][l1], idx.mass[k][lt_g]
        union = ma + mb - sh
        F[f"{k}_idf_jac"] = np.where(union > 0, sh / np.maximum(union, 1e-6), np.nan)
        F[f"{k}_idf_shared"] = sh
        F[f"{k}_idf_unshared"] = union - sh
        F[f"{k}_idf_cov_s1"] = np.where(ma > 0, sh / np.maximum(ma, 1e-6), np.nan)
        F[f"{k}_idf_cov_t"] = np.where(mb > 0, sh / np.maximum(mb, 1e-6), np.nan)
        F[f"{k}_idf_max_shared"] = mx[k]
    sh, ma, mb = shared["skel"], idx.mass["skel"][l1], idx.mass["skel"][lt_g]
    union = ma + mb - sh
    F["skel_idf_jac"] = np.where(union > 0, sh / np.maximum(union, 1e-6), np.nan)
    F["skel_idf_cov_t"] = np.where(mb > 0, sh / np.maximum(mb, 1e-6), np.nan)
    sh, na, nb = shared["num"], idx.mass["num"][l1], idx.mass["num"][lt_g]
    both = (na > 0) & (nb > 0)
    F["num_shared"] = sh
    F["num_jac"] = np.where(both, sh / np.maximum(na + nb - sh, 1), np.nan)
    F["num_conflict"] = np.where(both, (sh == 0).astype(np.float32), np.nan)
    F["num_cov_s1"] = np.where(na > 0, sh / np.maximum(na, 1), np.nan)


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame, idx,
                   name_freq: tuple[pd.Series, pd.Series]) -> pd.DataFrame:
    """Features for `pairs` (local rows l1 / lt into the country frames s1 / tgt and index idx).

    `pairs` must contain whole S1 groups: context features rank within an S1's candidates.
    """
    ia, it = pairs["l1"].values, pairs["lt"].values
    A = s1.iloc[ia].reset_index(drop=True)
    B = tgt.iloc[it].reset_index(drop=True)
    for D in (A, B):  # cheap derived columns (not stored, to keep per-country frames small)
        D["name_compact"] = D["name_core"].str.replace(" ", "", regex=False)
        D["addr_alpha"] = D["addr_clean"].str.replace(r"\b\d+\b", " ", regex=True).str.split().str.join(" ")
    F = {}

    # --- retrieval scores
    for c in ("comb_cos", "name_cos", "addr_cos", "stage1"):
        F[c] = pairs[c].to_numpy().astype(np.float32)
    F["is_s3"] = (pairs["source"].values == 3).astype(np.float32)

    # --- name strings
    a_core, b_core = A["name_core"].tolist(), B["name_core"].tolist()
    for k, sc in SCORERS.items():
        F[f"core_{k}"] = _cp(a_core, b_core, sc)
    a_cmp, b_cmp = A["name_compact"].tolist(), B["name_compact"].tolist()
    F["compact_ratio"] = _cp(a_cmp, b_cmp, fuzz.ratio)
    F["compact_partial"] = _cp(a_cmp, b_cmp, fuzz.partial_ratio)
    F["compact_equal"] = (A["name_compact"].values == B["name_compact"].values).astype(np.float32)
    F["clean_tset"] = _cp(A["name_clean"].tolist(), B["name_clean"].tolist(), fuzz.token_set_ratio)
    na_tok = A["name_core"].str.count(" ").values + 1
    nb_tok = B["name_core"].str.count(" ").values + 1
    F["name_ntok_s1"], F["name_ntok_t"] = na_tok.astype(np.float32), nb_tok.astype(np.float32)
    la, lb = A["name_compact"].str.len().values, B["name_compact"].str.len().values
    F["name_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    first_a = A["name_core"].str.split(" ", n=1).str[0].values
    first_b = B["name_core"].str.split(" ", n=1).str[0].values
    F["first_tok_equal"] = (first_a == first_b).astype(np.float32)
    F["name_nonlatin_t"] = B["name_nonlatin"].values.astype(np.float32)

    ga, gb = A["name_legal"].values, B["name_legal"].values
    both = (ga > 0) & (gb > 0)
    F["legal_s1"], F["legal_t"] = (ga > 0).astype(np.float32), (gb > 0).astype(np.float32)
    F["legal_equal"] = np.where(both, (ga == gb).astype(np.float32), np.nan)
    F["legal_conflict"] = np.where(both, ((ga & gb) == 0).astype(np.float32), np.nan)

    freq_s1, freq_t = name_freq
    F["name_freq_s1"] = np.log1p(B["name_core"].map(freq_s1).fillna(0).values).astype(np.float32)
    F["name_freq_t"] = np.log1p(A["name_core"].map(freq_t).fillna(0).values).astype(np.float32)

    # --- address strings
    a_ad, b_ad = A["addr_clean"].tolist(), B["addr_clean"].tolist()
    missing = (A["addr_clean"].values == "") | (B["addr_clean"].values == "")
    for k in ("ratio", "partial", "tset", "tsort"):
        F[f"addr_{k}"] = np.where(missing, np.nan, _cp(a_ad, b_ad, SCORERS[k]))
    F["addr_alpha_tset"] = np.where(missing, np.nan, _cp(A["addr_alpha"].tolist(), B["addr_alpha"].tolist(),
                                                         fuzz.token_set_ratio))
    F["addr_missing_t"] = (B["addr_clean"].values == "").astype(np.float32)
    F["addr_nonlatin_t"] = B["addr_nonlatin"].values.astype(np.float32)
    la, lb = A["addr_clean"].str.len().values, B["addr_clean"].str.len().values
    F["addr_len_ratio"] = np.where(missing, np.nan, np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1))

    fa = A["addr_numbers"].str.split(" ", n=1).str[0].values
    fb = B["addr_numbers"].str.split(" ", n=1).str[0].values
    has = (fa != "") & (fb != "")
    F["first_num_equal"] = np.where(has, (fa == fb).astype(np.float32), np.nan)
    F["first_num_prefix"] = np.where(
        has, [float(x.startswith(y) or y.startswith(x)) for x, y in zip(fa, fb)], np.nan)
    pa, pb = A["postal"].values, B["postal"].values
    hp = (pa != "") & (pb != "")
    F["postal_equal"] = np.where(hp, (pa == pb).astype(np.float32), np.nan)
    lm_a, lm_b = A["addr_landmark"].values, B["addr_landmark"].values
    hl = (lm_a != "") & (lm_b != "")
    F["landmark_tset"] = np.where(hl, _cp(lm_a.tolist(), lm_b.tolist(), fuzz.token_set_ratio), np.nan)

    # --- IDF token overlap
    _overlap(idx, ia, it, F)
    F = pd.DataFrame(F)

    # --- context: competition among this S1's candidates and for this target
    ctx = pd.DataFrame({"s1": ia, "src": pairs["source"].values, "comb": F["comb_cos"].values,
                        "tset": F["core_tset"].values, "addr": F["addr_idf_jac"].fillna(0).values})
    g = ctx.groupby(["s1", "src"])
    for c in ("comb", "tset", "addr"):
        F[f"rank_{c}"] = g[c].rank(ascending=False, method="min").values.astype(np.float32)
        F[f"gap_{c}"] = (g[c].transform("max") - ctx[c]).values.astype(np.float32)
    F["n_cands_src"] = g["comb"].transform("size").values.astype(np.float32)
    ga_ = ctx.groupby("s1")
    F["rank_comb_all"] = ga_["comb"].rank(ascending=False, method="min").values.astype(np.float32)
    tb, ts = pairs["t_best"].values, pairs["t_second"].values
    F["t_gap_best"] = (tb - F["comb_cos"].values).astype(np.float32)
    F["t_is_best"] = (F["comb_cos"].values >= tb - 1e-5).astype(np.float32)
    F["t_margin"] = (tb - ts).astype(np.float32)
    return F.astype(np.float32)

"""Candidate generation at scale: sparse token retrieval + a learned stage-1 re-ranker.

Every record becomes a bag of namespaced tokens:
    n|<name token>   c|<compact name>   p|<consonant skeleton of a name token>
    k|<consonant skeleton of the whole compact name>   a|<address token>
Skeleton tokens ("builders" and the romanised Tamil "piltrs" both -> "pltrs"; "vidyaalya" and
"vidyalaya" -> "ptl") let transliterated and vowel-typo names meet. Tokens seen once are dropped;
tokens with document frequency above `max_df` get zero *retrieval* weight.

Per country, three L2-normalised IDF-weighted views share one vocabulary -- name, address,
name+address. For each Source 1 record a deep pool of Source 2 / Source 3 records is retrieved
under every view (multithreaded sparse top-n matmul). Many businesses share a name across
cities, so raw view scores rank poorly; a small LightGBM ("stage 1") re-ranks the pool from the
view scores, cheap name / house-number evidence and their ranks, and the top `keep_per_source`
per (S1, source) become the candidates. Those candidates are exactly what the matching model scores (candidate_pairs.tsv).
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

from config import BlockingConfig

VIEWS = ("comb", "name", "addr")


def make_target(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """Stack S2 and S3 into one target frame with a `source` column (2 or 3)."""
    return pd.concat([s2.assign(source=2), s3.assign(source=3)], ignore_index=True)


def make_blocks(s1_region: np.ndarray, t_region: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Split one country into (name, S1 rows, target rows) blocks by detected state / region.

    True pairs share the region in >98% of training cases, and same-name businesses in other
    regions are the main source of decoys, so each region is its own retrieval problem (also
    keeping every index small). Targets without a region, or whose region no S1 record has,
    go to a residual block searched against ALL S1 records of the country.
    """
    s1_groups = pd.Series(np.arange(len(s1_region))).groupby(s1_region).indices
    t_groups = pd.Series(np.arange(len(t_region))).groupby(t_region).indices
    blocks, residual = [], [t_groups.get("", np.empty(0, np.int64))]
    for r, it in t_groups.items():
        if r == "":
            continue
        i1 = s1_groups.get(r)
        if i1 is None:
            residual.append(it)
        else:
            blocks.append((r, np.asarray(i1), np.asarray(it)))
    it_res = np.sort(np.concatenate(residual))
    if len(it_res):
        blocks.append(("~residual", np.arange(len(s1_region)), it_res))
    return blocks


# ------------------------------------------------------------------ skeleton key
_SK_CE = re.compile(r"c(?=[eiy])")
_SK_DIGRAPHS = (("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
                ("sh", "s"), ("ch", "c"), ("ck", "k"), ("qu", "k"), ("x", "ks"))
_SK_CLASS = str.maketrans({"b": "p", "f": "p", "v": "p", "w": "p", "d": "t", "g": "k", "q": "k",
                           "z": "s", "m": "n", "a": "", "e": "", "i": "", "o": "", "u": "",
                           "y": "", "h": ""})
_SK_REPEAT = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Consonant-class skeleton of a romanised token (voicing, aspiration and vowels dropped)."""
    t = _SK_CE.sub("s", tok)
    for a, b in _SK_DIGRAPHS:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    return _SK_REPEAT.sub(r"\1", t.translate(_SK_CLASS))


def record_tokens(df: pd.DataFrame):
    """Yield one space-joined token document per record (a generator: 7M+ strings would not
    fit comfortably in memory alongside the index)."""
    for core, addr in zip(df["name_core"], df["addr_clean"]):
        words = core.split()
        toks = ["n|" + t for t in words]
        compact = "".join(words)
        if compact:
            # emitted even for one-word names, so a domain-style target ("glistensons.com")
            # can meet the compact form of a multi-word S1 name ("Glisten & Sons")
            toks.append("c|" + compact)
            # skeleton of the whole compact name: "sevnkeyr" (romanised Devanagari) and
            # "sevencare" both -> "spnkr". Unlike the per-token skeletons it is long and rare,
            # so it carries a high IDF and can pull native-script names into the pool.
            ksk = skeleton(compact)
            if len(ksk) >= 4:
                toks.append("k|" + ksk)
        for t in words:
            if len(t) >= 3 and not t.isdigit():
                sk = skeleton(t)
                if len(sk) >= 2:
                    toks.append("p|" + sk)
        toks += ["a|" + t for t in addr.split()]
        yield " ".join(toks)


# ------------------------------------------------------------------ index
class TokenIndex:
    """Token incidence + retrieval views for the S1 and target records of one country.

    Rows 0..n1-1 are S1 records, rows n1.. are target records (both in frame order).
    """

    def __init__(self, s1: pd.DataFrame, tgt: pd.DataFrame, cfg: BlockingConfig):
        try:
            cv = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                 binary=True, min_df=2, dtype=np.float32)
            X = cv.fit_transform(itertools.chain(record_tokens(s1), record_tokens(tgt))).tocsr()
        except ValueError:  # tiny block: no token occurs twice (or no tokens at all)
            cv = CountVectorizer(tokenizer=lambda d: d.split() or ["_"], token_pattern=None,
                                 lowercase=False, binary=True, dtype=np.float32)
            X = cv.fit_transform(itertools.chain(record_tokens(s1), record_tokens(tgt))).tocsr()
        X.indices = X.indices.astype(np.int32, copy=False)
        dfreq = np.bincount(X.indices, minlength=X.shape[1])
        names = cv.get_feature_names_out()
        del cv
        ns = np.array([t[0] for t in names])
        is_num = np.array([t[0] == "a" and t[2:].isdigit() for t in names])
        idf = np.log(X.shape[0] / np.maximum(dfreq, 1)).astype(np.float32)
        self.n1, self.X = len(s1), X
        # per-kind weight vectors for overlap features: shared = (X[a] * X[b]) @ w
        self.w = {"name": idf * (ns == "n"), "skel": idf * (ns == "p"), "cskel": idf * (ns == "k"),
                  "addr": idf * ((ns == "a") & ~is_num), "num": is_num.astype(np.float32)}
        self.mass = {k: X @ v for k, v in self.w.items()}

        ridf = np.where(dfreq > cfg.max_df, 0.0, idf).astype(np.float32)
        is_name = ns != "a"
        name = l2norm(X @ sparse.diags(ridf * is_name), copy=False)
        addr = l2norm(X @ sparse.diags(ridf * ~is_name), copy=False)
        comb = l2norm(sparse.hstack([name, addr * cfg.addr_weight], format="csr"), copy=False)
        self.views = {}
        for k, m in (("comb", comb), ("name", name), ("addr", addr)):
            m = m.tocsr()
            m.eliminate_zeros()
            self.views[k] = m

        self._bt = {}

    def target_view_T(self, view: str, src: int, tgt_source: np.ndarray):
        """Transposed target view of one source, cached (~0.6 GB for all six on US data)."""
        key = (view, src)
        if key not in self._bt:
            cols = np.flatnonzero(tgt_source == src)
            self._bt[key] = (cols, self.views[view][self.n1 + cols].T.tocsr())
        return self._bt[key]

    def drop_views(self):
        self.views, self._bt = {}, {}


def _topn(A, BT, k, threshold, chunk, n_threads):
    """Top-k columns of BT for every row of A: (row, col, score)."""
    rows, cols, vals = [], [], []
    for s in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[s:s + chunk], BT, top_n=k, threshold=threshold, n_threads=n_threads).tocoo()
        rows.append(C.row.astype(np.int64) + s)
        cols.append(C.col.astype(np.int64))
        vals.append(C.data)
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def rowwise_dot(A, ia, B, ib, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(axis=1)).ravel()
    return out


# ------------------------------------------------------------------ stage 0: retrieval pool
def pool(idx: TokenIndex, tgt_source: np.ndarray, query: np.ndarray, cfg: BlockingConfig) -> pd.DataFrame:
    """Union of the top-k targets under each view for the S1 rows in `query`, with view scores."""
    found = []
    for src in (2, 3):
        if not (tgt_source == src).any():
            continue
        for view, k in (("comb", cfg.k_comb), ("name", cfg.k_name), ("addr", cfg.k_addr)):
            cols, BT = idx.target_view_T(view, src, tgt_source)
            r, c, _ = _topn(idx.views[view][query], BT, k, cfg.min_score, cfg.chunk_rows, cfg.n_threads)
            found.append(query[r] * np.int64(len(tgt_source)) + cols[c])
    if not found:
        return pd.DataFrame({"l1": [], "lt": [], "source": []}, dtype=np.int64)
    key = np.unique(np.concatenate(found))
    n_t = np.int64(len(tgt_source))
    p = pd.DataFrame({"l1": key // n_t, "lt": key % n_t})
    for v in VIEWS:
        p[f"{v}_cos"] = rowwise_dot(idx.views[v], p["l1"].to_numpy(), idx.views[v], p["lt"].to_numpy() + idx.n1)
    p["source"] = tgt_source[p["lt"].to_numpy()]
    return p


# cheap pair evidence added to the pool (pool_features): the cosine views alone cannot tell a
# same-address record with a garbled / native-script name, or a changed house number, from
# the many same-name decoys of a region
POOL_EXTRA = ["name_tset", "name_cov_t", "name_cov_s1", "cskel_shared", "num_shared", "num_jac",
              "first_num_eq", "t_nonlatin", "t_addr_empty"]
STAGE1_FEATURES = ["comb_cos", "name_cos", "addr_cos", "prod", "is_s3", "n_pool",
                   "rk_comb_cos", "rk_name_cos", "rk_addr_cos", "gap_comb_cos", "gap_name_cos",
                   "gap_addr_cos", "rk_prod", "gap_prod", "rk_comb_all"] + POOL_EXTRA + ["rk_name_tset", "rk_num_jac"]


def shared_mass(idx: TokenIndex, l1: np.ndarray, lt: np.ndarray, kinds, chunk: int = 1_000_000) -> dict:
    """IDF mass of the tokens each (S1 row, target row) pair shares, per token kind."""
    lt_g = lt + idx.n1  # target rows follow S1 rows in the index
    out = {k: np.empty(len(l1), np.float32) for k in kinds}
    for s in range(0, len(l1), chunk):
        P = idx.X[l1[s:s + chunk]].multiply(idx.X[lt_g[s:s + chunk]]).tocsr()
        for k in kinds:
            out[k][s:s + chunk] = P @ idx.w[k]
    return out


class BlockText:
    """Per-block record columns as numpy object arrays, so pool pairs index them cheaply."""

    def __init__(self, s1: pd.DataFrame, tgt: pd.DataFrame):
        def first_num(df):
            return df["addr_numbers"].str.split(" ", n=1).str[0].to_numpy(object)
        self.name1, self.name_t = s1["name_core"].to_numpy(object), tgt["name_core"].to_numpy(object)
        self.num1, self.num_t = first_num(s1), first_num(tgt)
        self.nonlatin_t = tgt["name_nonlatin"].to_numpy().astype(np.float32)
        self.addr_empty_t = (tgt["addr_clean"].to_numpy(object) == "").astype(np.float32)


def pool_features(p: pd.DataFrame, idx: TokenIndex, text: BlockText) -> pd.DataFrame:
    """Add POOL_EXTRA columns to a retrieval pool (see stage1_features)."""
    l1, lt = p["l1"].to_numpy(), p["lt"].to_numpy()
    if len(p) == 0:
        return p.assign(**{c: np.empty(0, np.float32) for c in POOL_EXTRA})
    p["name_tset"] = cpdist(text.name1[l1], text.name_t[lt], scorer=fuzz.token_set_ratio, workers=-1,
                            dtype=np.float32)
    sh = shared_mass(idx, l1, lt, ("name", "cskel", "num"))
    ma, mb = idx.mass["name"][l1], idx.mass["name"][lt + idx.n1]
    p["name_cov_t"] = np.where(mb > 0, sh["name"] / np.maximum(mb, 1e-6), np.nan).astype(np.float32)
    p["name_cov_s1"] = np.where(ma > 0, sh["name"] / np.maximum(ma, 1e-6), np.nan).astype(np.float32)
    p["cskel_shared"] = sh["cskel"]
    na, nb = idx.mass["num"][l1], idx.mass["num"][lt + idx.n1]
    p["num_shared"] = sh["num"]
    p["num_jac"] = np.where((na > 0) & (nb > 0), sh["num"] / np.maximum(na + nb - sh["num"], 1),
                            np.nan).astype(np.float32)
    fa, fb = text.num1[l1], text.num_t[lt]
    p["first_num_eq"] = np.where((fa != "") & (fb != ""), (fa == fb).astype(np.float32), np.nan)
    p["t_nonlatin"], p["t_addr_empty"] = text.nonlatin_t[lt], text.addr_empty_t[lt]
    return p


def stage1_features(p: pd.DataFrame) -> pd.DataFrame:
    """Cheap re-ranking features: view scores and how they compare within the S1's pool."""
    F = pd.DataFrame({c: p[c].to_numpy() for c in ("comb_cos", "name_cos", "addr_cos")})
    F["prod"] = F["name_cos"] * F["addr_cos"]
    for c in POOL_EXTRA:
        F[c] = p[c].to_numpy()
    F["is_s3"] = (p["source"].to_numpy() == 3).astype(np.float32)
    key = pd.DataFrame({"l1": p["l1"].to_numpy(), "src": p["source"].to_numpy()})
    g = pd.concat([key, F], axis=1).groupby(["l1", "src"])
    F["n_pool"] = g["comb_cos"].transform("size").to_numpy()
    for c in ("comb_cos", "name_cos", "addr_cos", "prod"):
        F[f"rk_{c}"] = g[c].rank(ascending=False, method="average").to_numpy()
        F[f"gap_{c}"] = (g[c].transform("max") - F[c]).to_numpy()
    F["rk_comb_all"] = F.groupby(key["l1"])["comb_cos"].rank(ascending=False, method="average").to_numpy()
    g = pd.concat([key, F[["name_tset", "num_jac"]].fillna(-1)], axis=1).groupby(["l1", "src"])
    F["rk_name_tset"] = g["name_tset"].rank(ascending=False, method="average").to_numpy()
    F["rk_num_jac"] = g["num_jac"].rank(ascending=False, method="average").to_numpy()
    return F[STAGE1_FEATURES].astype(np.float32)


def keep_top(p: pd.DataFrame, score: np.ndarray, cfg: BlockingConfig) -> pd.DataFrame:
    """Top `keep_per_source` pool entries per (S1, source) by stage-1 score."""
    p = p.assign(stage1=score.astype(np.float32))
    p = p[p["stage1"] >= cfg.stage1_min]
    p = p.sort_values(["l1", "source", "stage1"], ascending=[True, True, False])
    rk = p.groupby(["l1", "source"]).cumcount()
    return p[rk.to_numpy() < cfg.keep_per_source].reset_index(drop=True)


def target_best(idx: TokenIndex, lt: np.ndarray, cfg: BlockingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Best / second-best combined score of each target row in `lt` against ALL S1 rows.

    Uses the full S1 pool (not only queried rows), so training on an S1 sample sees the same
    competition for each target as inference on the whole test set.
    """
    V = idx.views["comb"]
    r, _, v = _topn(V[idx.n1 + lt], V[: idx.n1].T.tocsr(), 2, 0.0, cfg.chunk_rows, cfg.n_threads)
    best = np.zeros(len(lt), np.float32)
    second = np.zeros(len(lt), np.float32)
    order = np.lexsort((-v, r))
    r, v = r[order], v[order]
    first = np.r_[True, r[1:] != r[:-1]]
    best[r[first]] = v[first]
    sec = ~first
    second[r[sec]] = v[sec]
    return best, second

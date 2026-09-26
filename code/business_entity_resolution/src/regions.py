"""Region clean-up and inference, learned from each split's own records (no place-name tables).

Blocking splits every country into one block per detected region (see blocking.make_blocks).
normalize.address_fields detects a region only from an explicit region name / code, so two
things go wrong at scale, and both are fixed here by statistics alone -- the same code runs for
every country label, including one never seen in training:

1. Noise regions. A handful of records get a region code that is really something else (a
   floor number read as a state, a highway number, ...). Such a "region" holds a tiny share of
   its country's S1 records and turns into a tiny block that separates true pairs. Any region
   below `MIN_REGION_FRAC` of its country's S1 records is dropped back to "" (residual block).

2. Missing regions. Sources that write only a city or a sub-region (a district, a county, a
   département) get no region, and every such record lands in the country's residual block,
   searched against all of that country's S1 records -- more decoys, and a candidate set the
   models rarely saw in training. `infer_regions` learns which address words co-occur (almost)
   exclusively with one region among the records that DO carry one, then assigns that region
   to region-less records whose address contains such a word. Two passes let the evidence
   chain (a city identifies the region; the sub-region written next to that city in other
   records is learned from them in pass 2).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from normalize import REGION_MERGE, STATE_CODES

MIN_REGION_FRAC = 0.0005   # region with < 0.05% of its country's S1 records is noise
REGION_MIN_COUNT = 20      # a witness word must be seen with a region at least this often ...
REGION_MIN_PURITY = 0.98   # ... and with that region in at least this share of its records
INFER_PASSES = 2

# The last four tokens of a normalised address: locality and region trail the address in every
# source here, and looking only at the tail keeps street words out of the witness table.
_TAIL = re.compile(r"(?:(\S+)\s+)?(?:(\S+)\s+)?(?:(\S+)\s+)?(\S+)$")
# Region codes themselves are never witnesses (that would just re-learn the detector).
_REGION_CODES = frozenset(REGION_MERGE.get(v, v) for v in STATE_CODES.values()) | frozenset(STATE_CODES.values())


def region_freq(country: pd.Series, region: pd.Series) -> dict[str, dict[str, int]]:
    """Per country: how many S1 records carry each region unambiguously (a single candidate)."""
    df = pd.DataFrame({"c": country.to_numpy(), "r": region.to_numpy()})
    df = df[(df["r"] != "") & ~df["r"].str.contains("|", regex=False)]
    return {c: g.value_counts().to_dict() for c, g in df.groupby("c")["r"]}


def resolve_regions(region: pd.Series, freq: dict[str, int]) -> pd.Series:
    """Collapse multi-candidate regions ("or|wb") to the candidate most common in this country.

    Normalisation cannot tell a stray fragment that happens to spell a region code ("Oor" -> "or")
    from the real region; the country's own unambiguous records can. Ties break alphabetically,
    so the choice depends only on the candidate set, never on the order a source wrote them in.
    """
    multi = region.str.contains("|", regex=False).to_numpy()
    if not multi.any():
        return region
    uniq = pd.unique(region[multi])
    pick = {u: max(sorted(u.split("|")), key=lambda r: freq.get(r, 0)) for u in uniq}
    out = region.copy()
    out[multi] = region[multi].map(pick)
    return out


def rare_regions(country: pd.Series, region: pd.Series) -> dict[str, set]:
    """Per country: regions holding less than MIN_REGION_FRAC of that country's S1 records."""
    df = pd.DataFrame({"c": country.to_numpy(), "r": region.to_numpy()})
    df = df[df["r"] != ""]
    out = {}
    for c, g in df.groupby("c")["r"]:
        frac = g.value_counts(normalize=True)
        out[c] = set(frac.index[frac < MIN_REGION_FRAC])
    return out


def drop_regions(df: pd.DataFrame, bad: set) -> pd.DataFrame:
    if bad:
        m = df["region"].isin(bad).to_numpy()
        if m.any():
            df = df.copy()
            df.loc[m, "region"] = ""
    return df


def _witnesses(addr: pd.Series) -> pd.DataFrame:
    """Up to seven candidate locality n-grams per record (4 unigrams + 3 bigrams of the tail);
    digits and region codes are blanked."""
    t = addr.astype(str).str.extract(_TAIL).fillna("")
    t.columns = ["t4", "t3", "t2", "t1"]
    for c in t.columns:
        v = t[c]
        t.loc[v.str.isdigit() | v.isin(_REGION_CODES) | (v.str.len() < 3), c] = ""
    out = {"u1": t["t1"], "u2": t["t2"], "u3": t["t3"], "u4": t["t4"]}
    for a, b, name in (("t2", "t1", "b21"), ("t3", "t2", "b32"), ("t4", "t3", "b43")):
        both = (t[a] != "") & (t[b] != "")
        out[name] = (t[a] + " " + t[b]).where(both, "")
    return pd.DataFrame(out)


CHUNK = 1_000_000  # records per witness batch (bounds memory on the largest countries)


def _fit(addr: pd.Series, region: np.ndarray) -> dict[str, str]:
    """witness -> region, kept only when frequent and (near-)exclusive to one region."""
    have = np.flatnonzero(region != "")
    counts = None
    for s in range(0, len(have), CHUNK):
        rows = have[s:s + CHUNK]
        wit = _witnesses(addr.iloc[rows].reset_index(drop=True))
        reg = region[rows]
        for c in wit.columns:
            w = wit[c].to_numpy()
            m = w != ""
            part = pd.DataFrame({"w": w[m], "r": reg[m]}).value_counts()
            counts = part if counts is None else counts.add(part, fill_value=0)
    if counts is None or counts.empty:
        return {}
    cnt = counts.rename("n").reset_index()
    tot = cnt.groupby("w")["n"].transform("sum")
    best = cnt.assign(tot=tot).sort_values("n", ascending=False).drop_duplicates("w")
    keep = best[(best["tot"] >= REGION_MIN_COUNT) & (best["n"] >= REGION_MIN_PURITY * best["tot"])]
    return dict(zip(keep["w"], keep["r"]))


def _apply(addr: pd.Series, table: dict[str, str]) -> np.ndarray:
    """One region per record: the region all of its matching witnesses agree on, else ''."""
    out = np.full(len(addr), "", dtype=object)
    for s in range(0, len(addr), CHUNK):
        wit = _witnesses(addr.iloc[s:s + CHUNK].reset_index(drop=True))
        o = np.full(len(wit), "", dtype=object)
        conflict = np.zeros(len(wit), dtype=bool)
        for c in wit.columns:
            hit = wit[c].map(table).to_numpy()
            found = pd.notna(hit)
            conflict |= found & (o != "") & (hit != o)
            fill = found & (o == "")
            o[fill] = hit[fill]
        o[conflict] = ""
        out[s:s + len(o)] = o
    return out


def infer_regions(s1: pd.DataFrame, tgt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Fill region-less S1 / target records of ONE country (frames are modified in place and
    returned). Returns (s1, tgt, n_filled)."""
    if "addr_clean" not in s1.columns or "region" not in s1.columns:
        return s1, tgt, 0
    both_reg = np.concatenate([s1["region"].to_numpy(dtype=object), tgt["region"].to_numpy(dtype=object)])
    both_addr = pd.concat([s1["addr_clean"], tgt["addr_clean"]], ignore_index=True)
    filled = 0
    for _ in range(INFER_PASSES):
        empty = np.flatnonzero(both_reg == "")
        if not len(empty):
            break
        table = _fit(both_addr, both_reg)
        if not table:
            break
        new = _apply(both_addr.iloc[empty], table)
        hit = new != ""
        if not hit.any():
            break
        both_reg[empty[hit]] = new[hit]
        filled += int(hit.sum())
    if filled:
        s1["region"] = both_reg[: len(s1)].astype(str)
        tgt["region"] = both_reg[len(s1):].astype(str)
    return s1, tgt, filled

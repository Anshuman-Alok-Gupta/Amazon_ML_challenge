"""Reading challenge TSVs and writing / checking submission files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_tsv(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: never turn "NA"/"" into NaN, IDs stay strings.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def read_source(path: Path) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (got {list(df.columns)})")
    df = df[SOURCE_COLS].copy()
    for c in SOURCE_COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def read_ground_truth(path: Path) -> pd.DataFrame:
    """One row per true link: columns s1_id, cand_id (singletons have no rows).

    Kept as a frame (Arrow strings) rather than a dict of sets: ~7.6M links would cost
    over a gigabyte as Python objects. Use gt_dict() for the subset being scored.
    """
    df = read_tsv(path)
    ex = df.assign(cand_id=df["matched_entity_ids"].str.split(",")).explode("cand_id")
    ex["cand_id"] = ex["cand_id"].fillna("").str.strip()
    ex = ex[ex["cand_id"] != ""]
    return ex.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]].reset_index(drop=True)


def gt_dict(gt: pd.DataFrame, s1_ids) -> dict[str, set[str]]:
    """{s1_id: set of true matches} for the given S1 ids (empty set for singletons)."""
    ids = set(s1_ids)
    out: dict[str, set[str]] = {s: set() for s in ids}
    sub = gt[gt["s1_id"].isin(ids)]
    for s, c in zip(sub["s1_id"], sub["cand_id"]):
        out[s].add(c)
    return out


def load_split(data_dir: Path, split: str) -> dict:
    """Load {s1, s2, s3[, gt]} for split in {"train", "test"}."""
    d = Path(data_dir) / split
    out = {f"s{i}": read_source(d / f"{split}_source{i}.tsv") for i in (1, 2, 3)}
    gt_path = d / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        out["gt"] = read_ground_truth(gt_path)
    return out


def write_pairs(path: Path, header: tuple[str, str], s1_ids, pairs: pd.DataFrame) -> None:
    """One row per S1 id (in the given order) with its comma-joined cand_ids, empty if none.

    `pairs` has columns s1_id, cand_id; order within an S1 is kept, duplicates dropped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = pairs[["s1_id", "cand_id"]].drop_duplicates()
    joined = p.groupby("s1_id", sort=False)["cand_id"].agg(",".join).to_dict()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{joined.get(s1, '')}\n")


def check_pairs(matches: pd.DataFrame, cands: pd.DataFrame, s1_ids, target_ids) -> list[str]:
    """Local mirror of the official validator rules on pair frames. Returns a list of problems."""
    problems = []
    s1_ids, target_ids = pd.Index(s1_ids), pd.Index(target_ids)
    if s1_ids.has_duplicates:
        problems.append("duplicate S1 ids")
    for name, df in (("matching", matches), ("candidate", cands)):
        if not df["s1_id"].isin(s1_ids).all():
            problems.append(f"{name}: unknown S1 ids")
        bad = df.loc[~df["cand_id"].isin(target_ids), "cand_id"]
        if len(bad):
            problems.append(f"{name}: {len(bad)} ids not in test S2/S3, e.g. {bad.head(3).tolist()}")
    extra = matches.merge(cands[["s1_id", "cand_id"]], how="left", indicator=True)
    if (extra["_merge"] == "left_only").any():
        problems.append(f"{(extra['_merge'] == 'left_only').sum()} matches are not among the candidates")
    return problems

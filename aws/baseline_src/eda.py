"""Dataset statistics used to confirm the design assumptions (run before modelling)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def report(s1: pd.DataFrame, tgt: pd.DataFrame, gt: pd.DataFrame | None, n_samples: int = 6) -> None:
    """s1 / tgt: prepared frames; gt: link frame (s1_id, cand_id) or None for the test split."""
    print("== sizes / countries")
    for name, df in (("S1", s1), ("S2", tgt[tgt["source"] == 2]), ("S3", tgt[tgt["source"] == 3])):
        print(f"  {name}: {len(df):>9,}  countries={df['country'].value_counts().to_dict()}")
        print(f"    empty address {(df['addr_clean'] == '').mean():.1%}  "
              f"non-Latin name {df['name_nonlatin'].mean():.1%}  postal present {(df['postal'] != '').mean():.1%}")
    if gt is None:
        return

    per_s1 = gt.groupby("s1_id").size().reindex(s1["entity_id"]).fillna(0).astype(int)
    print("== ground truth")
    print(f"  links: {len(gt):,}  singletons: {(per_s1 == 0).mean():.1%}  matches per S1: mean {per_s1.mean():.2f} "
          f"max {per_s1.max()}  dist {per_s1.value_counts().sort_index().head(12).to_dict()}")
    print(f"  target ids linked to >1 S1: {gt['cand_id'].duplicated().sum()} (0 => exclusive assignment is safe)")
    for src in ("S2", "S3"):
        n_src = (tgt["source"] == int(src[1])).sum()
        print(f"  {src} records that match some S1: {gt['cand_id'].str.startswith(src).sum() / max(n_src, 1):.1%}")

    # Field agreement among a sample of true pairs.
    samp = gt.sample(min(len(gt), 100_000), random_state=0)
    a = s1.set_index("entity_id").loc[samp["s1_id"].values].reset_index()
    b = tgt.set_index("entity_id").loc[samp["cand_id"].values].reset_index()
    print("== agreement among true pairs (sample)")
    print(f"  same country label: {(a['country'].values == b['country'].values).mean():.1%}")
    print(f"  identical name_core: {(a['name_core'].values == b['name_core'].values).mean():.1%}")
    both = (a["postal"] != "").values & (b["postal"] != "").values
    print(f"  postal on both: {both.mean():.1%}; equal when both: {(a['postal'].values[both] == b['postal'].values[both]).mean():.1%}")
    print("== sample true pairs")
    for n in np.random.default_rng(0).choice(len(a), size=min(n_samples, len(a)), replace=False):
        print(f"  {a['business_name'][n]!r} | {a['business_address'][n]!r}\n"
              f"  {b['business_name'][n]!r} | {b['business_address'][n]!r}\n")

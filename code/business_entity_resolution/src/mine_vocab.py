"""Mine noise vocabulary from true training pairs (audit trail for the hand-added entries).

    python src/mine_vocab.py [--min-support 50] [--pairs 60000]

Only stage-1 sample A is used (the same seeded draw as pipeline.train_frame), so nothing mined
here comes from the S1 entities the out-of-fold F0.5 is measured on. Prints, per country:
  - 1:1 address-token substitutions between an S1 record and its true matches
    (e.g. "court -> ct": a canonicalisation the normaliser misses)
  - name tokens that a matching record adds / drops relative to the S1 name
    (e.g. honorifics "shri", "mr"; romanised legal forms "pra", "limird")
Entries were added to normalize.py only when they are generic (an abbreviation, honorific or
legal form, never a specific business or place) and have at least --min-support pairs.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_DATA_DIR, SEED, TrainConfig  # noqa: E402
from normalize import address_fields, name_fields  # noqa: E402


def read_ids(path: Path, ids: set, cols) -> pd.DataFrame:
    out = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=cols, chunksize=1_000_000):
        out.append(ch[ch["entity_id"].isin(ids)])
    return pd.concat(out).set_index("entity_id")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--min-support", type=int, default=50)
    ap.add_argument("--pairs", type=int, default=60_000, help="max S1 entities of sample A to mine")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)
    d = args.data_dir / "train"
    cols = ["entity_id", "business_name", "business_address", "country"]

    s1_ids = pd.read_csv(d / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                         usecols=["entity_id"])["entity_id"].to_numpy()
    sample_a = np.random.default_rng(SEED).permutation(s1_ids)[:TrainConfig().n_stage1_sample][: args.pairs]
    gt = pd.read_csv(d / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(set(sample_a)) & (gt["matched_entity_ids"] != "")]
    pairs = gt.assign(t=gt["matched_entity_ids"].str.split(",")).explode("t")[["source1_entity_id", "t"]]

    A = read_ids(d / "train_source1.tsv", set(pairs["source1_entity_id"]), cols)
    T = pd.concat([read_ids(d / f"train_source{k}.tsv", set(pairs["t"]), cols) for k in (2, 3)])
    pairs = pairs[pairs["t"].isin(T.index)]
    a, b = A.loc[pairs["source1_entity_id"]], T.loc[pairs["t"]]
    print(f"mining {len(pairs):,} true pairs from {pairs['source1_entity_id'].nunique():,} sample-A S1 entities")

    sub, added, dropped = collections.Counter(), collections.Counter(), collections.Counter()
    for (na, aa, c), (nb, ab) in zip(a[["business_name", "business_address", "country"]].itertuples(index=False),
                                     b[["business_name", "business_address"]].itertuples(index=False)):
        x, y = set(address_fields(aa)[1].split()), set(address_fields(ab)[1].split())
        if len(x - y) == 1 and len(y - x) == 1:
            sub[(c, next(iter(x - y)), next(iter(y - x)))] += 1
        x, y = set(name_fields(na)[1].split()), set(name_fields(nb)[1].split())
        for t in y - x:
            added[(c, t)] += 1
        for t in x - y:
            dropped[(c, t)] += 1

    def show(title, counter):
        print(f"\n{title} (support >= {args.min_support})")
        for k, v in counter.most_common(args.top):
            if v >= args.min_support:
                print(f"  {v:6d}  {k}")

    show("address 1:1 substitutions (country, S1 token, target token)", sub)
    show("name tokens added by the matching record (country, token)", added)
    show("name tokens dropped by the matching record (country, token)", dropped)


if __name__ == "__main__":
    main()

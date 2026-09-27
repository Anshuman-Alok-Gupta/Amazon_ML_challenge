"""Central configuration: paths, blocking sizes, model params."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# <package_root>/code/business_entity_resolution/src/config.py -> package root is 3 levels up.
# In the repo that is D:\Amazon_ml; in the submission zip it is the zip root (where output/ lives).
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent                      # code/business_entity_resolution
PACKAGE_ROOT = SRC_DIR.parents[2]

DEFAULT_DATA_DIR = Path(
    os.environ.get("ER_DATA_DIR", PACKAGE_ROOT / "student_resource" / "dataset")
)
DEFAULT_OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", PACKAGE_ROOT / "output"))
DEFAULT_ARTIFACT_DIR = Path(os.environ.get("ER_ARTIFACT_DIR", PROJECT_DIR / "artifacts"))
DEFAULT_CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", PROJECT_DIR / "cache"))

SEED = 42


@dataclass
class BlockingConfig:
    # Restrict candidates to records with the same (normalised) country string. Training data
    # shows true pairs always share the label. Country is an open label set: grouping is by
    # whatever strings appear, so France is handled like any other country.
    block_by_country: bool = True
    max_df: int = 20_000      # tokens more frequent than this get zero blocking weight
    addr_weight: float = 1.0  # weight of the address block inside the combined view
    k_comb: int = 20          # retrieval pool: top-k per S1 per target source, name+address view
    k_name: int = 15          # ... name-only view (address missing / reformatted)
    k_addr: int = 20          # ... address-only view (DBA names, native-script names)
    min_score: float = 0.05   # ignore retrieval scores below this
    keep_per_source: int = 8  # candidates kept per (S1, source) after stage-1 re-ranking
    stage1_min: float = 0.005 # ... and only if the stage-1 probability reaches this
    chunk_rows: int = 50_000  # S1 rows per sparse top-n call
    query_chunk: int = 50_000   # S1 rows per pool/re-rank batch (~5M pool rows; bounds memory)
    n_threads: int = max(1, (os.cpu_count() or 2))


@dataclass
class TrainConfig:
    # Train S1 entities used for model fitting. Their candidates are still retrieved from the
    # FULL target pool and "target best" scores use the FULL S1 pool, so features match test.
    n_s1_sample: int = 200_000   # sample B: stage-2 (matcher) training
    n_stage1_sample: int = 60_000  # sample A (disjoint from B): stage-1 re-ranker training
    stage1_rounds: int = 300
    n_ce_sample: int = 150_000   # sample C (disjoint from A and B): cross-encoder training


@dataclass
class ModelConfig:
    n_folds: int = 4
    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.08,
        "num_leaves": 127,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": SEED,
        "num_threads": 0,
    })
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    early_stopping_min_delta: float = 1e-5  # stop once logloss stops improving meaningfully


@dataclass
class PostConfig:
    # Each S2/S3 record is linked to at most one S1 entity (S1 is deduplicated).
    exclusive_candidates: bool = True
    threshold: float = 0.5    # overwritten by the tuned value saved in artifacts


@dataclass
class CEConfig:
    model: str = "intfloat/multilingual-e5-small"   # MIT license, 118M parameters
    max_len: int = 128            # tokens for both records together
    batch_size: int = 64
    score_batch_size: int = 512
    lr: float = 5e-5
    warmup: float = 0.06
    epochs: int = 1
    val_frac: float = 0.02
    max_train_pairs: int | None = 300_000   # CPU budget; None = all of sample C
    # only pairs whose level-0 probability lies in this band are scored (the rest get NaN, which
    # the tree models treat as "not scored"); None scores every pair
    band: tuple[float, float] | None = (0.02, 0.98)


@dataclass
class StackConfig:
    use_xgb: bool = True
    use_cat: bool = False          # CatBoost is slow on CPU; enable on a GPU machine
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.08,
        "grow_policy": "lossguide", "max_leaves": 127, "max_depth": 0, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "tree_method": "hist",
        "max_bin": 256, "seed": SEED,
    })
    cat_params: dict = field(default_factory=lambda: {
        "loss_function": "Logloss", "learning_rate": 0.1, "depth": 8, "random_seed": SEED,
    })
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100

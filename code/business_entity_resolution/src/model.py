"""LightGBM pairwise matcher: grouped CV for out-of-fold scores, full refit, persistence."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import ModelConfig


def assign_folds(groups: np.ndarray, n_folds: int) -> np.ndarray:
    """Fold number of every row: GroupKFold by S1 entity, so an entity's candidates share a fold.

    Saved with the training frame, so every model stacked on it later uses the same folds.
    """
    fold = np.empty(len(groups), np.int8)
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups)):
        fold[va] = k
    return fold


def train_cv(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, cfg: ModelConfig):
    """Out-of-fold LightGBM probabilities for the given fold assignment (see assign_folds)."""
    oof = np.zeros(len(X), dtype=np.float64)
    best_iters = []
    for fold in range(int(folds.max()) + 1):
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        booster = lgb.train(
            cfg.params,
            lgb.Dataset(X.iloc[tr], y[tr]),
            num_boost_round=cfg.num_boost_round,
            valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False,
                                          min_delta=cfg.early_stopping_min_delta)],
        )
        oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        best_iters.append(booster.best_iteration)
        print(f"  fold {fold}: best_iter={booster.best_iteration}")
    return oof, best_iters


def fit(X: pd.DataFrame, y: np.ndarray, cfg: ModelConfig, num_rounds: int) -> lgb.Booster:
    return lgb.train(cfg.params, lgb.Dataset(X, y), num_boost_round=max(num_rounds, 1))


def save(booster: lgb.Booster, meta: dict, artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(artifact_dir / "lgbm.txt"))
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load(artifact_dir: Path) -> tuple[lgb.Booster, dict]:
    artifact_dir = Path(artifact_dir)
    booster = lgb.Booster(model_file=str(artifact_dir / "lgbm.txt"))
    meta = json.loads((artifact_dir / "meta.json").read_text())
    return booster, meta

"""Train XGBoost classifiers for race winner + podium prediction."""
from __future__ import annotations

import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from f1_predictor.config import FEATURE_COLS, MODEL_PATH, MODELS_DIR, TRAIN_SEASONS
from f1_predictor.features import build_feature_table
from f1_predictor.ingest import ingest_seasons, load_raw

logger = logging.getLogger(__name__)


def _build_training_table(refresh: bool = False) -> pd.DataFrame:
    raw = load_raw()
    if refresh or any(raw[k].empty for k in ("results", "qualifying", "schedule")):
        logger.info("Refreshing raw data from APIs (this may take a few minutes)…")
        raw = ingest_seasons(TRAIN_SEASONS, use_fastf1=True)

    return build_feature_table(
        results=raw["results"],
        qualifying=raw["qualifying"],
        standings=raw["standings"],
        weather=raw.get("weather"),
        fp_pace=raw.get("fp_pace"),
    )


def train(refresh: bool = False) -> dict:
    df = _build_training_table(refresh=refresh)
    if df.empty:
        raise RuntimeError("No training data available. Run ingestion first.")

    df = df.dropna(subset=["is_winner", "is_podium"])
    df = df.fillna(df[FEATURE_COLS].median(numeric_only=True))

    X = df[FEATURE_COLS].values
    y_win = df["is_winner"].values
    y_pod = df["is_podium"].values
    groups = (df["season"].astype(str) + "_" + df["round"].astype(str)).values

    cv = GroupKFold(n_splits=5)

    def _cv_auc(y: np.ndarray) -> tuple[float, float]:
        scores = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model = _make_model(y[train_idx])
            model.fit(X[train_idx], y[train_idx])
            scores.append(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))
        return float(np.mean(scores)), float(np.std(scores))

    logger.info("Cross-validating winner model…")
    win_mean, win_std = _cv_auc(y_win)
    logger.info("Cross-validating podium model…")
    pod_mean, pod_std = _cv_auc(y_pod)
    logger.info("Winner CV AUC: %.3f ± %.3f", win_mean, win_std)
    logger.info("Podium CV AUC: %.3f ± %.3f", pod_mean, pod_std)

    winner_model = _make_model(y_win).fit(X, y_win)
    podium_model = _make_model(y_pod).fit(X, y_pod)

    bundle = {
        "winner": winner_model,
        "podium": podium_model,
        "features": FEATURE_COLS,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": {
            "winner_cv_auc_mean": win_mean,
            "winner_cv_auc_std": win_std,
            "podium_cv_auc_mean": pod_mean,
            "podium_cv_auc_std": pod_std,
        },
        "training_rows": int(len(df)),
        "training_seasons": TRAIN_SEASONS,
    }
    joblib.dump(bundle, MODEL_PATH)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))
    logger.info("Saved model → %s", MODEL_PATH)
    return bundle["metrics"]


def _make_model(y: np.ndarray) -> XGBClassifier:
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        tree_method="hist",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(train(refresh=False), indent=2))

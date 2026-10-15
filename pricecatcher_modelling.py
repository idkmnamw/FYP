"""Shared, explicit checks and calculations for Notebook 06."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def load_experiment(project_root, name, dates):
    """Load train/validation only. Test is deliberately loaded later in the notebook."""
    folder = Path(project_root) / "experiments" / name / "05_modelling"
    if not (folder / "experiment_settings.json").exists():
        raise FileNotFoundError(f"Missing {name} preparation. Run python prepare.py --experiment all first.")
    settings = json.loads((folder / "experiment_settings.json").read_text())
    if tuple(settings[key] for key in ["train_end", "validation_end", "test_end"]) != tuple(dates):
        raise ValueError(f"{name}: preparation dates differ from configuration; rerun 02–05.")
    if settings["preparation"]["experiment"] != name:
        raise ValueError(f"{name}: experiment metadata mismatch.")
    frames = {}
    for split in ["train", "validation"]:
        frames[split] = check_frame(pd.read_parquet(folder / f"{split}.parquet"), settings, split)
    if frames["train"].surge_label.nunique() != 2:
        raise ValueError(f"{name}: training needs both classes. Inspect the coverage/label audits.")
    return folder, settings, frames


def check_frame(frame, settings, split):
    """Validate the modelling contract and time horizon without changing any rows."""
    features = settings["feature_columns"]
    required = features + ["surge_label", "week_start", "next_week_end", "item", "unit"]
    declared = settings["numeric_features"] + settings["categorical_features"]
    if set(declared) != set(features) or len(declared) != len(features):
        raise ValueError("Numeric/categorical columns must partition the declared features exactly.")
    if len(features) != len(set(features)) or any(c.startswith("next_") or c in ["surge_label", "split"] for c in features):
        raise ValueError("Invalid feature list: identifiers/outcomes must not enter X.")
    if frame.empty or set(required) - set(frame.columns):
        raise ValueError(f"{split}: empty data or missing modelling columns.")
    if frame.surge_label.isna().any() or not frame.surge_label.isin([0, 1]).all():
        raise ValueError(f"{split}: labels must be observed 0/1 values.")
    frame = frame.copy()
    for column in ["week_start", "next_week_end"]:
        frame[column] = pd.to_datetime(frame[column])
    if frame[["week_start", "next_week_end"]].isna().any().any():
        raise ValueError("Missing split dates.")
    if not frame.week_start.dt.dayofweek.eq(0).all() or not frame.next_week_end.eq(frame.week_start + pd.Timedelta(days=13)).all():
        raise ValueError("Expected Monday-start weeks and exact next-week outcome horizons.")
    upper = pd.Timestamp(settings[{"train": "train_end", "validation": "validation_end", "test": "test_end"}[split]])
    if not frame.next_week_end.le(upper).all():
        raise ValueError(f"{split}: outcome crosses its cutoff.")
    if split != "train":
        lower = pd.Timestamp(settings["train_end" if split == "validation" else "validation_end"])
        if not (frame.week_start + pd.Timedelta(days=6)).gt(lower).all():
            raise ValueError(f"{split}: prediction overlaps the earlier split.")
    keys = ["item_code", "state", "district", "premise_type", "week_start"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{split}: duplicate local series/week rows.")
    if np.isinf(frame[settings["numeric_features"]].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric features; inspect preparation.")
    return frame


def model_inputs(frame, settings):
    """Select the whitelist; normalise categorical types without learning from data."""
    X = frame[settings["feature_columns"]].copy()
    for column in settings["categorical_features"]:
        X[column] = X[column].astype("string").fillna("__missing__").astype(str)
    return X


def build_pipeline(family, parameters, settings, y_train, *, seed=42, n_jobs=2, trees=150):
    """A fresh preprocessor and estimator for every candidate/fold."""
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if family == "Logistic Regression":
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), settings["numeric_features"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), settings["categorical_features"]),
    ], sparse_threshold=1.0)
    # No SMOTE: weighting uses this training fold only and preserves observed categories.
    if family == "Logistic Regression":
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed, **parameters)
    elif family == "Random Forest":
        model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs, **parameters)
    elif family == "XGBoost":
        positives = int(np.asarray(y_train).sum())
        if not 0 < positives < len(y_train):
            raise ValueError("XGBoost weighting requires both training classes.")
        model = XGBClassifier(n_estimators=trees, learning_rate=.05, subsample=.8,
            colsample_bytree=.8, tree_method="hist", eval_metric="logloss",
            scale_pos_weight=(len(y_train) - positives) / positives,
            random_state=seed, n_jobs=n_jobs, **parameters)
    elif family == "LightGBM":
        model = LGBMClassifier(n_estimators=trees, learning_rate=.05, min_child_samples=30,
            colsample_bytree=.8, class_weight="balanced", verbosity=-1,
            random_state=seed, n_jobs=n_jobs, **parameters)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def evaluate(y, scores, predictions):
    """AP is sklearn average precision, not trapezoidal PR-AUC. Undefined AP is NaN."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    if len(y) == 0 or len(scores) != len(y) or len(predictions) != len(y) or not np.isfinite(scores).all():
        raise ValueError("Metrics need aligned nonempty labels, finite scores and predictions.")
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    has_positives = bool(y.sum())
    return {"rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "predicted_surges": int(predictions.sum()),
            "average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)) if has_positives else np.nan,
            "f1": float(f1_score(y, predictions, zero_division=0)) if has_positives else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_cutoff(y_validation, scores):
    """Maximise validation F1; break ties with the highest cutoff (fewer alerts)."""
    if len(np.unique(y_validation)) != 2:
        raise ValueError("Cutoff selection needs both validation classes. Inspect labels; do not use test to choose it.")
    precision, recall, cutoffs = precision_recall_curve(y_validation, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros(len(cutoffs)), where=(precision[:-1] + recall[:-1]) > 0)
    return float(cutoffs[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]])


def baseline_scores(frame, surge_threshold):
    """A constant classifier and a fixed current-change rule, with no fitted inputs."""
    current_change = frame["weekly_price_change_pct"].to_numpy(dtype=float)
    if not np.isfinite(current_change).all():
        raise ValueError("The current-change baseline requires the prior-week price eligibility rule.")
    return {
        "Always no surge": (np.zeros(len(frame)), np.zeros(len(frame), dtype=int)),
        "Current-change rule": (current_change, (current_change >= surge_threshold).astype(int)),
    }


def grouped_metrics(frame, scores, predictions, level):
    """Report support and undefined metrics explicitly for each geography/food group."""
    results = []
    for group, positions in frame.groupby(level, sort=True).indices.items():
        group = group if isinstance(group, tuple) else (group,)
        results.append({**dict(zip(level, group)), **evaluate(frame.iloc[positions].surge_label,
                        np.asarray(scores)[positions], np.asarray(predictions)[positions])})
    return pd.DataFrame(results)

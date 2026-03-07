"""CLI for training baseline `is_like` models (logistic, lightgbm).

Usage examples:
  python -m src.train_baseline --config configs/data.yaml --model logistic
  python -m src.train_baseline --config configs/data.yaml --model lightgbm
"""
from pathlib import Path
import argparse
import yaml
import json
import os

import pandas as pd
import numpy as np

from .models.baseline import train_logistic, train_lightgbm
from .evaluate import compute_classification_metrics, ranking_metrics
from .feature_registry import get_training_columns, validate_no_banned_columns
from .utils import get_git_commit, write_run_metadata, ensure_dir
from .seed import set_seed
import time


def load_processed(processed_dir: str):
    processed_dir = Path(processed_dir)
    X = pd.read_csv(processed_dir / "X.csv")
    y = pd.read_csv(processed_dir / "y.csv")
    meta = pd.read_csv(processed_dir / "meta.csv") if (processed_dir / "meta.csv").exists() else pd.DataFrame()
    # indices
    def read_idx(name):
        p = processed_dir / name
        if p.exists():
            return pd.read_csv(p, header=None).iloc[:, 0].tolist()
        return []

    train_idx = read_idx("train_idx.csv")
    val_idx = read_idx("val_idx.csv")
    test_idx = read_idx("test_idx.csv")
    return X, y.squeeze() if isinstance(y, pd.DataFrame) and y.shape[1] == 1 else y, meta, train_idx, val_idx, test_idx


def save_predictions(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def run(config_path: str, model_name: str):
    cfg = yaml.safe_load(open(config_path))
    # set seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    processed_dir = cfg.get("processed_dir", "data/processed")
    X, y, meta, train_idx, val_idx, test_idx = load_processed(processed_dir)

    # ensure indices available
    if not train_idx or not val_idx:
        raise RuntimeError("Train/val indices not found in processed dir; run preprocessing first")

    # validate banned columns
    validate_no_banned_columns(X)

    # pick columns
    train_cols = get_training_columns(X)
    # Partition types: infer numeric and categorical
    numeric_cols = X[train_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_cols if c not in numeric_cols]

    # Prepare splits
    X_train = X.loc[train_idx, train_cols]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx, train_cols]
    y_val = y.loc[val_idx]
    X_test = X.loc[test_idx, train_cols] if test_idx else pd.DataFrame()
    y_test = y.loc[test_idx] if test_idx else pd.Series(dtype=int)

    artifacts_dir = Path("artifacts/models")
    metrics_dir = Path("reports/metrics")
    preds_dir = Path("reports/predictions")

    if model_name == "logistic":
        model_path = artifacts_dir / "logistic_pipeline.joblib"
        model = train_logistic(X_train, y_train, numeric_cols=numeric_cols, categorical_cols=categorical_cols, save_path=str(model_path))
    elif model_name == "lightgbm":
        model_path = artifacts_dir / "lightgbm_pipeline.joblib"
        model = train_lightgbm(X_train, y_train, numeric_cols=numeric_cols, categorical_cols=categorical_cols, save_path=str(model_path))
    else:
        raise ValueError(f"unknown model: {model_name}")

    # Predict
    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_classification_metrics(y_val.values, val_prob)
    val_metrics.update(ranking_metrics(meta.loc[val_idx] if not meta.empty else pd.DataFrame(), y_val.reset_index(drop=True), pd.Series(val_prob), ks=[1, 5, 10]))

    test_metrics = None
    if not X_test.empty:
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_classification_metrics(y_test.values, test_prob)
        test_metrics.update(ranking_metrics(meta.loc[test_idx] if not meta.empty else pd.DataFrame(), y_test.reset_index(drop=True), pd.Series(test_prob), ks=[1, 5, 10]))

    # Save metrics
    metrics_out = {"val": val_metrics, "test": test_metrics}
    save_metrics(metrics_out, metrics_dir / f"{model_name}_metrics.json")

    # write run metadata for reproducibility
    ensure_dir("reports/runs")
    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": config_path,
        "config": cfg,
        "git_commit": get_git_commit(),
        "dataset": {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)},
        "metrics": metrics_out,
    }
    write_run_metadata(f"reports/runs/{model_name}_run_{int(time.time())}.json", run_meta)

    # Save prediction CSVs with id, truth, prob
    def make_pred_df(idx_list, probs, split_name):
        # Prefer to include id columns from meta (processed meta contains ids + timestamp)
        out = pd.DataFrame()
        if not meta.empty and any(c in meta.columns for c in ["user_id", "video_id"]):
            for col in ["user_id", "video_id"]:
                if col in meta.columns:
                    out[col] = meta.loc[idx_list, col].values
        else:
            out["row_index"] = idx_list
        out["y_true"] = y.loc[idx_list].values
        out["y_prob"] = probs
        return out

    val_df = make_pred_df(val_idx, val_prob, "val")
    save_predictions(val_df, preds_dir / f"{model_name}_val_predictions.csv")
    if test_metrics is not None:
        test_df = make_pred_df(test_idx, test_prob, "test")
        save_predictions(test_df, preds_dir / f"{model_name}_test_predictions.csv")

    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_dir / f"{model_name}_metrics.json")
    print("Saved predictions to:", preds_dir)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["logistic", "lightgbm"])
    args = parser.parse_args()
    run(args.config, args.model)


if __name__ == "__main__":
    cli()

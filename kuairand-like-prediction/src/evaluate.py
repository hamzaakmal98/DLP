from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = None
    try:
        out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)))
    except Exception:
        out["log_loss"] = None
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def precision_at_k_by_user(meta: pd.DataFrame, y_true: pd.Series, y_prob: pd.Series, k: int = 5, user_col: str = "user_id") -> float:
    """Compute mean precision@k across users.

    If `user_col` is not present, compute global precision@k across dataset.
    """
    if user_col not in meta.columns:
        order = np.argsort(-np.array(y_prob))
        topk = order[:k]
        return float(np.mean(np.array(y_true)[topk]))

    df = pd.DataFrame({"user_id": meta[user_col].values, "y_true": y_true.values, "y_prob": y_prob.values})
    precisions = []
    for uid, g in df.groupby("user_id"):
        g_sorted = g.sort_values("y_prob", ascending=False)
        top = g_sorted.head(k)
        if len(top) == 0:
            continue
        precisions.append(top["y_true"].mean())
    return float(np.nanmean(precisions)) if precisions else 0.0


def ranking_metrics(meta: pd.DataFrame, y_true: pd.Series, y_prob: pd.Series, ks: List[int] = [1, 5, 10], user_col: str = "user_id") -> Dict:
    out = {}
    for k in ks:
        out[f"prec_at_{k}"] = precision_at_k_by_user(meta, y_true, y_prob, k=k, user_col=user_col)
    return out

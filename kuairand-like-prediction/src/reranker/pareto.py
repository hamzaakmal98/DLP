import numpy as np
import pandas as pd
from typing import Tuple


def pareto_front(points: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points.

    points: (n_items, n_tasks) array of objective values.
    maximize: if True, larger is better; otherwise smaller is better.
    Returns: boolean array length n_items, True if point is on Pareto front.
    """
    if not maximize:
        pts = -points
    else:
        pts = points

    n_points = pts.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_efficient[i]:
            continue
        # a point j dominates i if j >= i on all dims and > on at least one
        domination = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        is_efficient[domination] = False
    return is_efficient


def scalarize_rank(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute scalarized score (weighted sum) and return descending order indices."""
    scores = points.dot(weights)
    return np.argsort(-scores)


def topk_by_pareto(df: pd.DataFrame, score_cols: Tuple[str], k: int = 10) -> pd.DataFrame:
    """Return top-k items per user using Pareto front selection then fallback to scalarization.

    df: must contain 'user_id' and item id column (keeps all other columns).
    score_cols: tuple of column names to use as objectives (e.g. ('like','longview')).
    k: number of items to select per user.
    """
    out_rows = []
    if 'user_id' not in df.columns:
        # global selection
        points = df[list(score_cols)].to_numpy()
        mask = pareto_front(points, maximize=True)
        pareto_df = df[mask]
        if len(pareto_df) >= k:
            return pareto_df.head(k)
        # fallback scalarize
        weights = np.ones(len(score_cols))
        idxs = scalarize_rank(points, weights)
        return df.iloc[idxs[:k]]

    for uid, group in df.groupby('user_id'):
        points = group[list(score_cols)].to_numpy()
        mask = pareto_front(points, maximize=True)
        pareto_group = group[mask]
        if len(pareto_group) >= k:
            sel = pareto_group
        else:
            # fallback to scalarized ranking with equal weights
            weights = np.ones(len(score_cols))
            idxs = scalarize_rank(points, weights)
            sel = group.iloc[idxs[:k]]
        out_rows.append(sel.head(k))
    if not out_rows:
        return df.iloc[0:0]
    return pd.concat(out_rows).reset_index(drop=True)

"""Train a PyTorch MLP for is_like prediction.

Usage:
  python -m src.train_mlp --config configs/mlp.yaml
"""
from pathlib import Path
import argparse
import json
import time
import random

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .models.mlp import MLPModel
from .evaluate import compute_classification_metrics, ranking_metrics
from .feature_registry import get_training_columns, validate_no_banned_columns
from .utils import ensure_dir, get_git_commit, write_run_metadata
from .seed import set_seed
import time


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InteractionDataset(Dataset):
    def __init__(self, X_numeric: np.ndarray, X_cats: list, y: np.ndarray):
        self.X_numeric = X_numeric.astype(np.float32)
        self.X_cats = [x.astype(np.int64) for x in X_cats] if X_cats else []
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        num = torch.from_numpy(self.X_numeric[idx])
        cats = [torch.from_numpy(col[idx]) for col in self.X_cats] if self.X_cats else []
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return num, cats, y


def build_category_maps(X_train: pd.DataFrame, categorical_cols: list):
    maps = {}
    cardinalities = []
    for c in categorical_cols:
        vals = X_train[c].fillna("__MISSING__").astype(str).unique().tolist()
        mapping = {v: i + 1 for i, v in enumerate(vals)}  # reserve 0 for unknown/missing
        maps[c] = mapping
        cardinalities.append(len(mapping))
    return maps, cardinalities


def apply_maps(X: pd.DataFrame, categorical_cols: list, maps: dict):
    cat_arrays = []
    for c in categorical_cols:
        mapping = maps.get(c, {})
        arr = X[c].fillna("__MISSING__").astype(str).map(lambda v: mapping.get(v, 0)).to_numpy(dtype=np.int64)
        cat_arrays.append(arr)
    return cat_arrays


def collate_fn(batch):
    nums = [b[0].numpy() for b in batch]
    cats = [ [c.numpy() for c in b[1]] for b in batch ]
    ys = [b[2].item() for b in batch]
    nums = np.stack(nums)
    if cats and cats[0]:
        # transpose list-of-lists into list of arrays per cat
        cat_arrays = [np.array([row[i] for row in cats]) for i in range(len(cats[0]))]
    else:
        cat_arrays = []
    return torch.from_numpy(nums), [torch.from_numpy(a) for a in cat_arrays], torch.tensor(ys, dtype=torch.float32)


def train_loop(model, optim, loss_fn, dl):
    model.train()
    total_loss = 0.0
    for xb_num, xb_cats, yb in dl:
        xb_num = xb_num.to(device)
        xb_cats = [c.to(device) for c in xb_cats]
        yb = yb.to(device)
        optim.zero_grad()
        logits = model(xb_num, xb_cats)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        total_loss += loss.item() * yb.size(0)
    return total_loss / len(dl.dataset)


@torch.no_grad()
def eval_loop(model, dl):
    model.eval()
    ys = []
    probs = []
    for xb_num, xb_cats, yb in dl:
        xb_num = xb_num.to(device)
        xb_cats = [c.to(device) for c in xb_cats]
        logits = model(xb_num, xb_cats)
        p = torch.sigmoid(logits)
        ys.append(yb.numpy())
        probs.append(p.cpu().numpy())
    ys = np.concatenate(ys)
    probs = np.concatenate(probs)
    return ys, probs


def run(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    processed_dir = cfg.get("processed_dir", "data/processed")
    processed_dir = Path(processed_dir)
    X = pd.read_csv(processed_dir / "X.csv")
    y = pd.read_csv(processed_dir / "y.csv")
    y = y.iloc[:, 0] if y.shape[1] == 1 else y.squeeze()
    meta = pd.read_csv(processed_dir / "meta.csv") if (processed_dir / "meta.csv").exists() else pd.DataFrame()
    train_idx = pd.read_csv(processed_dir / "train_idx.csv", header=None).iloc[:, 0].tolist()
    val_idx = pd.read_csv(processed_dir / "val_idx.csv", header=None).iloc[:, 0].tolist()
    test_idx = pd.read_csv(processed_dir / "test_idx.csv", header=None).iloc[:, 0].tolist() if (processed_dir / "test_idx.csv").exists() else []

    # validate
    validate_no_banned_columns(X)

    # choose training columns
    train_cols = cfg.get("feature_cols") or get_training_columns(X)
    numeric_cols = X[train_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_cols if c not in numeric_cols]

    # build maps and datasets
    X_train = X.loc[train_idx, train_cols].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True).to_numpy(dtype=np.int64)
    X_val = X.loc[val_idx, train_cols].reset_index(drop=True)
    y_val = y.loc[val_idx].reset_index(drop=True).to_numpy(dtype=np.int64)
    X_test = X.loc[test_idx, train_cols].reset_index(drop=True) if test_idx else pd.DataFrame()
    y_test = y.loc[test_idx].reset_index(drop=True).to_numpy(dtype=np.int64) if test_idx else np.array([])

    maps, cardinalities = build_category_maps(X_train, categorical_cols)
    X_train_cats = apply_maps(X_train, categorical_cols, maps)
    X_val_cats = apply_maps(X_val, categorical_cols, maps)
    X_test_cats = apply_maps(X_test, categorical_cols, maps) if not X_test.empty else []

    # numeric normalization
    numeric_mean = X_train[numeric_cols].mean().to_numpy(dtype=np.float32) if numeric_cols else np.array([])
    numeric_std = X_train[numeric_cols].std().replace(0, 1).to_numpy(dtype=np.float32) if numeric_cols else np.array([])
    def norm(df):
        return ((df[numeric_cols].to_numpy(dtype=np.float32) - numeric_mean) / numeric_std) if numeric_cols else np.zeros((len(df), 0), dtype=np.float32)

    X_train_num = norm(X_train)
    X_val_num = norm(X_val)
    X_test_num = norm(X_test) if not X_test.empty else np.zeros((0, 0), dtype=np.float32)

    # datasets and loaders
    batch_size = cfg.get("batch_size", 128)
    train_ds = InteractionDataset(X_train_num, X_train_cats, y_train)
    val_ds = InteractionDataset(X_val_num, X_val_cats, y_val)
    test_ds = InteractionDataset(X_test_num, X_test_cats, y_test) if X_test.size != 0 else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None

    # device and seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    global device
    # Prefer CUDA if available, otherwise try DirectML (for AMD on Windows), fall back to CPU
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                import torch_directml

                device = torch_directml.device()
            except Exception:
                device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    # model
    hidden_sizes = cfg.get("hidden_sizes", [128, 64])
    dropout = cfg.get("dropout", 0.2)
    lr = cfg.get("lr", 1e-3)
    epochs = cfg.get("epochs", 10)
    patience = cfg.get("patience", 3)

    model = MLPModel(num_numeric=len(numeric_cols), categorical_cardinalities=cardinalities, hidden_sizes=hidden_sizes, dropout=dropout)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_path = Path("artifacts/models/mlp_best.pth")
    ensure_dir(str(best_path.parent))
    history = {"train_loss": [], "val_loss": []}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_loop(model, optim, loss_fn, train_dl)
        ys_val, probs_val = eval_loop(model, val_dl)
        val_loss = float(F.binary_cross_entropy(torch.tensor(probs_val, dtype=torch.float32), torch.tensor(ys_val, dtype=torch.float32)).item())
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={time.time()-t0:.1f}s")
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    # load best
    model.load_state_dict(torch.load(best_path, map_location=device))

    # evaluate val and test
    ys_val, probs_val = eval_loop(model, val_dl)
    val_metrics = compute_classification_metrics(ys_val, probs_val)
    val_metrics.update(ranking_metrics(meta.loc[val_idx] if not meta.empty else pd.DataFrame(), pd.Series(ys_val), pd.Series(probs_val), ks=[1, 5, 10]))

    test_metrics = None
    if test_dl:
        ys_test, probs_test = eval_loop(model, test_dl)
        test_metrics = compute_classification_metrics(ys_test, probs_test)
        test_metrics.update(ranking_metrics(meta.loc[test_idx] if not meta.empty else pd.DataFrame(), pd.Series(ys_test), pd.Series(probs_test), ks=[1, 5, 10]))

    # Save artifacts
    ensure_dir("reports/metrics")
    ensure_dir("reports/predictions")
    ensure_dir("artifacts/models")
    with open("reports/metrics/mlp_metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    # history
    with open("reports/metrics/mlp_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # predictions
    val_df = pd.DataFrame({"y_true": ys_val, "y_prob": probs_val})
    val_df.to_csv("reports/predictions/mlp_val_predictions.csv", index=False)
    if test_dl:
        test_df = pd.DataFrame({"y_true": ys_test, "y_prob": probs_test})
        test_df.to_csv("reports/predictions/mlp_test_predictions.csv", index=False)

    print("Saved best checkpoint to", best_path)
    print("Saved metrics to reports/metrics/mlp_metrics.json")

    # write run metadata
    ensure_dir("reports/runs")
    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": config_path,
        "config": cfg,
        "git_commit": get_git_commit(),
        "dataset": {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)},
        "metrics": {"val": val_metrics, "test": test_metrics},
    }
    write_run_metadata(f"reports/runs/mlp_run_{int(time.time())}.json", run_meta)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    cli()

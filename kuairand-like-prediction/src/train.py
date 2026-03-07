import argparse
import yaml
from .data_loader import load_csv
from .dataset import build_dataset, temporal_train_test_split
from .features import simple_preprocess
from .models.logreg import train_logreg
from .models.lgbm_model import train_lgbm
from .models.torch_mlp import train_mlp
from .eval import evaluate, precision_at_k_by_user
import os
import joblib
import numpy as np


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["data"]["path"]
    target = cfg["data"].get("target", "is_like")
    ts_col = cfg["data"].get("timestamp_col")

    df = load_csv(data_path)
    X, y, meta = build_dataset(df, target=target, timestamp_col=ts_col)
    X, scaler = simple_preprocess(X)

    # temporal split if possible
    masks = temporal_train_test_split(meta, test_size=cfg["train"].get("test_size", 0.2), timestamp_col=ts_col)
    if masks is None:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["train"].get("test_size", 0.2), random_state=cfg["train"].get("random_state", 42))
        meta_train = meta.iloc[X_train.index]
        meta_test = meta.iloc[X_test.index]
    else:
        train_mask, test_mask = masks
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        meta_train, meta_test = meta.loc[train_mask], meta.loc[test_mask]

    out_dir = cfg["output"].get("dir", "models")
    os.makedirs(out_dir, exist_ok=True)

    models_to_run = cfg["train"].get("models", ["logreg"])
    results = {}
    if "logreg" in models_to_run:
        path = os.path.join(out_dir, "logreg.joblib")
        clf = train_logreg(X_train, y_train, C=cfg["models"]["logreg"].get("C", 1.0), save_path=path)
        preds = clf.predict_proba(X_test)[:, 1]
        results["logreg"] = evaluate(y_test, preds)

    if "lgbm" in models_to_run:
        path = os.path.join(out_dir, "lgbm.txt")
        bst = train_lgbm(X_train, y_train, num_boost_round=cfg["models"]["lgbm"].get("num_boost_round", 100), learning_rate=cfg["models"]["lgbm"].get("learning_rate", 0.1), save_path=path)
        preds = bst.predict(X_test)
        results["lgbm"] = evaluate(y_test, preds)

    if "mlp" in models_to_run:
        path = os.path.join(out_dir, "mlp.pth")
        model = train_mlp(X_train, y_train, epochs=cfg["models"]["mlp"].get("epochs", 5), batch_size=cfg["models"]["mlp"].get("batch_size", 128), lr=cfg["models"]["mlp"].get("lr", 1e-3), save_path=path)
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            preds = model(xb).cpu().numpy()
        results["mlp"] = evaluate(y_test, preds)

    # compute a couple of top-k metrics if user_id present
    for name, r in results.items():
        try:
            preds = None
            if name == "logreg":
                preds = joblib.load(os.path.join(out_dir, "logreg.joblib")).predict_proba(X_test)[:, 1]
            elif name == "lgbm":
                import lightgbm as lgb
                preds = lgb.Booster(model_file=os.path.join(out_dir, "lgbm.txt")).predict(X_test)
            elif name == "mlp":
                import torch
                from models.torch_mlp import MLP
                model = MLP(X_test.shape[1])
                model.load_state_dict(torch.load(os.path.join(out_dir, "mlp.pth")))
                model.eval()
                preds = model(torch.tensor(X_test.values, dtype=torch.float32)).detach().numpy()
        except Exception:
            preds = None
        if preds is not None:
            r["precision_at_1_user"] = precision_at_k_by_user(meta_test, y_test, preds, k=1)
            r["precision_at_5_user"] = precision_at_k_by_user(meta_test, y_test, preds, k=5)

    print("Results:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)

import argparse
import os
import yaml
import math
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from src.reranker.pareto import topk_by_pareto
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from src.models.mmoe import MMoE
from src.evaluate import compute_metrics


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, cat_cols, num_cols, targets, mappings=None):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.targets = targets

        # build factorization mappings if not provided
        self.mappings = mappings or {}
        for c in cat_cols:
            if c not in self.mappings:
                codes, uniques = pd.factorize(self.X[c].astype('str'))
                self.mappings[c] = {u: i for i, u in enumerate(uniques)}
                self.X[c] = codes
            else:
                # map with unknown->0
                self.X[c] = self.X[c].map(self.mappings[c]).fillna(0).astype(int)

        if len(num_cols) > 0:
            # ensure numeric columns exist; if missing, fill with zeros (prevents crashes in smoke runs)
            for c in num_cols:
                if c not in self.X.columns:
                    self.X[c] = 0.0
            self.X[num_cols] = self.X[num_cols].astype(float).fillna(0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        x_cats = {c: torch.tensor(int(row[c]), dtype=torch.long) for c in self.cat_cols}
        x_nums = torch.tensor(row[self.num_cols].values.astype(float), dtype=torch.float32) if len(self.num_cols) > 0 else None
        labels = {t: torch.tensor(float(self.y.iloc[idx][t]), dtype=torch.float32) for t in self.targets if t in self.y.columns}
        return x_cats, x_nums, labels


def collate_fn(batch):
    # batch is list of (x_cats, x_nums, labels)
    cat_keys = batch[0][0].keys()
    batched_cats = {k: torch.stack([b[0][k] for b in batch]) for k in cat_keys}
    nums = None
    if batch[0][1] is not None:
        nums = torch.stack([b[1] for b in batch])
    labels = {}
    label_keys = batch[0][2].keys()
    for k in label_keys:
        labels[k] = torch.stack([b[2][k] for b in batch])
    return batched_cats, nums, labels


def load_feature_lists(processed_dir):
    cat_file = Path(processed_dir) / 'cols_categorical.txt'
    num_file = Path(processed_dir) / 'cols_numeric.txt'
    cat_cols = []
    num_cols = []
    if cat_file.exists():
        cat_cols = [l.strip() for l in open(cat_file) if l.strip()]
    if num_file.exists():
        num_cols = [l.strip() for l in open(num_file) if l.strip()]
    return cat_cols, num_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # minimal console logging mode — print only epoch, train loss and val loss
    minimal_logging = cfg.get('minimal_console_logging', True)

    # data paths (assume processed data dir)
    processed = Path('data/processed')
    # read processed files without forcing the first column as index to be robust
    X = pd.read_csv(processed / 'X.csv')
    y = pd.read_csv(processed / 'y.csv')
    # ensure stable default integer index alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # enforce leakage prevention
    forbidden = cfg.get('forbidden_target_columns', ['is_like', 'is_click', 'long_view', 'creator_interest'])
    for col in forbidden:
        if col in X.columns:
            X = X.drop(columns=[col])

    cat_cols, num_cols = load_feature_lists(processed)

    # optional creator task
    creator_enabled = cfg.get('creator_task_enabled', True)
    targets = ['is_like', 'long_view']
    if creator_enabled:
        targets.append('creator_interest')

    # split using provided idx files if available
    train_idx = np.loadtxt(processed / 'train_idx.csv', delimiter=',', dtype=int) if (processed / 'train_idx.csv').exists() else None
    val_idx = np.loadtxt(processed / 'val_idx.csv', delimiter=',', dtype=int) if (processed / 'val_idx.csv').exists() else None

    if train_idx is not None and val_idx is not None:
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
    else:
        # default 90/10 split
        n = len(X)
        cut = int(0.9 * n)
        X_train = X.iloc[:cut]
        y_train = y.iloc[:cut]
        X_val = X.iloc[cut:]
        y_val = y.iloc[cut:]

    # build mappings and cardinalities from train
    mappings = {}
    cat_cardinalities = {}
    for c in cat_cols:
        codes, uniques = pd.factorize(X_train[c].astype('str'))
        cat_cardinalities[c] = len(uniques) + 1
        mappings[c] = {u: i for i, u in enumerate(uniques)}

    embedding_dim = cfg.get('embedding_dim', 16)
    # allow per-feature dict
    if isinstance(embedding_dim, int):
        emb_cfg = embedding_dim
    else:
        emb_cfg = embedding_dim

    model = MMoE(cat_cardinalities=cat_cardinalities,
                 embedding_dim=emb_cfg,
                 numeric_feat_dims=len(num_cols),
                 num_experts=cfg.get('num_experts', 3),
                 expert_hidden=cfg.get('expert_hidden_size', 128),
                 tower_hidden=cfg.get('tower_hidden_size', 64),
                 creator_task_enabled=creator_enabled)

    # allow forcing a device via config (e.g. 'cuda', 'cuda:0', 'cpu') useful for AMD/ROCm setups
    cfg_device = cfg.get('device')
    if cfg_device == 'dml':
        # use DirectML on Windows for AMD GPUs via the torch-directml package
        try:
            import torch_directml

            device = torch_directml.device()
        except Exception as e:
            raise RuntimeError(
                "torch-directml is required for device='dml'. Install with `pip install torch-directml` and restart.`"
            ) from e
    elif cfg_device:
        device = torch.device(cfg_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = TabularDataset(X_train, y_train, cat_cols, num_cols, targets, mappings=mappings)
    val_ds = TabularDataset(X_val, y_val, cat_cols, num_cols, targets, mappings=mappings)

    # class imbalance handling: compute pos_weight from training labels when enabled
    use_pos_weight = cfg.get('use_pos_weight', True)
    pos_weights = {}
    if use_pos_weight:
        for t in targets:
            if t in y_train.columns:
                counts = y_train[t].value_counts()
                n_pos = int(counts.get(1, 0))
                n_neg = int(counts.get(0, 0))
                if n_pos > 0:
                    pos_weights[t] = float(n_neg) / max(1.0, float(n_pos))
                else:
                    pos_weights[t] = 1.0
            else:
                pos_weights[t] = 1.0

    batch_size = cfg.get('batch_size', 1024)
    # optional sampler to up/down-sample training examples (helps with imbalance)
    use_sampler = cfg.get('use_sampler', False)
    train_loader = None
    if use_sampler:
        from torch.utils.data import WeightedRandomSampler
        # build weights: higher weight for positive examples across primary task 'is_like' if present
        weights = []
        primary = cfg.get('primary_task', 'is_like')
        if primary in y_train.columns:
            pos_mask = y_train[primary].fillna(0).astype(int).values
            # assign weight 1.0 to negatives and (n_neg/n_pos) to positives
            n_pos = int((pos_mask == 1).sum())
            n_neg = int((pos_mask == 0).sum())
            pos_w = float(n_neg) / max(1.0, float(n_pos)) if n_pos > 0 else 1.0
            for v in pos_mask:
                weights.append(pos_w if v == 1 else 1.0)
        else:
            weights = [1.0] * len(train_ds)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optim_lr = cfg.get('learning_rate', 1e-3)
    epochs = cfg.get('epochs', 5)
    # support max_epochs (preferred) with fallback to epochs for backward compatibility
    max_epochs = cfg.get('max_epochs', epochs)
    patience = cfg.get('patience', 5)
    min_delta = cfg.get('min_delta', 1e-4)

    optimizer = optim.Adam(model.parameters(), lr=optim_lr)

    # losses (optionally using pos_weight computed from training labels)
    def _bce_with_posw(tname):
        if use_pos_weight and tname in pos_weights:
            # ensure pos_weight tensor lives on the same device as model tensors
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights[tname], device=device, dtype=torch.float32))
        return nn.BCEWithLogitsLoss()

    loss_like = _bce_with_posw('is_like')
    loss_long = _bce_with_posw('long_view')
    loss_creator = _bce_with_posw('creator_interest')

    best_val_loss = float('inf')
    out_dir = Path(cfg.get('checkpoint_dir', 'artifacts/models'))
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_batches = cfg.get('debug_print_batches', 0)
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        total_examples = 0
        batch_i = 0
        for xb_cats, xb_nums, yb in train_loader:
            batch_i += 1
            # move to device
            xb_cats = {k: v.to(device) for k, v in xb_cats.items()}
            xb_nums = xb_nums.to(device) if xb_nums is not None else None
            # get label tensors or None
            y_like = yb.get('is_like')
            y_like = y_like.to(device) if y_like is not None else None
            y_long = yb.get('long_view')
            y_long = y_long.to(device) if y_long is not None else None
            y_creator = yb.get('creator_interest')
            y_creator = y_creator.to(device) if y_creator is not None else None

            preds = model(xb_cats, xb_nums)

            # compute task losses only when labels exist
            l_terms = []
            weights = cfg.get('loss_weights', {'like': 1.0, 'long_view': 1.0, 'creator_interest': 1.0})
            if y_like is not None:
                l_like = loss_like(preds['like_logit'], y_like)
                l_terms.append(weights.get('like', 1.0) * l_like)
            else:
                l_like = None
            if y_long is not None:
                l_long = loss_long(preds['longview_logit'], y_long)
                l_terms.append(weights.get('long_view', 1.0) * l_long)
            else:
                l_long = None
            l_creator = None
            if creator_enabled and y_creator is not None:
                l_creator = loss_creator(preds['creator_logit'], y_creator)
                l_terms.append(weights.get('creator_interest', 1.0) * l_creator)

            if len(l_terms) == 0:
                # nothing to train on in this batch
                batch_loss_val = 0.0
            else:
                total_loss = sum(l_terms)
                optimizer.zero_grad()
                try:
                    total_loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    # Handle DirectML graph compile/runtime errors by falling back to CPU
                    errstr = str(e)
                    if not minimal_logging:
                        print('RuntimeError during backward():', errstr)
                        print('Falling back to CPU for training to avoid DML compile issues.')
                    # move model to CPU and recreate optimizer/losses on CPU
                    device = torch.device('cpu')
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(), lr=optim_lr)
                    # recreate losses so pos_weight tensors live on CPU
                    loss_like = _bce_with_posw('is_like')
                    loss_long = _bce_with_posw('long_view')
                    loss_creator = _bce_with_posw('creator_interest')

                    # move inputs to CPU and recompute forward/backward
                    xb_cats_cpu = {k: v.to(device) for k, v in xb_cats.items()}
                    xb_nums_cpu = xb_nums.to(device) if xb_nums is not None else None
                    y_like_cpu = y_like.to(device) if y_like is not None else None
                    y_long_cpu = y_long.to(device) if y_long is not None else None
                    y_creator_cpu = y_creator.to(device) if y_creator is not None else None

                    preds_cpu = model(xb_cats_cpu, xb_nums_cpu)
                    l_terms_cpu = []
                    if y_like_cpu is not None:
                        l_terms_cpu.append(weights.get('like', 1.0) * loss_like(preds_cpu['like_logit'], y_like_cpu))
                    if y_long_cpu is not None:
                        l_terms_cpu.append(weights.get('long_view', 1.0) * loss_long(preds_cpu['longview_logit'], y_long_cpu))
                    if creator_enabled and y_creator_cpu is not None:
                        l_terms_cpu.append(weights.get('creator_interest', 1.0) * loss_creator(preds_cpu['creator_logit'], y_creator_cpu))

                    if len(l_terms_cpu) > 0:
                        total_loss_cpu = sum(l_terms_cpu)
                        optimizer.zero_grad()
                        total_loss_cpu.backward()
                        optimizer.step()
                        batch_loss_val = float(total_loss_cpu.detach().cpu().item())
                    else:
                        batch_loss_val = 0.0
                else:
                    batch_loss_val = float(total_loss.detach().cpu().item())

            bsize = xb_nums.shape[0] if xb_nums is not None else (y_like.shape[0] if y_like is not None else 0)
            total_examples += bsize
            train_loss += batch_loss_val * (bsize if bsize > 0 else 0)

            if (not minimal_logging) and debug_batches and batch_i <= debug_batches:
                l_like_val = None if l_like is None else float(l_like.detach().cpu().item())
                l_long_val = None if l_long is None else float(l_long.detach().cpu().item())
                l_creator_val = None if l_creator is None else float(l_creator.detach().cpu().item())
                print(f"DEBUG epoch={epoch:2d} batch={batch_i:4d} bsize={bsize:4d} l_like={l_like_val!s:10} l_long={l_long_val!s:10} l_creator={l_creator_val!s:10} total={batch_loss_val:.6f}")

        train_loss = train_loss / total_examples if total_examples > 0 else 0.0

        # validation
        model.eval()
        val_loss = 0.0
        all_preds = {t: [] for t in targets}
        all_trues = {t: [] for t in targets}
        with torch.no_grad():
            val_examples = 0
            for xb_cats, xb_nums, yb in val_loader:
                xb_cats = {k: v.to(device) for k, v in xb_cats.items()}
                xb_nums = xb_nums.to(device) if xb_nums is not None else None
                y_like = yb.get('is_like')
                y_like = y_like.to(device) if y_like is not None else None
                y_long = yb.get('long_view')
                y_long = y_long.to(device) if y_long is not None else None
                y_creator = yb.get('creator_interest')
                y_creator = y_creator.to(device) if y_creator is not None else None

                preds = model(xb_cats, xb_nums)

                # compute total loss only over available labels
                v_terms = []
                if y_like is not None:
                    l_like = loss_like(preds['like_logit'], y_like)
                    v_terms.append(weights.get('like', 1.0) * l_like)
                if y_long is not None:
                    l_long = loss_long(preds['longview_logit'], y_long)
                    v_terms.append(weights.get('long_view', 1.0) * l_long)
                if creator_enabled and y_creator is not None:
                    l_creator = loss_creator(preds['creator_logit'], y_creator)
                    v_terms.append(weights.get('creator_interest', 1.0) * l_creator)

                batch_examples = xb_nums.shape[0] if xb_nums is not None else (y_like.shape[0] if y_like is not None else 0)
                val_examples += batch_examples
                if len(v_terms) > 0:
                    batch_val_loss = float(sum(v_terms).detach().cpu().item())
                    val_loss += batch_val_loss * batch_examples

                # collect preds
                if y_like is not None:
                    all_preds['is_like'].append(torch.sigmoid(preds['like_logit']).detach().cpu().numpy())
                    all_trues['is_like'].append(y_like.detach().cpu().numpy())
                if y_long is not None:
                    all_preds['long_view'].append(torch.sigmoid(preds['longview_logit']).detach().cpu().numpy())
                    all_trues['long_view'].append(y_long.detach().cpu().numpy())
                if creator_enabled and y_creator is not None:
                    all_preds['creator_interest'].append(torch.sigmoid(preds['creator_logit']).detach().cpu().numpy())
                    all_trues['creator_interest'].append(y_creator.detach().cpu().numpy())

            val_loss = val_loss / val_examples if val_examples > 0 else 0.0

        # aggregate metrics
        metrics = {}
        for t in targets:
            if len(all_preds[t]) > 0:
                preds = np.concatenate(all_preds[t])
                trues = np.concatenate(all_trues[t])
                metrics[t] = compute_metrics(trues, preds, topk=tuple(cfg.get('topk', [100])))
            else:
                metrics[t] = None

        # write per-epoch metrics JSON and print minimal/verbose epoch summary
        metrics_out_dir = Path(cfg.get('metrics_dir', out_dir / 'metrics'))
        metrics_out_dir.mkdir(parents=True, exist_ok=True)
        epoch_metrics_path = metrics_out_dir / f'metrics_epoch{epoch}.json'
        with open(epoch_metrics_path, 'w') as mf:
            json.dump(metrics, mf, indent=2)

        # also write/overwrite a combined report with latest epoch metrics
        report_path = out_dir / 'report_metrics.json'
        report_obj = {'epoch': epoch, 'metrics': metrics}
        with open(report_path, 'w') as rf:
            json.dump(report_obj, rf, indent=2)

        # pretty/concise epoch summary printed to console
        if minimal_logging:
            print(f"Epoch {epoch}/{max_epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        else:
            print('-' * 72)
            print(f"Epoch {epoch}/{max_epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
            for t, m in metrics.items():
                if m is None:
                    print(f"Task {t}: no predictions/labels for metrics")
                    continue
                roc = m.get('roc_auc')
                pr = m.get('pr_auc')
                ll = m.get('logloss')
                precs = {k: v for k, v in m.items() if k.startswith('prec_at_')}
                prec_str = ' '.join([f"{k}={v:.4f}" if v is not None else f"{k}=None" for k, v in precs.items()])
                roc_str = f"{roc:.4f}" if roc is not None else "None"
                pr_str = f"{pr:.6f}" if pr is not None else "None"
                ll_str = f"{ll:.6f}" if ll is not None else "None"
                print(f"Task {t}: roc_auc={roc_str}  pr_auc={pr_str}  logloss={ll_str}  {prec_str}")

        # early stopping based on validation loss
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # save best checkpoint
            best_path = out_dir / 'mmoe_best.pt'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, best_path)
            if not minimal_logging:
                print(f"Saved checkpoint to {best_path}")
        else:
            no_improve += 1
            if not minimal_logging:
                print(f"No improvement for {no_improve} epoch(s) (patience={patience})")

        if no_improve >= patience:
            if not minimal_logging:
                print(f"Early stopping triggered (no improvement for {patience} epochs). Stopping training.")
            break

        # save metrics JSON and ROC/PR plots per-epoch
        metrics_out_dir = Path(cfg.get('metrics_dir', out_dir / 'metrics'))
        metrics_out_dir.mkdir(parents=True, exist_ok=True)
        epoch_metrics_path = metrics_out_dir / f'metrics_epoch{epoch}.json'
        with open(epoch_metrics_path, 'w') as mf:
            json.dump(metrics, mf, indent=2)

        plots_dir = Path(cfg.get('plots_dir', out_dir / 'plots'))
        plots_dir.mkdir(parents=True, exist_ok=True)
        for t in targets:
            if metrics.get(t) is None:
                continue
            if len(all_preds[t]) == 0:
                continue
            preds = np.concatenate(all_preds[t])
            trues = np.concatenate(all_trues[t])
            try:
                fpr, tpr, _ = roc_curve(trues, preds)
                prec, recall, _ = precision_recall_curve(trues, preds)

                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC AUC {metrics[t].get("roc_auc")}')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title(f'ROC Curve - {t} - epoch {epoch}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plots_dir / f'roc_{t}_epoch{epoch}.png')
                plt.close()

                plt.figure()
                plt.plot(recall, prec, label=f'PR AUC {metrics[t].get("pr_auc")}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'PR Curve - {t} - epoch {epoch}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plots_dir / f'pr_{t}_epoch{epoch}.png')
                plt.close()
            except Exception:
                continue

        # save checkpoint on improved val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / f'mmoe_best.pt'
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'cfg': cfg}, ckpt_path)
            if not minimal_logging:
                print(f"Saved checkpoint to {ckpt_path}")

        # optionally export validation predictions and run Pareto reranker
        if cfg.get('export_predictions', False):
            pred_rows = []
            id_col = cfg.get('id_column')
            user_col = cfg.get('user_column')
            # produce per-row predictions from val dataset
            model.eval()
            with torch.no_grad():
                for i in range(len(val_ds)):
                    x_cats_row, x_nums_row, y_row = val_ds[i]
                    x_cats_b = {k: v.unsqueeze(0).to(device) for k, v in x_cats_row.items()}
                    x_nums_b = x_nums_row.unsqueeze(0).to(device) if x_nums_row is not None else None
                    preds_row = model(x_cats_b, x_nums_b)
                    row = { 'idx': int(val_ds.X.index[i]) }
                    # map probabilities to readable columns
                    row['like_score'] = float(torch.sigmoid(preds_row['like_logit']).cpu().item())
                    row['longview_score'] = float(torch.sigmoid(preds_row['longview_logit']).cpu().item())
                    if cfg.get('creator_task_enabled', True) and preds_row.get('creator_logit') is not None:
                        row['creator_score'] = float(torch.sigmoid(preds_row['creator_logit']).cpu().item())
                    if id_col and id_col in val_ds.X.columns:
                        row[id_col] = val_ds.X.iloc[i][id_col]
                    if user_col and user_col in val_ds.X.columns:
                        row[user_col] = val_ds.X.iloc[i][user_col]
                    pred_rows.append(row)

            preds_df = pd.DataFrame(pred_rows)
            preds_path = Path(cfg.get('predictions_path', out_dir / f'predictions_epoch{epoch}.csv'))
            preds_df.to_csv(preds_path, index=False)
            if not minimal_logging:
                print(f"Saved validation predictions to {preds_path}")

            if cfg.get('run_pareto', False):
                score_cols = cfg.get('pareto_score_cols', ['like_score', 'longview_score'])
                k = cfg.get('pareto_k', 10)
                pareto_df = topk_by_pareto(preds_df, tuple(score_cols), k=k)
                pareto_out = Path(cfg.get('pareto_out', out_dir / f'pareto_epoch{epoch}.csv'))
                pareto_df.to_csv(pareto_out, index=False)
                if not minimal_logging:
                    print(f"Saved pareto top-{k} to {pareto_out}")

if __name__ == '__main__':
    main()

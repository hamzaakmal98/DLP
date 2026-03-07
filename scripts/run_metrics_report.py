import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_curve, roc_curve


def main():
    art = 'kuairand-like-prediction/artifacts/models/mmoe_amd'
    preds_path = os.path.join(art, 'val_predictions.csv')
    y_path = 'kuairand-like-prediction/data/processed/y.csv'
    metrics_out = os.path.join(art, 'report_metrics.json')
    plots_dir = os.path.join(art, 'plots_report')
    os.makedirs(plots_dir, exist_ok=True)

    preds = pd.read_csv(preds_path)
    y = pd.read_csv(y_path)
    y['idx'] = y.index
    merged = preds.merge(y, on='idx', how='left')

    print('Merged columns:', merged.columns.tolist())
    tasks = []
    if 'is_like' in merged.columns:
        tasks.append(('is_like', 'like_score'))
    if 'long_view' in merged.columns:
        tasks.append(('long_view', 'longview_score'))
    if 'creator_interest' in merged.columns:
        tasks.append(('creator_interest', 'creator_score'))

    report = {}
    for label_col, score_col in tasks:
        sub = merged[[label_col, score_col]].dropna()
        if len(sub) == 0:
            report[label_col] = {'error': 'no labels found for this task in merged data'}
            continue
        y_true = sub[label_col].astype(int).values
        y_score = sub[score_col].values
        r = {}
        try:
            r['roc_auc'] = float(roc_auc_score(y_true, y_score))
        except Exception:
            r['roc_auc'] = None
        try:
            r['pr_auc'] = float(average_precision_score(y_true, y_score))
        except Exception:
            r['pr_auc'] = None
        try:
            r['logloss'] = float(log_loss(y_true, y_score, labels=[0, 1]))
        except Exception:
            r['logloss'] = None
        ks = [10, 100]
        for k in ks:
            kk = min(k, len(y_score))
            if kk <= 0:
                r[f'prec_at_{k}'] = None
            else:
                topk_idx = np.argsort(-y_score)[:kk]
                r[f'prec_at_{k}'] = float(y_true[topk_idx].sum() / kk)
        report[label_col] = r

        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            prec, recall, _ = precision_recall_curve(y_true, y_score)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, lw=2)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'ROC - {label_col}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'roc_{label_col}.png'))
            plt.close()

            plt.figure(figsize=(6, 5))
            plt.plot(recall, prec, lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR - {label_col}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'pr_{label_col}.png'))
            plt.close()
        except Exception as e:
            report[label_col]['plot_error'] = str(e)

    with open(metrics_out, 'w') as f:
        json.dump(report, f, indent=2)

    print('Report written to', metrics_out)
    print('Plots written to', plots_dir)
    print('Summary:')
    for t, v in report.items():
        print(t, v)


if __name__ == '__main__':
    main()

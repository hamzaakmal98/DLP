import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.reranker.pareto import topk_by_pareto, pareto_front, scalarize_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True, help='CSV with predictions; columns: item_id,user_id(optional),score columns')
    parser.add_argument('--scores', nargs='+', required=True, help='Score column names in CSV in order')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--out', type=str, default='rerank_output.csv')
    parser.add_argument('--mode', choices=['pareto','scalarize'], default='pareto')
    parser.add_argument('--weights', nargs='+', type=float, help='Weights for scalarization (if mode scalarize)')
    args = parser.parse_args()

    df = pd.read_csv(args.preds)
    if args.mode == 'pareto':
        out = topk_by_pareto(df, tuple(args.scores), k=args.k)
    else:
        pts = df[args.scores].to_numpy()
        w = np.array(args.weights) if args.weights is not None else np.ones(len(args.scores))
        idxs = scalarize_rank(pts, w)
        out = df.iloc[idxs[: args.k]]

    out.to_csv(args.out, index=False)
    print(f'Wrote {len(out)} rows to {args.out}')


if __name__ == '__main__':
    main()

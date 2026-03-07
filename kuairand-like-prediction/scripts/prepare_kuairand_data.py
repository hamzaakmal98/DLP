import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys


def infer_time_col(df, hint=None):
    candidates = []
    if hint:
        candidates.append(hint)
    candidates += ['timestamp', 'ts', 'time', 'created_at', 'created', 'date', 'datetime']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=False,
                   default='real_data/KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv')
    p.add_argument('--time-col', type=str, default=None)
    p.add_argument('--targets', type=str, default='is_like,is_click,long_view,creator_interest')
    p.add_argument('--out', type=str, default='data/processed')
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f'Input file not found: {inp}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(inp)

    time_col = infer_time_col(df, args.time_col)
    if time_col is None:
        print('No time column detected. Please provide --time-col', file=sys.stderr)
        sys.exit(2)

    print(f'Using time column: {time_col}')
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    if df[time_col].isna().all():
        print('All parsed times are NaT. Check --time-col', file=sys.stderr)
        sys.exit(2)

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    for t in targets:
        if t not in df.columns:
            df[t] = 0

    # keep only rows that have at least one positive in targets
    df_pos = df[df[targets].sum(axis=1) > 0].copy()
    if df_pos.empty:
        print('No rows with positive targets found after filtering. Exiting.', file=sys.stderr)
        sys.exit(2)

    # assign week buckets
    df_pos['week_start'] = df_pos[time_col].dt.to_period('W').apply(lambda p: p.start_time)
    weeks = sorted(df_pos['week_start'].dropna().unique())
    if len(weeks) >= 3:
        n = len(weeks)
        n_train = max(1, int(np.ceil(0.6 * n)))
        n_val = max(1, int(np.ceil(0.2 * n)))
        # remainder for test
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train + n_val + n_test > n:
                # adjust n_val
                n_val = max(1, n - n_train - n_test)

        train_weeks = weeks[:n_train]
        val_weeks = weeks[n_train:n_train + n_val]
        test_weeks = weeks[n_train + n_val: n_train + n_val + n_test]

        df_pos['split'] = 'test'
        df_pos.loc[df_pos['week_start'].isin(train_weeks), 'split'] = 'train'
        df_pos.loc[df_pos['week_start'].isin(val_weeks), 'split'] = 'val'
    else:
        # fallback: not enough distinct weeks — use chronological 60/20/20 split by timestamp
        df_pos = df_pos.sort_values(by=time_col).reset_index(drop=True)
        nrows = len(df_pos)
        cut1 = int(0.6 * nrows)
        cut2 = int(0.8 * nrows)
        df_pos['split'] = 'test'
        df_pos.loc[:cut1 - 1, 'split'] = 'train'
        df_pos.loc[cut1:cut2 - 1, 'split'] = 'val'

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # build X (features) and y (targets)
    drop_cols = targets + ['week_start', time_col]
    X = df_pos.drop(columns=[c for c in drop_cols if c in df_pos.columns]).copy()
    y = df_pos[targets].copy()

    # ensure index is unique and stable
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # save files
    X.to_csv(outdir / 'X.csv')
    y.to_csv(outdir / 'y.csv')

    # save split indices
    train_idx = X.index[df_pos['split'] == 'train'].to_numpy()
    val_idx = X.index[df_pos['split'] == 'val'].to_numpy()
    test_idx = X.index[df_pos['split'] == 'test'].to_numpy()
    np.savetxt(outdir / 'train_idx.csv', train_idx, fmt='%d', delimiter=',')
    np.savetxt(outdir / 'val_idx.csv', val_idx, fmt='%d', delimiter=',')
    np.savetxt(outdir / 'test_idx.csv', test_idx, fmt='%d', delimiter=',')

    # summary
    print('Split counts:')
    print(' train:', len(train_idx))
    print(' val:  ', len(val_idx))
    print(' test: ', len(test_idx))
    for t in targets:
        print(f"Target {t}: train_pos={int(y.loc[train_idx][t].sum())} val_pos={int(y.loc[val_idx][t].sum())} test_pos={int(y.loc[test_idx][t].sum())}")


if __name__ == '__main__':
    main()

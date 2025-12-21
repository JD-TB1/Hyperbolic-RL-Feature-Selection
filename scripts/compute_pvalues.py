# scripts/compute_pvalues.py
from __future__ import annotations

import argparse
import numpy as np

from rlfs.data.pvalue import (
    compute_logit_pvalues,
    select_topk_features,
    export_topk_npy,
)
from rlfs.data.splits import train_test_split_fixed


def parse_args():
    p = argparse.ArgumentParser("Compute p-values and export top-k features")
    p.add_argument("--X", type=str, required=True)
    p.add_argument("--y", type=str, required=True)
    p.add_argument("--feature-names", type=str, required=True)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--topk", type=int, nargs="+", required=True)
    p.add_argument("--out-prefix", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    X = np.load(args.X)
    y = np.load(args.y)
    feature_names = list(np.load(args.feature_names, allow_pickle=True))

    X_train, X_test, y_train, y_test = train_test_split_fixed(
        X, y, train_size=args.train_size
    )

    pvals = compute_logit_pvalues(X_train, y_train, feature_names)

    for k in args.topk:
        idx = select_topk_features(pvals, k)
        prefix = f"{args.out_prefix}_k{k}"
        export_topk_npy(
            X_train, X_test, y_train, y_test, idx, prefix=prefix
        )
        print(f"Saved top-{k} features to {prefix}_*.npy")


if __name__ == "__main__":
    main()

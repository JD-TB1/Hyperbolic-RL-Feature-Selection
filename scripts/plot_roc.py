# scripts/plot_roc.py
from __future__ import annotations

import argparse
import numpy as np

from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.evaluation.roc import (
    load_qnetwork_from_checkpoint,
    rollout_selected_features,
    compute_roc_auc,
    plot_roc_curve,
)


def parse_args():
    p = argparse.ArgumentParser("Plot ROC from trained RL feature selector")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-x", type=str, required=True)
    p.add_argument("--train-y", type=str, required=True)
    p.add_argument("--test-x", type=str, required=True)
    p.add_argument("--test-y", type=str, required=True)
    p.add_argument("--episode-max-steps", type=int, default=100)
    p.add_argument("--save-path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    X_train = np.load(args.train_x)
    y_train = np.load(args.train_y)
    X_test = np.load(args.test_x)
    y_test = np.load(args.test_y)

    n_feat = X_train.shape[1]

    env = FeatureSelectionEnv(
        n_feat=n_feat,
        max_steps=args.episode_max_steps,
        reward_fn=lambda _: 0.0,
        lam=0.0,
    )

    q = load_qnetwork_from_checkpoint(
        checkpoint_path=args.checkpoint,
        obs_dim=n_feat + 1,
        n_actions=n_feat + 1,
    )

    selected = rollout_selected_features(q, env)
    print(f"Selected {len(selected)} features")

    fpr, tpr, roc_auc, acc = compute_roc_auc(
        X_train[:, selected],
        y_train,
        X_test[:, selected],
        y_test,
    )

    print(f"Accuracy: {acc:.4f}, AUC: {roc_auc:.4f}")

    plot_roc_curve(
        fpr,
        tpr,
        roc_auc,
        title="ROC Curve (RL Feature Selection)",
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

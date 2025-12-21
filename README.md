# RL Feature Selection Pipeline


- [<span class="toc-section-number">1</span> Overview](#overview)
- [<span class="toc-section-number">2</span> Folder
  structure](#folder-structure)
- [<span class="toc-section-number">3</span> RL component
  design](#rl-component-design)
  - [<span class="toc-section-number">3.1</span>
    Environment](#environment)
  - [<span class="toc-section-number">3.2</span> Reward](#reward)
  - [<span class="toc-section-number">3.3</span> DQN
    trainer](#dqn-trainer)
- [<span class="toc-section-number">4</span> Scripts (public
  interface)](#scripts-public-interface)
  - [<span class="toc-section-number">4.1</span> 1)
    scripts/train_dqn.py](#1-scriptstrain_dqnpy)
    - [<span class="toc-section-number">4.1.1</span> Minimal
      example](#minimal-example)
    - [<span class="toc-section-number">4.1.2</span> Full example
      (recommended)](#full-example-recommended)
    - [<span class="toc-section-number">4.1.3</span>
      Arguments](#arguments)
  - [<span class="toc-section-number">4.2</span> 2)
    scripts/compute_pvalues.py](#2-scriptscompute_pvaluespy)
  - [<span class="toc-section-number">4.3</span> 3)
    scripts/plot_roc.py](#3-scriptsplot_rocpy)
- [<span class="toc-section-number">5</span> Common
  workflow](#common-workflow)
- [<span class="toc-section-number">6</span> Notes](#notes)

# Overview

This repository provides a modular pipeline for
reinforcement-learning-based feature selection. The RL agent learns to
sequentially select a subset of features that maximizes a downstream
classification accuracy reward.

The design goal is to keep all core functionality inside a reusable
Python package (`rlfs/`) and expose only a few clean entry scripts in
`scripts/`.

# Folder structure

``` text
rlfs/
  config/
    schema.py
    io.py
  data/
    io.py
    build.py
    splits.py
    pvalue.py
  env/
    feature_selection.py
  models/
    q_network.py
  replay/
    buffer.py
  rewards/
    accuracy.py
  trainers/
    dqn_trainer.py
  evaluation/
    roc.py
  utils/
    seed.py
    paths.py
    masking.py

scripts/
  train_dqn.py
  compute_pvalues.py
  plot_roc.py

runs/
  <timestamp>_<run-name>/
    fs_config.json
    evaluated_returns.npy
    best_model_ep*_step*.pth
    checkpoint_ep*_step*.pth
```

# RL component design

## Environment

File: `rlfs/env/feature_selection.py`

The environment models feature selection as a sequential decision
process.

State:

- a binary mask for selected features, shape `(n_feat,)`
- a normalized time step scalar `t / max_steps`, shape `(1,)`
- total observation dimension is `obs_dim = n_feat + 1`

Actions: `0 .. n_feat`

- `0 .. n_feat-1`: select feature `i` if not selected yet
- `n_feat`: STOP action

Termination:

- STOP action
- reaching `max_steps` (episode length)
- selecting all features
- illegal action (should not happen if masking is correct)

Reward:

- intermediate reward is `0`

- terminal reward is `10 * acc - lam * |S|`

  - `acc = reward_fn(selected_features)`
  - `lam` is a feature-count penalty

## Reward

File: `rlfs/rewards/accuracy.py`

`AccuracyReward` computes terminal reward as a classification accuracy
on a fixed train/test split.

- results are cached using the sorted feature set as key

- supported models:

  - `svc` (default)
  - `logreg`

Important: the environment multiplies accuracy by `10.0` when forming
terminal reward, consistent with the original implementation.

## DQN trainer

File: `rlfs/trainers/dqn_trainer.py`

The trainer implements a standard DQN loop with:

- epsilon-greedy exploration with linear decay
- replay buffer stored on CPU and sampled to the configured device
- target network updated by Polyak averaging when `target_tau < 1.0`,
  otherwise hard updates
- periodic evaluation and checkpoint saving under `runs/`

Illegal actions are masked by assigning very negative Q-values before
action selection.

# Scripts (public interface)

All user-facing commands live in `scripts/`. These scripts only assemble
components and call into `rlfs/`.

## 1) scripts/train_dqn.py

Trains a DQN agent to select features.

### Minimal example

``` bash
python scripts/train_dqn.py \
  --train-x ./I10_data_20000_instances/Training_X_selected_I10_1000_20000_Patients.npy \
  --train-y ./I10_data_20000_instances/Training_y.npy \
  --run-name I10_RL_SVC
```

### Full example (recommended)

``` bash
python scripts/train_dqn.py \
  --train-x ./I10_data_20000_instances/Training_X_selected_I10_1000_20000_Patients.npy \
  --train-y ./I10_data_20000_instances/Training_y.npy \
  --run-name Standard_SVC_NoF_50_Training_15000_Test_5000_SelectFeatures_1000 \
  --max-total-steps 1000000 \
  --episode-max-steps 50 \
  --seed 42 \
  --reward-model svc \
  --train-size 15000 \
  --eps-end 0.1 \
  --eps-decay 10000 \
  --lr 1e-4 \
  --batch-size 2048 \
  --buffer-size 500000 \
  --target-tau 0.005
```

### Arguments

- `--train-x`: path to NumPy array `(N, p)` used for reward evaluation
- `--train-y`: path to NumPy label array `(N,)`
- `--max-total-steps`: total environment interaction steps, typical
  `1000000`
- `--episode-max-steps`: max feature-selection steps per episode,
  typical `50`
- `--run-name`: descriptive experiment name appended to run directory
- `--seed`: random seed, typical `42`

Reward options:

- `--reward-model`: `svc` or `logreg`
- `--train-size`: split point for reward train/test, typical `15000`

Exploration:

- `--eps-end`: final epsilon, typical `0.05–0.2`
- `--eps-decay`: decay duration in steps, typical `10000–50000`

Optimization:

- `--lr`: learning rate, typical `1e-4`
- `--batch-size`: typical `2048` for expensive rewards
- `--buffer-size`: typical `500000`

Target network:

- `--target-tau`: Polyak factor, typical `0.005`

Outputs are written to `runs/<timestamp>_<run-name>/`.

## 2) scripts/compute_pvalues.py

Computes univariate logistic regression p-values for each feature and
exports top-k feature subsets.

``` bash
python scripts/compute_pvalues.py \
  --X ./X_full.npy \
  --y ./y_full.npy \
  --feature-names ./feature_names.npy \
  --train-size 20000 \
  --topk 200 500 1000 \
  --out-prefix ./I10
```

Arguments:

- `--X`: full feature matrix `(N, p)`
- `--y`: labels `(N,)`
- `--feature-names`: array of length `p`
- `--train-size`: train/test split boundary
- `--topk`: one or more integers
- `--out-prefix`: prefix for exported `.npy` files

## 3) scripts/plot_roc.py

Evaluates a trained RL feature selector by rolling out the greedy policy
and plotting ROC/AUC.

``` bash
python scripts/plot_roc.py \
  --checkpoint runs/<run>/best_model_ep*_step*.pth \
  --train-x ./Training_X.npy --train-y ./Training_y.npy \
  --test-x ./Test_X.npy --test-y ./Test_y.npy \
  --episode-max-steps 100 \
  --save-path roc.png
```

Arguments:

- `--checkpoint`: path to saved DQN checkpoint
- `--train-x`, `--train-y`: training data for evaluation classifier
- `--test-x`, `--test-y`: test data
- `--episode-max-steps`: rollout horizon
- `--save-path`: optional output image path

# Common workflow

1.  Use `compute_pvalues.py` to obtain a reduced feature set.
2.  Train RL on the reduced matrix using `train_dqn.py`.
3.  Evaluate selected features with `plot_roc.py`.

# Notes

- Reward evaluation is computationally expensive; caching is essential.
- Ensure `--train-size` matches how your datasets are constructed.
- Each run is stored in a separate directory and never overwritten.

# rlfs/env/feature_selection.py
from __future__ import annotations

from typing import List, Tuple, Callable, Dict
import numpy as np


class FeatureSelectionEnv:
    """
    Combinatorial RL environment for sequential feature selection.

    State:
        - Binary mask of selected features: shape (n_feat,)
        - Normalized step position: shape (1,)
        - Concatenated => obs_dim = n_feat + 1

    Actions:
        - 0 .. n_feat-1 : select feature i (if not selected)
        - n_feat        : STOP action

    Reward:
        - 0 for intermediate steps
        - Terminal reward:
              reward_fn(selected_features) - lam * |selected_features|

    Episode ends when:
        - STOP is selected
        - max_steps reached
        - all features selected
        - illegal action taken
    """

    def __init__(
        self,
        n_feat: int,
        max_steps: int,
        reward_fn: Callable[[List[int]], float],
        lam: float = 0.0,
    ):
        self.n_feat = n_feat
        self.n_actions = n_feat + 1
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.lam = lam

        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.t = 0
        self.selected: List[int] = []
        self.done = False
        return self._state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("step() called on terminated episode; call reset()")

        info: Dict = {}
        reward = 0.0
        self.t += 1

        if action == self.n_feat:
            # STOP
            self.done = True

        elif 0 <= action < self.n_feat and action not in self.selected:
            self.selected.append(action)

        else:
            # Illegal action
            self.done = True
            info["illegal_action"] = True

        # termination conditions
        if self.t >= self.max_steps:
            self.done = True

        if len(self.selected) == self.n_feat:
            self.done = True

        if self.done:
            acc = self.reward_fn(self.selected)
            reward = 10.0 * acc - self.lam * len(self.selected)
            info["acc_val"] = acc
            info["num_features"] = len(self.selected)

        return self._state(), float(reward), bool(self.done), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def legal_actions_mask(self) -> np.ndarray:
        """
        Returns:
            mask: boolean array of shape (n_actions,)
                  True indicates the action is legal.
        """
        mask = np.ones(self.n_actions, dtype=bool)

        if self.selected:
            mask[self.selected] = False

        # STOP is always legal
        return mask

    def _state(self) -> np.ndarray:
        mask = np.zeros(self.n_feat, dtype=np.float32)

        if self.selected:
            mask[np.array(self.selected, dtype=int)] = 1.0

        pos = np.array(
            [self.t / max(1, self.max_steps)],
            dtype=np.float32,
        )

        return np.concatenate([mask, pos], axis=0)

import random
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

@dataclass
class FSConfig:
    n_feat: int
    max_steps: int
    lam: float = 0.0 # feature-count penalty in terminal reward
    gamma: float = 1.0 # episodic tasks typically use gamma=1.0
    eps_start: float = 0.2
    eps_end: float = 0.05
    eps_decay: int = 2000 # steps for epsilon exponential decay
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    target_tau: float = 1.0 # hard update when 1.0; else soft update (Polyak)
    target_update_every: int = 1000 # steps between hard updates if tau==1
    train_start: int = 1000 # fill buffer before training
    train_freq: int = 1
    hidden: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
# ==========================================
class FeatureSelectionEnv:
    """Combinatorial RL environment for sequential feature selection.


    - State: binary mask of selected features (shape [n_feat]) concatenated with
    a scalar indicating normalized step t/max_steps (shape [1]).
    - Actions: 0..n_feat-1 pick feature i if unselected, n_feat indicates STOP.
    - Reward: 0 per step; terminal reward = reward_fn(selected) - lam*|S|.
    - Episode ends upon STOP, reaching max_steps, or no legal actions.
    """
    def __init__(self, n_feat: int, max_steps: int, reward_fn: Callable[[List[int]], float], lam: float = 0.0):
        self.n_feat = n_feat
        self.n_actions = n_feat + 1
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.lam = lam
        self.reset()
        
    def reset(self):
        self.t = 0
        self.selected: List[int] = []
        self.done = False
        return self._state()
    
    def legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        # legal feature picks: those not yet selected
        mask[:, :]= True
        if self.selected:
            mask[self.selected] = False
        # STOP is always allowed
        return mask
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self.done, "Call reset() before stepping a finished episode"
        info = {}
        reward = 0.0
        self.t += 1


        if action == self.n_feat:
        # STOP
            self.done = True
        elif action < self.n_feat and action not in self.selected:
            self.selected.append(action)
        else:
        # Illegal action (should be masked by agent). Penalize lightly and continue, or end.
        # Here we end the episode to simplify credit assignment.
            self.done = True
            info["illegal_action"] = True


        # termination conditions
        if self.t >= self.max_steps:
            self.done = True
        if len(self.selected) == self.n_feat:
            self.done = True


        if self.done:
        # Sparse terminal reward
            acc = self.reward_fn(self.selected)
            reward = acc - self.lam * len(self.selected)
            info["acc_val"] = acc
            info["num_features"] = len(self.selected)


        return self._state(), float(reward), bool(self.done), info
    
    
    def _state(self):
        mask = np.zeros(self.n_feat, dtype=np.float32)
        if self.selected:
            mask[np.array(self.selected, dtype=int)] = 1.0
        pos = np.array([self.t / max(1, self.max_steps)], dtype=np.float32)
        return np.concatenate([mask, pos], axis=0)
    

# ==========================================

class ReplayBuffer:
    def __init__(self, obs_dim: int, n_actions: int, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False


        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_legal_masks = torch.zeros((capacity, n_actions), dtype=torch.bool)


    def add(self, s, a, r, s2, done, next_legal_mask):
        i = self.ptr
        self.obs[i] = torch.as_tensor(s)
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_obs[i] = torch.as_tensor(s2)
        self.dones[i] = float(done)
        self.next_legal_masks[i] = torch.as_tensor(next_legal_mask)


        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0


    def sample(self, batch: int):
        n = self.capacity if self.full else self.ptr
        idx = torch.randint(0, n, (batch,), device=self.device)
        return (
        self.obs[idx].to(self.device),
        self.actions[idx].to(self.device),
        self.rewards[idx].to(self.device),
        self.next_obs[idx].to(self.device),
        self.dones[idx].to(self.device),
        self.next_legal_masks[idx].to(self.device),
        )
        
# ==========================================

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_actions)
        )


    def forward(self, x):
        return self.net(x)




def dqn_train(
env: FeatureSelectionEnv,
cfg: FSConfig,
episodes: int = 200,
max_total_steps: Optional[int] = None,
seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    obs_dim = env.n_feat + 1
    n_actions = env.n_actions


    device = torch.device(cfg.device)


    q = QNetwork(obs_dim, n_actions, hidden=cfg.hidden).to(device)
    q_tgt = QNetwork(obs_dim, n_actions, hidden=cfg.hidden).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = Adam(q.parameters(), lr=cfg.lr)


    buf = ReplayBuffer(obs_dim, n_actions, cfg.buffer_size, device=device)


    # epsilon schedule
    def epsilon_by_step(step):
        frac = math.exp(-step / max(1, cfg.eps_decay))
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * frac


    global_step = 0
    returns = []


    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_return = 0.0


        while not done:
            eps = epsilon_by_step(global_step)
            legal_mask = env.legal_actions_mask() # bool [n_actions]


            if random.random() < eps:
                breakpoint()
                legal_idxs = np.flatnonzero(legal_mask)
                a = int(np.random.choice(legal_idxs))
            else:
                with torch.no_grad():
                    qs = q(torch.as_tensor(s, device=device).unsqueeze(0))[0]
                    # Mask illegal actions by setting them to -inf
                    q_values = qs.clone()
                    illegal = torch.as_tensor(~legal_mask, device=device)
                    q_values[illegal] = -1e9
                    a = int(torch.argmax(q_values).item())


            s2, r, done, info = env.step(a)
            next_legal = env.legal_actions_mask()


            buf.add(s, a, r, s2, done, next_legal)
            s = s2
            ep_return += r
            global_step += 1


            # Learn
            if global_step > cfg.train_start and global_step % cfg.train_freq == 0:
                batch = cfg.batch_size
                (b_s, b_a, b_r, b_s2, b_done, b_next_legal) = buf.sample(batch)


                # Q(s,a)
                q_sa = q(b_s).gather(1, b_a)


                with torch.no_grad():
                    q_next_all = q_tgt(b_s2)
                    # mask illegal next actions
                    neg_inf = torch.full_like(q_next_all, -1e9)
                    q_next_masked = torch.where(b_next_legal, q_next_all, neg_inf)
                    q_next_max = q_next_masked.max(dim=1, keepdim=True).values
                    target = b_r + (1.0 - b_done) * cfg.gamma * q_next_max


                loss = F.smooth_l1_loss(q_sa, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()


                # target updates
                if cfg.target_tau >= 1.0:
                    if global_step % cfg.target_update_every == 0:
                        q_tgt.load_state_dict(q.state_dict())
                else:
                    with torch.no_grad():
                        for p, p_tgt in zip(q.parameters(), q_tgt.parameters()):
                            p_tgt.mul_(1.0 - cfg.target_tau).add_(cfg.target_tau * p)


        if max_total_steps and global_step >= max_total_steps:
            break

        returns.append(ep_return)
        if (ep % 10) == 0:
            print(f"Episode {ep} | return={ep_return:.4f} | selected={len(env.selected)} | acc_val={info.get('acc_val')}")
        if max_total_steps and global_step >= max_total_steps:
            break


    return {
    "q": q,
    "q_target": q_tgt,
    "returns": returns,
    }
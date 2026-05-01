"""Trajectory generation and Dataset for the particle environment."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_trajectories(
    env_fn: Callable,
    n_episodes: int = 1000,
    length: int = 50,
    action_dim: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out random-action trajectories.

    Returns
    -------
    obs : (E, T, H, W, C) float32 in [0, 1]
    actions : (E, T, A) float32 — action a_t leads from obs[t] to obs[t+1];
              the last action is replicated to keep arrays rectangular.
    states : (E, T, 2) float32 — ground-truth (x, y), useful for probing later.
    """
    rng = np.random.default_rng(seed)
    obs_traj, act_traj, state_traj = [], [], []
    for _ in range(n_episodes):
        env = env_fn(seed=int(rng.integers(0, 2**31 - 1)))
        obs = [env.reset()]
        states = [env.state()]
        actions = []
        for _ in range(length - 1):
            a = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            o = env.step(a)
            obs.append(o)
            actions.append(a)
            states.append(env.state())
        # pad action so len matches obs
        actions.append(actions[-1].copy())
        obs_traj.append(np.stack(obs))
        act_traj.append(np.stack(actions))
        state_traj.append(np.stack(states))
    return (
        np.stack(obs_traj).astype(np.float32),
        np.stack(act_traj).astype(np.float32),
        np.stack(state_traj).astype(np.float32),
    )


class TrajectoryDataset(Dataset):
    """Sub-trajectory sampler.

    Each item is a contiguous window of `sub_len` frames + matching actions,
    drawn from the cached trajectory tensor. Uses fixed enumeration of windows
    (no random shifts) so DataLoader shuffling is what introduces stochasticity.
    """

    def __init__(self, obs: np.ndarray, actions: np.ndarray, sub_len: int = 4):
        # obs: (E, T, H, W, C)  actions: (E, T, A)
        assert obs.ndim == 5 and actions.ndim == 3
        self.obs = obs
        self.actions = actions
        self.E, self.T = obs.shape[:2]
        self.sub_len = sub_len
        self.indices = [
            (e, s) for e in range(self.E) for s in range(self.T - sub_len + 1)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        e, s = self.indices[idx]
        o = self.obs[e, s : s + self.sub_len]  # (L, H, W, C)
        a = self.actions[e, s : s + self.sub_len]  # (L, A)
        # HWC -> CHW
        o = np.transpose(o, (0, 3, 1, 2))  # (L, C, H, W)
        return torch.from_numpy(np.ascontiguousarray(o)), torch.from_numpy(np.ascontiguousarray(a))

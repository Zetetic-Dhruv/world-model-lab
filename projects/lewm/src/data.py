"""Trajectory generation and Datasets for the LeWM reproduction.

Two datasets:
  * TrajectoryDataset — wraps in-memory numpy arrays (synthetic envs).
  * HDF5TrajectoryDataset — loads on-disk HDF5 files for real-data workflows.

Two trajectory generators:
  * generate_trajectories — random-action trajectories (good for ParticleEnv).
  * generate_pusht_trajectories — biased policy trajectories for MiniPushT
    (random-walk with periodic bias toward agent-near-block to ensure contact
    events appear in the dataset).
"""

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


def generate_weak_policy_trajectories(
    env_fn: Callable,
    n_episodes: int = 500,
    length: int = 100,
    action_dim: int = 2,
    dist_constraint: float = 6.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Path C canonical trajectory generator — WeakPolicy contact-rich exploration.

    Mirrors stable-worldmodel's WeakPolicy. Agent's commanded position is
    constrained to within `dist_constraint` of the block. Contact rate ~85%.
    NOT a goal-reaching policy.

    Returns (obs, actions, states) with shapes
        obs:     (E, T, H, W, C) float32
        actions: (E, T, action_dim) float32
        states:  (E, T, state_dim) float32
    """
    from .env_pusht import weak_policy
    rng = np.random.default_rng(seed)
    obs_traj, act_traj, state_traj = [], [], []
    for _ in range(n_episodes):
        env = env_fn(seed=int(rng.integers(0, 2**31 - 1)))
        obs = [env.reset()]
        states = [env.state()]
        actions = []
        for _ in range(length - 1):
            a = weak_policy(env.state(), rng, dist_constraint=dist_constraint)
            o = env.step(a)
            obs.append(o)
            actions.append(a)
            states.append(env.state())
        actions.append(actions[-1].copy())  # rectangular pad
        obs_traj.append(np.stack(obs))
        act_traj.append(np.stack(actions))
        state_traj.append(np.stack(states))
    return (
        np.stack(obs_traj).astype(np.float32),
        np.stack(act_traj).astype(np.float32),
        np.stack(state_traj).astype(np.float32),
    )


def generate_weak_policy_reacher_trajectories(
    env_fn: Callable,
    n_episodes: int = 500,
    length: int = 60,
    action_dim: int = 2,
    bias_toward_target: float = 0.4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Path C canonical trajectory generator for dm_control Reacher.

    Mirrors WeakPolicy semantics: contact-rich exploration with a moderate
    bias toward the target so trajectories include actual reach events
    (analogue of MiniPushT's "block-touching" rate). The bias is NOT a
    goal-reaching policy — it just ensures the dataset isn't pure random
    torques (which essentially never reach 0.015-radius targets).

    Returns (obs, actions, states) with shapes
        obs:     (E, T, H, W, C) float32 in [0,1]
        actions: (E, T, action_dim) float32
        states:  (E, T, state_dim) float32   state_dim=6
    """
    from .env_reacher import weak_policy_reacher
    rng = np.random.default_rng(seed)
    obs_traj, act_traj, state_traj = [], [], []
    for _ in range(n_episodes):
        env = env_fn(seed=int(rng.integers(0, 2**31 - 1)))
        obs = [env.reset()]
        states = [env.state()]
        actions = []
        for _ in range(length - 1):
            a = weak_policy_reacher(
                env.state(), rng, bias_toward_target=bias_toward_target,
            )
            o = env.step(a)
            obs.append(o)
            actions.append(a)
            states.append(env.state())
        actions.append(actions[-1].copy())  # rectangular pad
        obs_traj.append(np.stack(obs))
        act_traj.append(np.stack(actions))
        state_traj.append(np.stack(states))
    return (
        np.stack(obs_traj).astype(np.float32),
        np.stack(act_traj).astype(np.float32),
        np.stack(state_traj).astype(np.float32),
    )


def generate_pusht_trajectories(
    env_fn: Callable,
    n_episodes: int = 1000,
    length: int = 50,
    action_dim: int = 2,
    bias_strength: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[Diagnostic only] Biased-random trajectory generator. NOT canonical.

    Use generate_weak_policy_trajectories for Path C training data.
    """
    rng = np.random.default_rng(seed)
    obs_traj, act_traj, state_traj = [], [], []
    for _ in range(n_episodes):
        env = env_fn(seed=int(rng.integers(0, 2**31 - 1)))
        obs = [env.reset()]
        states = [env.state()]
        actions = []
        for _ in range(length - 1):
            s = env.state()
            agent_xy = s[0:2]
            block_xy = s[2:4]
            # Bias direction: from agent toward block
            d = block_xy - agent_xy
            n = np.linalg.norm(d) + 1e-6
            bias = (d / n).astype(np.float32)
            random_noise = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            a = bias_strength * bias + (1.0 - bias_strength) * random_noise
            a = np.clip(a, -1.0, 1.0)
            o = env.step(a)
            obs.append(o)
            actions.append(a)
            states.append(env.state())
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


class HDF5TrajectoryDataset(Dataset):
    """Trajectory dataset that reads from one or more HDF5 files.

    Expected HDF5 layout (one or both per file):
        /obs       (E, T, H, W, C) float32 in [0, 1]   OR   uint8 [0, 255]
        /actions   (E, T, A)        float32

    For real-data workflows: point this at a directory of .h5 files (one file
    per recording session, or one per episode block). The dataset fans out to
    cover all (file, episode, window) combinations.

    The interface matches TrajectoryDataset so the trainer doesn't need to
    distinguish: items are (obs_window, action_window) tensors of shape
    (sub_len, C, H, W) and (sub_len, A).
    """

    def __init__(self, paths: list[str] | str, sub_len: int = 4):
        try:
            import h5py  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HDF5TrajectoryDataset requires h5py. Install with: pip install h5py"
            ) from e
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        self.sub_len = sub_len
        self._index: list[tuple[int, int, int]] = []  # (file_idx, episode_idx, start_t)
        self._handles: list[None | "h5py.File"] = [None] * len(self.paths)
        self._meta: list[tuple[int, int]] = []        # (E, T) per file
        self._build_index()

    def _open(self, file_idx: int):
        import h5py
        if self._handles[file_idx] is None:
            self._handles[file_idx] = h5py.File(self.paths[file_idx], "r")
        return self._handles[file_idx]

    def _build_index(self):
        import h5py
        for fi, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                E, T = f["obs"].shape[0], f["obs"].shape[1]
                if "actions" in f:
                    aE, aT = f["actions"].shape[0], f["actions"].shape[1]
                    assert (aE, aT) == (E, T), f"{p}: obs/actions shape mismatch"
                self._meta.append((E, T))
            for e in range(E):
                for s in range(T - self.sub_len + 1):
                    self._index.append((fi, e, s))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, e, s = self._index[idx]
        f = self._open(file_idx)
        obs_chunk = np.asarray(f["obs"][e, s : s + self.sub_len])  # (L, H, W, C)
        if obs_chunk.dtype == np.uint8:
            obs_chunk = obs_chunk.astype(np.float32) / 255.0
        else:
            obs_chunk = obs_chunk.astype(np.float32)
        act_chunk = np.asarray(f["actions"][e, s : s + self.sub_len]).astype(np.float32)
        # HWC -> CHW
        obs_chunk = np.transpose(obs_chunk, (0, 3, 1, 2))
        return (
            torch.from_numpy(np.ascontiguousarray(obs_chunk)),
            torch.from_numpy(np.ascontiguousarray(act_chunk)),
        )

    def close(self):
        for h in self._handles:
            if h is not None:
                h.close()
        self._handles = [None] * len(self.paths)


def write_trajectories_h5(path: str, obs: np.ndarray, actions: np.ndarray,
                          states: np.ndarray | None = None,
                          compression: str = "gzip"):
    """Save trajectories to HDF5 in the format HDF5TrajectoryDataset expects."""
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("obs", data=obs, compression=compression)
        f.create_dataset("actions", data=actions, compression=compression)
        if states is not None:
            f.create_dataset("states", data=states, compression=compression)


# ===========================================================================
# Episode-directory format — canonical external data format
# ===========================================================================
#
# Layout:
#   data_dir/
#   ├── manifest.json
#   ├── ep_00000.npz         (mode A)   -- contains obs, actions, optional states
#   └── ep_00000/            (mode B)   -- video + side-channel labels
#       ├── video.mp4        (T frames, HxWxC)
#       ├── actions.npy      (T, action_dim) float32
#       └── states.npy       (T, state_dim)  float32   [optional]
#
# manifest.json schema:
#   {
#     "version":          "1.0",
#     "format":           "npz" | "video",
#     "action_dim":       int,
#     "image_size":       [H, W],
#     "channels":         3,
#     "n_episodes":       int,
#     "episode_length":   int  (or null if variable per-episode),
#     "fps":              int  (info only),
#     "obs_dtype":        "uint8" | "float32",
#     "obs_range":        "[0, 255]" | "[0, 1]"
#   }
# ===========================================================================


class EpisodeDirectoryDataset(Dataset):
    """Read episodes from a directory matching the canonical episode-directory format.

    Auto-detects mode (npz vs video) from manifest.json. Yields (obs, actions)
    sub-windows of length sub_len, same interface as TrajectoryDataset.

    Real-data on-ramp: drop a folder of episodes here, point train.py at it.
    """

    def __init__(self, data_dir: str | "Path", sub_len: int = 4):
        import json
        from pathlib import Path
        self.root = Path(data_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {data_dir}. See README §Data format."
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.sub_len = sub_len
        self.format = self.manifest["format"]
        self.action_dim = int(self.manifest["action_dim"])
        self.obs_dtype = self.manifest.get("obs_dtype", "float32")
        self.obs_range_norm = self.manifest.get("obs_range", "[0, 1]") == "[0, 1]"

        # Enumerate episode files / directories
        self.episodes: list = []
        if self.format == "npz":
            for p in sorted(self.root.glob("ep_*.npz")):
                self.episodes.append(p)
        elif self.format == "video":
            try:
                import imageio.v3  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "format='video' requires imageio. Install with: pip install imageio imageio-ffmpeg"
                ) from e
            for p in sorted(self.root.glob("ep_*")):
                if p.is_dir() and (p / "video.mp4").exists():
                    self.episodes.append(p)
        else:
            raise ValueError(f"Unknown format '{self.format}' in manifest.json")

        if not self.episodes:
            raise RuntimeError(f"No episodes found under {data_dir} for format={self.format}")

        # Build (episode_path, start_t) index by enumerating each episode's length.
        self._index: list[tuple[int, int]] = []
        self._episode_lengths: list[int] = []
        for i, p in enumerate(self.episodes):
            T = self._episode_length(p)
            self._episode_lengths.append(T)
            for s in range(T - sub_len + 1):
                self._index.append((i, s))

    # ------------------------- mode dispatch -------------------------

    def _episode_length(self, p) -> int:
        if self.format == "npz":
            with np.load(p) as f:
                return int(f["obs"].shape[0])
        else:
            actions_path = p / "actions.npy"
            return int(np.load(actions_path).shape[0])

    def _read_window(self, p, start: int) -> tuple[np.ndarray, np.ndarray]:
        end = start + self.sub_len
        if self.format == "npz":
            with np.load(p) as f:
                obs = np.asarray(f["obs"][start:end])
                actions = np.asarray(f["actions"][start:end])
        else:
            import imageio.v3 as iio
            # FFMPEG plugin is what we write with; use the same for read so we
            # don't need pyav as an extra dependency.
            video = iio.imread(p / "video.mp4", plugin="FFMPEG")
            obs = np.asarray(video[start:end])
            actions = np.load(p / "actions.npy")[start:end]
        return obs, actions

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, start = self._index[idx]
        obs, actions = self._read_window(self.episodes[ep_idx], start)
        # Normalize obs dtype
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
        elif not self.obs_range_norm:
            obs = obs.astype(np.float32) / 255.0
        else:
            obs = obs.astype(np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        # HWC -> CHW
        obs = np.transpose(obs, (0, 3, 1, 2))
        return (
            torch.from_numpy(np.ascontiguousarray(obs)),
            torch.from_numpy(np.ascontiguousarray(actions)),
        )


def write_episode_directory(
    out_dir: str | "Path",
    obs: np.ndarray,         # (E, T, H, W, C)
    actions: np.ndarray,     # (E, T, A)
    states: np.ndarray | None = None,
    *,
    format: str = "npz",     # "npz" or "video"
    fps: int = 10,
    obs_dtype: str = "uint8",  # how to store the pixels
):
    """Convert in-memory trajectories to the canonical episode-directory format.

    Useful for: caching synthetic data in a real-data-compatible format, OR
    converting external recordings into this format.
    """
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    E, T, H, W, C = obs.shape
    A = actions.shape[-1]

    # Convert obs to requested dtype
    obs_to_save = obs
    if obs_dtype == "uint8" and obs.dtype != np.uint8:
        obs_to_save = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        obs_range = "[0, 255]"
    elif obs_dtype == "float32":
        obs_to_save = obs.astype(np.float32)
        obs_range = "[0, 1]"
    else:
        obs_range = "[0, 255]" if obs.dtype == np.uint8 else "[0, 1]"

    manifest = {
        "version": "1.0",
        "format": format,
        "action_dim": int(A),
        "image_size": [int(H), int(W)],
        "channels": int(C),
        "n_episodes": int(E),
        "episode_length": int(T),
        "fps": int(fps),
        "obs_dtype": obs_dtype,
        "obs_range": obs_range,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if format == "npz":
        for e in range(E):
            payload = {"obs": obs_to_save[e], "actions": actions[e].astype(np.float32)}
            if states is not None:
                payload["states"] = states[e].astype(np.float32)
            np.savez_compressed(out_dir / f"ep_{e:05d}.npz", **payload)
    elif format == "video":
        import imageio.v3 as iio
        for e in range(E):
            ep_dir = out_dir / f"ep_{e:05d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            # frames must be uint8 for video codecs
            frames = obs_to_save[e]
            if frames.dtype != np.uint8:
                frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
            iio.imwrite(ep_dir / "video.mp4", list(frames), fps=fps,
                        codec="libx264", plugin="FFMPEG", macro_block_size=8)
            np.save(ep_dir / "actions.npy", actions[e].astype(np.float32))
            if states is not None:
                np.save(ep_dir / "states.npy", states[e].astype(np.float32))
    else:
        raise ValueError(f"format must be 'npz' or 'video', got {format!r}")
    return out_dir


# ===========================================================================
# Path C canonical dataset infrastructure
# ===========================================================================
#
# HDF5 schema (matches stable-worldmodel pusht_expert_train.h5):
#
#   /obs           (E, T, H, W, C)  uint8 [0,255]   — RGB frames per env step
#   /actions       (E, T, A)        float32          — per-env-step actions
#   /states        (E, T, S)        float32          — env state for diagnostics
#   /episode_ends  (E,)             int64            — cumulative end indices
#                                                      (sw-style; T*ep for fixed)
#
# Frameskip semantics live ABOVE this layer in PathCStrideDataset, NOT in the
# HDF5 (which records per-env-step). One source-of-truth for env steps.
# ===========================================================================


def write_canonical_h5(
    out_path: str | "Path",
    obs: np.ndarray,         # (E, T, H, W, C) float32 [0,1] OR uint8 [0,255]
    actions: np.ndarray,     # (E, T, A) float32 — per-env-step
    states: np.ndarray,      # (E, T, S) float32
    *,
    obs_as_uint8: bool = True,
    metadata: dict | None = None,
):
    """Save trajectories in the canonical HDF5 schema."""
    import h5py
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs_to_save = obs
    if obs_as_uint8 and obs.dtype != np.uint8:
        obs_to_save = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
    elif not obs_as_uint8 and obs.dtype != np.float32:
        obs_to_save = obs.astype(np.float32) / 255.0

    E, T = obs.shape[:2]
    episode_ends = np.cumsum(np.full(E, T, dtype=np.int64))

    with h5py.File(out_path, "w") as f:
        f.create_dataset("obs", data=obs_to_save, compression="gzip")
        f.create_dataset("actions", data=actions.astype(np.float32),
                         compression="gzip")
        f.create_dataset("states", data=states.astype(np.float32),
                         compression="gzip")
        f.create_dataset("episode_ends", data=episode_ends)
        meta = {
            "schema_version": "1.0",
            "obs_dtype": str(obs_to_save.dtype),
            "obs_range": "[0, 255]" if obs_to_save.dtype == np.uint8 else "[0, 1]",
            "n_episodes": int(E),
            "episode_length": int(T),
            "action_dim": int(actions.shape[-1]),
            "state_dim": int(states.shape[-1]),
            "image_shape": list(obs.shape[2:]),
        }
        if metadata is not None:
            meta.update(metadata)
        for k, v in meta.items():
            f.attrs[k] = v
    return out_path


def split_episodes_by_trajectory(
    n_episodes: int,
    splits: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Episode-level (NOT frame-level) train/val/test split. Prevents
    leakage between splits; each trajectory is wholly in one split.
    """
    splits = splits or {"train": 0.8, "val": 0.1, "test": 0.1}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    out = {}
    cur = 0
    keys = list(splits.keys())
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            out[k] = perm[cur:]
        else:
            sz = int(round(splits[k] * n_episodes))
            out[k] = perm[cur:cur + sz]
            cur += sz
    return out


class PathCStrideDataset(Dataset):
    """Path C training dataset: stride=5 transitions, action_token_dim=10.

    Each item is a window of `history_size` "stride steps". Each stride step:
      - obs at env step t  (uint8 or float32)
      - obs at env step t+5 (the next-step target)
      - action_token = concat(action[t], action[t+1], ..., action[t+4])

    A window of size `history_size=3` covers 3 stride-steps × 5 = 15 env steps.

    Args:
      h5_path: canonical HDF5 file
      episode_indices: list of episode indices to include (for split semantics)
      history_size: window length in stride-steps (default 3, matches canonical)
      stride: env-steps per stride-step (default 5)
    """

    def __init__(
        self,
        h5_path: str | "Path",
        episode_indices: np.ndarray | list,
        history_size: int = 3,
        stride: int = 5,
    ):
        import h5py
        self.h5_path = str(h5_path)
        self.episode_indices = np.asarray(episode_indices)
        self.history_size = history_size
        self.stride = stride
        self._h5: "h5py.File" | None = None

        # Index: list of (episode_idx, start_env_step) pairs.
        # A window starts at env step s and spans (history_size + 1) * stride
        # env steps total — we need (history_size + 1) latents to compute
        # MSE(predict[:, :-1], emb[:, 1:]) over a history_size-token window.
        with h5py.File(self.h5_path, "r") as f:
            T = int(f["obs"].shape[1])
        max_start = T - (history_size + 1) * stride
        if max_start < 0:
            raise ValueError(
                f"Episode length {T} too short for history_size={history_size} "
                f"× stride={stride}; need ≥ {(history_size + 1) * stride} steps"
            )
        self._index = []
        for e in self.episode_indices:
            for s in range(0, max_start + 1):
                self._index.append((int(e), int(s)))

    def _open(self):
        import h5py
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        e, s = self._index[idx]
        f = self._open()
        # window of (history_size + 1) obs frames at strided positions
        H1 = self.history_size + 1
        # pick obs at s, s+stride, s+2*stride, ..., s+history_size*stride
        obs_indices = np.arange(H1) * self.stride + s
        obs_window = np.asarray(f["obs"][e, obs_indices])  # (H1, H, W, C) uint8
        if obs_window.dtype == np.uint8:
            obs_window = obs_window.astype(np.float32) / 255.0
        # action tokens: for each stride-step h ∈ [0, history_size),
        # token h = concat(action[s + h*stride : s + (h+1)*stride])
        action_tokens = []
        for h in range(self.history_size):
            base = s + h * self.stride
            chunk = np.asarray(f["actions"][e, base:base + self.stride])  # (stride, A)
            action_tokens.append(chunk.reshape(-1))  # (stride * A,)
        action_tokens = np.stack(action_tokens).astype(np.float32)  # (history_size, 10)
        # HWC -> CHW
        obs_window = np.transpose(obs_window, (0, 3, 1, 2)).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(obs_window)),  # (H1, C, H, W)
            torch.from_numpy(np.ascontiguousarray(action_tokens)),  # (history_size, 10)
        )

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def sample_offset_pair(
    h5_path: str | "Path",
    episode_indices: np.ndarray,
    rng: np.random.Generator,
    *,
    offset_steps: int = 25,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Path C eval pair sampler.

    Returns (episode_idx, init_step, goal_step, init_obs, goal_obs,
             init_state, goal_state).

    The goal is the recorded state at init_step + offset_steps within the SAME
    trajectory. Reachability is guaranteed by construction.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        T = int(f["obs"].shape[1])
        if T <= offset_steps:
            raise ValueError(f"Episode length {T} ≤ offset_steps {offset_steps}")
        e = int(rng.choice(episode_indices))
        max_start = T - 1 - offset_steps
        s = int(rng.integers(0, max_start + 1))
        init_obs = np.asarray(f["obs"][e, s])
        goal_obs = np.asarray(f["obs"][e, s + offset_steps])
        init_state = np.asarray(f["states"][e, s]).astype(np.float32)
        goal_state = np.asarray(f["states"][e, s + offset_steps]).astype(np.float32)
        if init_obs.dtype == np.uint8:
            init_obs = init_obs.astype(np.float32) / 255.0
            goal_obs = goal_obs.astype(np.float32) / 255.0
    return e, s, s + offset_steps, init_obs, goal_obs, init_state, goal_state

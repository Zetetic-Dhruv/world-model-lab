"""TwoRoom env wrapper — direct import from canonical stable_worldmodel.

The TwoRoom environment and its ExpertPolicy are the OFFICIAL implementations
from `stable_worldmodel` (Maes, Le Lidec, Balestriero — MIT-licensed,
arXiv:2603.19312, 2026). We do NOT reimplement them; we import directly.

This module only adds:
  1. API translation (gymnasium 5-tuple <-> our reset/step/observe/state)
  2. Optional render-resize to enable our resolution-sweep experiments
     (canonical renders at 224×224; we downsample post-render so the
      env's geometry, dynamics, and policy are bit-exact canonical)
  3. Uniform `state()` / `set_state()` interface matching env_pusht.py and
     env_reacher.py for reuse of train.py / eval_planning.py infrastructure

Source:
  https://github.com/galilai-group/stable-worldmodel/tree/main/stable_worldmodel/envs/two_room
  pip package: stable-worldmodel >= 0.0.6
"""

from __future__ import annotations

import numpy as np

# Lazy imports — stable_worldmodel is a heavy dependency
_gym = None
_swm_imported = False


def _ensure_imports():
    global _gym, _swm_imported
    if _gym is None:
        import gymnasium as gym
        _gym = gym
    if not _swm_imported:
        import stable_worldmodel  # noqa: F401  registers swm/* envs
        _swm_imported = True


class DMTwoRoomEnv:
    """Wrap canonical `swm/TwoRoom-v1` to match our env API.

    State exposed as 10-d: [agent_xy(2), target_xy(2), door_centers(3*2)].
    This matches the canonical `observation_space` exactly.

    Image is rendered at canonical 224×224, then optionally resized via
    bilinear interpolation to `image_size` for resolution-sweep experiments.
    """

    state_dim = 10
    action_dim = 2
    canonical_image_size = 224

    def __init__(self, seed: int | None = None, image_size: int = 64):
        _ensure_imports()
        self._env = _gym.make('swm/TwoRoom-v1')
        if seed is not None:
            self._obs, self._info = self._env.reset(seed=seed)
        else:
            self._obs, self._info = self._env.reset()
        self.image_size = int(image_size)

    # ----------------------- gym-like API -----------------------

    def reset(self) -> np.ndarray:
        self._obs, self._info = self._env.reset()
        return self.observe()

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(2)
        self._obs, _, _, _, self._info = self._env.step(a)
        return self.observe()

    def observe(self) -> np.ndarray:
        img = self._env.render()  # (224, 224, 3) uint8
        if self.image_size != self.canonical_image_size:
            from PIL import Image
            img = np.asarray(
                Image.fromarray(img).resize(
                    (self.image_size, self.image_size), Image.BILINEAR,
                )
            )
        return img.astype(np.float32) / 255.0

    def state(self) -> np.ndarray:
        """Return the canonical (10,) observation vector — matches their
        observation_space exactly: agent(2), target(2), door_centers(3*2)."""
        import torch
        s = self._obs
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        return np.asarray(s, dtype=np.float32)

    def to_target_distance(self) -> float:
        return float(self._info.get('distance_to_target', float('inf')))

    def is_terminated(self, threshold: float = 16.0) -> bool:
        """Canonical termination: dist(agent, target) < 16 px (env step()
        uses the same value at line 273 of env.py)."""
        return self.to_target_distance() < threshold

    is_success = is_terminated

    # ---------------------- canonical-eval seeding ----------------------

    def set_state(self, state: np.ndarray):
        """Set env to a recorded 10-d state.

        State layout: [agent(2), target(2), door_centers(3*2)] — same as state().

        The canonical env exposes private setters:
          _set_state(agent_xy)         — moves agent only
          _set_goal_state(target_xy)   — moves target + re-renders target image

        Door positions are reset-time randomized; they cannot be overridden
        post-reset without surgery on the variation_space. For the canonical
        recorded-future-state eval, we set agent + target only and accept
        that doors may differ from the recorded trajectory's episode. The
        latent τ-match metric tolerates this if the encoder is door-invariant
        (testable; if not, we add door-control via reset(options=...)).
        """
        s = np.asarray(state, dtype=np.float32)
        agent_xy = s[0:2]
        target_xy = s[2:4]
        base = self._env.unwrapped
        base._set_state(agent_xy)
        base._set_goal_state(target_xy)
        # Refresh cached info so observe() / state() reflect the new pose.
        # Mirror env.step(): recompute distance + obs.
        import torch
        self._info['state'] = base.agent_position.detach().cpu().numpy()
        self._info['goal_state'] = base.target_position.detach().cpu().numpy()
        self._info['distance_to_target'] = float(
            torch.norm(base.agent_position - base.target_position)
        )
        self._obs = base._get_obs()


def expert_policy_tworoom(
    env: DMTwoRoomEnv,
    rng: np.random.Generator,
    *,
    action_noise: float = 2.0,
    action_repeat_prob: float = 0.05,
):
    """Path-C-style data-collection policy for TwoRoom.

    Constructs the canonical `ExpertPolicy` from stable_worldmodel with the
    SAME hyperparameters as the canonical data-collection script
    (`scripts/data/collect_tworooms.py`):
        action_noise=2.0, action_repeat_prob=0.05

    Returns a closure that takes a state (unused by the canonical policy —
    the policy reads from the env directly) and returns an action. The
    `state` and `rng` arguments are kept for API compatibility with our
    other env policies.

    Source for hyperparameters:
      https://github.com/galilai-group/stable-worldmodel/blob/main/scripts/data/collect_tworooms.py#L16
    """
    _ensure_imports()
    from stable_worldmodel.envs.two_room import ExpertPolicy
    policy = ExpertPolicy(
        action_noise=action_noise,
        action_repeat_prob=action_repeat_prob,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    policy.set_env(env._env)

    def step_policy(state: np.ndarray, rng_unused) -> np.ndarray:
        info_dict = {
            'state': env._info['state'],
            'goal_state': env._info['goal_state'],
        }
        return np.asarray(policy.get_action(info_dict), dtype=np.float32).reshape(2)

    return step_policy

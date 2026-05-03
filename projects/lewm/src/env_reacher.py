"""dm_control Reacher (easy) wrapper matching our env interface.

Reacher: 2-joint arm in 2D, target ball at random position. Action ∈ [-1,1]^2
joint torques. Goal: reach the target. Image obs at 64×64.

This wrapper exposes the same API as MiniPushTEnv:
    reset()                 -> (H, W, C) float32 in [0,1]
    step(action)            -> obs (same shape)
    state()                 -> (state_dim,) float32 — for diagnostics
    observe()               -> obs
    is_terminated()         -> bool, success criterion
    set_state(state)        -> seed env at recorded state (used by canonical eval)

Plus the WeakPolicy-style data generator (`weak_policy_reacher`) and a
`canonical_render_from_state(...)` helper for goal-frame reconstruction.
"""

from __future__ import annotations

import numpy as np

# Lazy import — dm_control is heavy
_suite = None
def _load_suite():
    global _suite
    if _suite is None:
        from dm_control import suite as _s  # noqa: F401
        _suite = _s
    return _suite


class DMReacherEnv:
    """64×64 RGB wrapper around dm_control reacher-easy.

    State exposed as 6-d: [joint_pos (2), to_target (2), joint_vel (2)].
    """

    state_dim = 6
    action_dim = 2
    image_size = 64

    def __init__(self, seed: int | None = None, image_size: int = 64):
        suite = _load_suite()
        self._env = suite.load(domain_name="reacher", task_name="easy",
                               task_kwargs={"random": seed})
        self._ts = self._env.reset()
        self.image_size = image_size
        # Cache action_spec
        self._aspec = self._env.action_spec()

    # ----------------------- gym-like API -----------------------

    def reset(self) -> np.ndarray:
        self._ts = self._env.reset()
        return self.observe()

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, self._aspec.minimum, self._aspec.maximum).astype(np.float64)
        self._ts = self._env.step(a)
        return self.observe()

    def observe(self) -> np.ndarray:
        img = self._env.physics.render(self.image_size, self.image_size, camera_id=0)
        return img.astype(np.float32) / 255.0

    def state(self) -> np.ndarray:
        obs = self._ts.observation
        # Order matches MiniPushT's "agent (2), goal-relative (2), velocity (2)"
        return np.concatenate(
            [obs["position"], obs["to_target"], obs["velocity"]]
        ).astype(np.float32)

    def to_target_distance(self) -> float:
        return float(np.linalg.norm(self._ts.observation["to_target"]))

    def is_terminated(self, threshold: float = 0.015) -> bool:
        """Default reacher-easy success: tip within `threshold` of target.
        threshold=0.015 matches dm_control's internal reward radius."""
        return self.to_target_distance() < threshold

    is_success = is_terminated

    # ---------------------- canonical-eval seeding ----------------------

    def set_state(self, state: np.ndarray):
        """Set env to a recorded 6-d state. Manipulates physics directly.

        State layout: [pos(2), to_target(2), vel(2)] — same as state().

        Order of operations matters: target_xy = finger_xy(qpos_new) + to_target,
        which requires finger Cartesian position to reflect the NEW qpos. We call
        physics.forward() inside the reset_context to propagate qpos through
        forward kinematics before reading finger_xy.
        """
        s = np.asarray(state, dtype=np.float64)
        pos = s[0:2]
        to_target = s[2:4]
        vel = s[4:6]
        phys = self._env.physics
        # Reacher MJCF: qpos[0:2] are joint angles (shoulder, elbow);
        # qvel[0:2] are joint angular velocities. Target position is a
        # standalone body whose Cartesian position is set via model.geom_pos.
        with phys.reset_context():
            phys.data.qpos[:2] = pos
            phys.data.qvel[:2] = vel
            # Forward-propagate qpos through kinematics so geom_xpos['finger']
            # reflects the new joint angles, NOT the previous reset's pose.
            phys.forward()
            finger_xy = phys.named.data.geom_xpos["finger"][:2].copy()
            target_xy = finger_xy + to_target
            phys.named.model.geom_pos["target", :2] = target_xy
        # Refresh the cached timestep so observe() returns the new pose.
        # We synthesize a TimeStep from the current physics rather than stepping
        # zero action (which would advance qvel under any nonzero qfrc/gravity).
        from dm_env import TimeStep, StepType
        obs = {
            "position": phys.data.qpos[:2].copy().astype(np.float64),
            "to_target": (
                phys.named.data.geom_xpos["target"][:2]
                - phys.named.data.geom_xpos["finger"][:2]
            ).astype(np.float64),
            "velocity": phys.data.qvel[:2].copy().astype(np.float64),
        }
        self._ts = TimeStep(step_type=StepType.MID, reward=0.0,
                            discount=1.0, observation=obs)


def weak_policy_reacher(
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    bias_toward_target: float = 0.4,
) -> np.ndarray:
    """Path C canonical data-collection policy for Reacher.

    Pure random torques rarely reach the target (low contact rate analogue —
    here, low "task-touching" rate). We bias actions toward the joint-angle
    direction that reduces to_target distance, mixed with random exploration.

    NOT a goal-reaching policy — just contact-rich data collection.
    """
    # Random component
    a_rand = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    if bias_toward_target <= 0:
        return a_rand
    # Bias component: torques along the gradient that reduces to_target.
    # Approximation: push joints in the direction of to_target (which is
    # already in finger-frame coordinates).
    to_target = state[2:4]
    norm = np.linalg.norm(to_target)
    if norm < 1e-6:
        return a_rand
    bias = (to_target / norm).astype(np.float32)
    a = (1.0 - bias_toward_target) * a_rand + bias_toward_target * bias
    return np.clip(a, -1.0, 1.0).astype(np.float32)

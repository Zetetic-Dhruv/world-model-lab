"""MiniPushT — a 64×64 mini version of the PushT manipulation benchmark.

Scene: green agent particle (4×4 px) + blue T-block (~10×10 bounding box) +
faint red T-outline at goal position. Agent moves under 2D velocity action.
When the agent's bounding box overlaps the block, the block translates by the
agent's effective movement that step (simplified contact, no rotation).

Goal observation: a frame where the block sits at the target. Used by the
CEM planner as the goal embedding target via encoder lookup.

State: (agent_x, agent_y, block_x, block_y, target_x, target_y) — 6-D.

Designed for the LeWM reproduction's planning eval: the encoder must
distinguish three colored shapes; the predictor must learn contact dynamics;
CEM in latent space must find action sequences that drive the block toward
the target.
"""

from __future__ import annotations

import numpy as np


def _draw_T(img: np.ndarray, cx: float, cy: float, color, alpha: float = 1.0):
    """Draw a T-shape centered at (cx, cy) on an HxWxC image, in-place.

    The T is a 10-px-wide horizontal bar (height 3) atop a 4-px-wide vertical
    stem (height 7). Bounding box is roughly 10×10. `alpha` blends with bg.
    """
    H, W, _ = img.shape
    color = np.asarray(color, dtype=np.float32)

    def fill(x0, x1, y0, y1):
        x0i = int(max(0, x0)); x1i = int(min(W, x1))
        y0i = int(max(0, y0)); y1i = int(min(H, y1))
        if x1i <= x0i or y1i <= y0i:
            return
        if alpha >= 1.0:
            img[y0i:y1i, x0i:x1i, :] = color
        else:
            img[y0i:y1i, x0i:x1i, :] = (
                (1 - alpha) * img[y0i:y1i, x0i:x1i, :] + alpha * color
            )

    # Top crossbar: 10 wide × 3 tall, centered at (cx, cy - 2)
    fill(cx - 5, cx + 5, cy - 4, cy - 1)
    # Stem: 4 wide × 7 tall, centered at (cx, cy + 1.5)
    fill(cx - 2, cx + 2, cy - 1, cy + 6)


class MiniPushTEnv:
    """Mini PushT environment with simplified contact dynamics."""

    def __init__(
        self,
        size: int = 64,
        agent_size: int = 4,
        action_scale: float = 4.0,
        noise: float = 0.05,
        success_radius: float = 6.0,
        seed: int | None = None,
    ):
        self.size = size
        self.agent_size = agent_size
        self.action_scale = action_scale
        self.noise = noise
        self.success_radius = success_radius
        self.rng = np.random.default_rng(seed)

        self.bg = np.array([0.95, 0.95, 0.95], dtype=np.float32)
        self.agent_color = np.array([0.20, 0.70, 0.30], dtype=np.float32)  # green
        self.block_color = np.array([0.20, 0.30, 0.85], dtype=np.float32)  # blue
        self.target_color = np.array([0.85, 0.30, 0.30], dtype=np.float32)  # faint red

        # Margin so the T's bounding box stays in-frame.
        self.margin = 8.0

        self.agent_x = 0.0
        self.agent_y = 0.0
        self.block_x = 0.0
        self.block_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.reset()

    def reset(self) -> np.ndarray:
        m, M = self.margin, self.size - self.margin
        # Place agent, block, target with minimum separations to keep tasks non-trivial.
        for _ in range(50):
            self.agent_x = float(self.rng.uniform(m, M))
            self.agent_y = float(self.rng.uniform(m, M))
            self.block_x = float(self.rng.uniform(m, M))
            self.block_y = float(self.rng.uniform(m, M))
            self.target_x = float(self.rng.uniform(m, M))
            self.target_y = float(self.rng.uniform(m, M))
            d_ab = np.hypot(self.agent_x - self.block_x, self.agent_y - self.block_y)
            d_bt = np.hypot(self.block_x - self.target_x, self.block_y - self.target_y)
            if d_ab > 8 and d_bt > 12:
                break
        return self.observe()

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, -1.0, 1.0).astype(np.float32) * self.action_scale
        a = a + self.rng.normal(0, self.noise, size=2).astype(np.float32)
        m, M = self.margin, self.size - self.margin

        # Tentative agent move
        new_ax = float(np.clip(self.agent_x + a[0], m, M))
        new_ay = float(np.clip(self.agent_y + a[1], m, M))
        dx = new_ax - self.agent_x
        dy = new_ay - self.agent_y

        # Contact check: agent bbox vs block bbox (simplified — bbox is the
        # T's enclosing 10×10).
        contact = (
            abs(new_ax - self.block_x) < (self.agent_size / 2 + 5)
            and abs(new_ay - self.block_y) < (self.agent_size / 2 + 5)
        )
        if contact:
            # Push: block translates by the agent's delta (full transfer).
            self.block_x = float(np.clip(self.block_x + dx, m, M))
            self.block_y = float(np.clip(self.block_y + dy, m, M))

        self.agent_x = new_ax
        self.agent_y = new_ay
        return self.observe()

    def observe(self) -> np.ndarray:
        """Render to (H, W, C) float32 in [0, 1]."""
        img = np.tile(self.bg, (self.size, self.size, 1)).copy()
        # Faint target T (drawn first so block can occlude it on success)
        _draw_T(img, self.target_x, self.target_y, self.target_color, alpha=0.35)
        # Block T
        _draw_T(img, self.block_x, self.block_y, self.block_color, alpha=1.0)
        # Agent (small square)
        ax = int(round(self.agent_x))
        ay = int(round(self.agent_y))
        h = self.agent_size // 2
        x0 = max(0, ax - h); x1 = min(self.size, ax + h)
        y0 = max(0, ay - h); y1 = min(self.size, ay + h)
        img[y0:y1, x0:x1, :] = self.agent_color
        return img

    def state(self) -> np.ndarray:
        return np.array([self.agent_x, self.agent_y,
                         self.block_x, self.block_y,
                         self.target_x, self.target_y], dtype=np.float32)

    def block_to_target_distance(self) -> float:
        return float(np.hypot(self.block_x - self.target_x,
                              self.block_y - self.target_y))

    def is_success(self) -> bool:
        return self.block_to_target_distance() < self.success_radius

    def goal_observation(self) -> np.ndarray:
        """[DEPRECATED for canonical eval] Render an OOD frame with block teleported to
        target while agent stays at current position. Path C uses real expert-trajectory
        goal frames instead — see eval_planning.py. Kept for diagnostic ablations only.
        """
        bx, by = self.block_x, self.block_y
        self.block_x = self.target_x
        self.block_y = self.target_y
        img = self.observe()
        self.block_x, self.block_y = bx, by
        return img

    def set_state(self, state: np.ndarray):
        """Set env state directly from a 6-d (agent_x, agent_y, block_x, block_y,
        target_x, target_y) vector. Used for expert-trajectory eval to seed an
        episode at a recorded init state."""
        s = np.asarray(state, dtype=np.float32)
        self.agent_x = float(s[0])
        self.agent_y = float(s[1])
        self.block_x = float(s[2])
        self.block_y = float(s[3])
        self.target_x = float(s[4])
        self.target_y = float(s[5])

    def is_terminated(self) -> bool:
        """Canonical (Gym-style) termination: True iff goal reached.
        Headline metric for Path C eval. Identical to is_success() under our
        simplified geometry; renamed to match Gym semantics."""
        return self.is_success()


def weak_policy(
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    env_size: int = 64,
    action_scale: float = 4.0,
    dist_constraint: float = 6.0,
) -> np.ndarray:
    """Path C canonical data-collection policy. Mirrors stable-worldmodel's
    WeakPolicy: contact-biased random exploration. NOT a goal-reaching policy.

    Algorithm (matches stable-worldmodel/envs/pusht/expert_policy.py:WeakPolicy):
      1. Sample uniform action in [-1, 1] (target world-coord delta).
      2. Convert to world-coordinate target = agent_pos + action * action_scale.
      3. Constrain target to a box around the block: target ∈
         [block - dist_constraint, block + dist_constraint].
      4. Rescale back to action space: action = (target - agent) / action_scale.

    Effect: agent's commanded position stays within `dist_constraint` of the
    block, producing contact-rich trajectories. No success requirement.

    Returns 2-d action in approximately [-1, 1]^2 (clipped at env.step level).
    """
    agent = state[:2].astype(np.float32)
    block = state[2:4].astype(np.float32)
    # Random delta in [-1, 1]
    a_raw = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    # World-coord target the agent would move toward
    target_world = agent + a_raw * action_scale
    # Constrain to box around block
    target_world = np.clip(target_world,
                           block - dist_constraint,
                           block + dist_constraint)
    # Rescale back to action space
    action = (target_world - agent) / action_scale
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def expert_policy(
    state: np.ndarray,
    env_size: int = 64,
    margin: float = 8.0,
    push_offset: float = 8.0,
    contact_radius: float = 7.0,
    side_radius: float = 12.0,
) -> np.ndarray:
    """Heuristic push-toward-target expert for MiniPushT.

    Decision tree:
      1. If agent is on the PUSH side (behind block from target): navigate to
         push position; if close enough, push toward target.
      2. If agent is on the TARGET side (would push block away from target):
         pick a SIDE WAYPOINT (90° off push_dir, on the side closer to agent)
         and navigate toward it. Once at the side, the next step's geometry
         puts the agent on the push side and the policy switches to (1).

    The key fix vs naive perpendicular sidestep: we navigate TO a concrete
    side waypoint (computed from block + perp * side_radius, clipped to env
    bounds), instead of just moving perpendicular indefinitely. Without the
    waypoint, the agent moves perpendicular until hitting a wall and stalls.

    Returns 2-d action in [-1, 1]^2.
    """
    agent = state[:2].astype(np.float32)
    block = state[2:4].astype(np.float32)
    target = state[4:6].astype(np.float32)

    push_dir = target - block
    push_dist = float(np.linalg.norm(push_dir))
    if push_dist < 1.0:
        return np.zeros(2, dtype=np.float32)
    push_dir = push_dir / push_dist
    perp = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)

    block_to_agent = agent - block
    bta_dist = float(np.linalg.norm(block_to_agent))

    # Decide side: agent is on push side iff dot(agent-block, -push_dir) > 0,
    # i.e. agent lies in the half-plane opposite to target across block's center.
    on_push_side = (bta_dist > 0.1 and float(np.dot(block_to_agent, -push_dir)) > 2.0)

    if on_push_side:
        # Phase B/C — straight navigate or push
        if bta_dist < contact_radius:
            return np.clip(push_dir, -1.0, 1.0).astype(np.float32)
        push_pos = block - push_dir * push_offset
        push_pos = np.clip(push_pos, margin, env_size - margin)
        diff = push_pos - agent
        diff_dist = float(np.linalg.norm(diff))
        if diff_dist > 0.5:
            return np.clip(diff / diff_dist, -1.0, 1.0).astype(np.float32)
        return np.clip(push_dir, -1.0, 1.0).astype(np.float32)

    # Phase A — agent is on target side; route via a side waypoint.
    # Pick the side closer to the agent.
    side_a = block + perp * side_radius
    side_b = block - perp * side_radius
    side_a_clipped = np.clip(side_a, margin, env_size - margin)
    side_b_clipped = np.clip(side_b, margin, env_size - margin)
    da = float(np.linalg.norm(side_a_clipped - agent))
    db = float(np.linalg.norm(side_b_clipped - agent))
    side_wp = side_a_clipped if da <= db else side_b_clipped

    diff = side_wp - agent
    diff_dist = float(np.linalg.norm(diff))
    if diff_dist > 0.5:
        return np.clip(diff / diff_dist, -1.0, 1.0).astype(np.float32)
    # If we're at the side already, take a step toward push_pos
    push_pos = block - push_dir * push_offset
    push_pos = np.clip(push_pos, margin, env_size - margin)
    diff2 = push_pos - agent
    return np.clip(diff2 / max(np.linalg.norm(diff2), 1e-6), -1.0, 1.0).astype(np.float32)

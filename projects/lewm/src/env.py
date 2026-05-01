"""2D particle environment for synthetic JEPA experiments.

A point particle (rendered as a small green square) lives in a 64x64 RGB scene.
State = (x, y); action = bounded velocity in R^2; observation = rendered scene.

Designed so that the latent should encode (x, y) — verifiable by linear probe.
"""

from __future__ import annotations

import numpy as np


class ParticleEnv:
    def __init__(
        self,
        size: int = 64,
        particle_size: int = 6,
        action_scale: float = 4.0,
        noise: float = 0.1,
        seed: int | None = None,
    ):
        self.size = size
        self.particle_size = particle_size
        self.action_scale = action_scale
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        # Background and foreground colors (near-white bg, green particle).
        self.bg = np.array([0.95, 0.95, 0.95], dtype=np.float32)
        self.fg = np.array([0.20, 0.70, 0.30], dtype=np.float32)
        self.x = 0.0
        self.y = 0.0
        self.reset()

    def reset(self) -> np.ndarray:
        margin = self.particle_size
        self.x = float(self.rng.uniform(margin, self.size - margin))
        self.y = float(self.rng.uniform(margin, self.size - margin))
        return self.observe()

    def step(self, action: np.ndarray) -> np.ndarray:
        # action: (2,) in [-1, 1]; clipped, scaled, noised.
        a = np.clip(action, -1.0, 1.0).astype(np.float32) * self.action_scale
        a = a + self.rng.normal(0, self.noise, size=2).astype(np.float32)
        margin = self.particle_size
        self.x = float(np.clip(self.x + a[0], margin, self.size - margin))
        self.y = float(np.clip(self.y + a[1], margin, self.size - margin))
        return self.observe()

    def observe(self) -> np.ndarray:
        """Render to (H, W, C) float32 in [0, 1]."""
        img = np.tile(self.bg, (self.size, self.size, 1))  # (H, W, C)
        half = self.particle_size // 2
        x_int = int(round(self.x))
        y_int = int(round(self.y))
        x0 = max(0, x_int - half)
        x1 = min(self.size, x_int + half)
        y0 = max(0, y_int - half)
        y1 = min(self.size, y_int + half)
        img[y0:y1, x0:x1, :] = self.fg
        return img

    def state(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

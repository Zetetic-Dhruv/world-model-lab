"""LinearWarmupCosineAnnealingLR — Round 3 canonical scheduler.

Matches stable-pretraining LinearWarmupCosineAnnealingLR semantics:
  - Linear warmup from warmup_start_lr (default 0) to base_lr over warmup_steps.
  - Cosine anneal from base_lr to eta_min (default 0) over remaining steps.

We implement directly with a LambdaLR multiplier so we don't depend on
stable-pretraining (which is not installed here).
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def linear_warmup_cosine_annealing(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_fraction: float = 0.01,
    eta_min_fraction: float = 0.0,
    warmup_start_fraction: float = 0.0,
) -> LambdaLR:
    """Build a LambdaLR that:
        steps in [0, warmup_steps]     : linear from warmup_start_fraction → 1.0
        steps in [warmup_steps, total] : cosine from 1.0 → eta_min_fraction

    Multipliers are fractions of the optimizer's base lr (set in AdamW).
    """
    if warmup_steps is None:
        warmup_steps = max(1, int(warmup_fraction * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear from warmup_start_fraction → 1.0
            t = step / max(1, warmup_steps)
            return warmup_start_fraction + (1.0 - warmup_start_fraction) * t
        # Cosine from 1.0 → eta_min_fraction across remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_fraction + (1.0 - eta_min_fraction) * cos

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

"""SIGReg — Sketched Isotropic Gaussian Regularizer (canonical-mechanism form).

Round 3 rewrite to match official lucas-maes/le-wm/module.py SIGReg:

- Input shape: (T, B, D)  — time, batch, embedding dim. SIGReg evaluates per
  time-step and averages.
- Test: trapezoidal quadrature over t ∈ [0, 3] with 17 knots; weighted by
  Gaussian window exp(-t² / 2). Empirical characteristic function via
  cos(x·t) and sin(x·t) means.
- Statistic scaling: multiplied by proj.size(-2) = B (sample size per
  projection at one time-step). This is the load-bearing factor that gives
  SIGReg actual gradient pressure.

Cramér–Wold theorem: a Borel measure on R^D is determined by its 1-D
projections. Match all 1-D marginals of µ_f to N(0,1) ⇔ µ_f = N(0, I).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_trapezoid_weights(knots: int = 17, t_max: float = 3.0,
                            device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (t, weights) for trapezoidal quadrature on [0, t_max] with `knots` knots.

    The weights match the official module.py convention:
        weights = [dt, 2·dt, 2·dt, …, 2·dt, dt]  (NOT the standard [dt/2, dt, …, dt/2])
    multiplied by the Gaussian window exp(-t²/2). This is 2× the textbook
    trapezoid coefficient, matching the official LeWM implementation. The
    factor cancels with their normalization elsewhere; we replicate exactly.
    """
    t = torch.linspace(0.0, t_max, knots, device=device, dtype=dtype)
    dt = t_max / (knots - 1)
    w_trap = torch.full((knots,), 2.0 * dt, device=device, dtype=dtype)
    w_trap[0] = dt
    w_trap[-1] = dt
    weights = w_trap * torch.exp(-t.pow(2) / 2.0)  # Gaussian-window weighting
    return t, weights


class SIGReg(torch.nn.Module):
    """Canonical SIGReg matching official module.py.

    Parameters
    ----------
    embed_dim : int
        D (latent dim).
    num_proj : int
        M (number of random unit projections). Default 1024.
    knots : int
        Quadrature knot count. Default 17.
    t_max : float
        Quadrature upper bound. Default 3.0.
    """

    def __init__(self, embed_dim: int, num_proj: int = 1024,
                 knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_proj = num_proj
        self.knots = knots
        self.t_max = t_max
        # Pre-compute quadrature nodes/weights as buffers (move with .to())
        t, weights = _make_trapezoid_weights(knots, t_max)
        self.register_buffer("t", t)              # (knots,)
        self.register_buffer("weights", weights)  # (knots,)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        z : (T, B, D) — time-leading layout (matches official transpose(0,1)).
            B is the per-time-step sample count used for the multiplier.

        Returns scalar tensor.
        """
        if z.dim() != 3:
            raise ValueError(f"SIGReg expects (T, B, D); got shape {tuple(z.shape)}")
        T, B, D = z.shape

        # Sample M unit-norm projections per call. Match official: A is (D, M) with
        # column-wise unit norm.
        A = torch.randn(D, self.num_proj, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        # Project: (T, B, D) @ (D, M) → (T, B, M)
        proj = z @ A

        # Outer-product with quadrature nodes: (T, B, M, knots)
        # x*t for each t ∈ self.t
        xt = proj.unsqueeze(-1) * self.t  # (T, B, M, knots)

        cos_mean = xt.cos().mean(dim=-3)  # average over B → (T, M, knots)
        sin_mean = xt.sin().mean(dim=-3)  # (T, M, knots)

        target = (-self.t.pow(2) / 2.0).exp()  # φ_0(t) = exp(-t²/2), (knots,)

        err = (cos_mean - target).pow(2) + sin_mean.pow(2)  # (T, M, knots)

        # Integrate against weights: err @ weights → (T, M)
        statistic = err @ self.weights  # (T, M)

        # KEY: scale by sample count per time-step (B). This is the ~11.5×
        # pressure factor missing from our pre-Round-3 implementation.
        statistic = statistic * proj.size(-2)  # (T, M)

        return statistic.mean()


def sigreg(z: torch.Tensor,
           num_proj: int = 1024,
           knots: int = 17,
           t_max: float = 3.0) -> torch.Tensor:
    """Functional form. Allocates fresh nodes/weights per call (slower than
    the SIGReg module). Use the module form in production training; this is
    for ad-hoc evaluation."""
    if z.dim() != 3:
        raise ValueError(f"sigreg expects (T, B, D); got shape {tuple(z.shape)}")
    module = SIGReg(z.size(-1), num_proj=num_proj, knots=knots, t_max=t_max)
    module = module.to(device=z.device, dtype=z.dtype)
    return module(z)


# ---------------------------------------------------------------------------
# Backward-compat shim: older tests called sigreg(z_2d) / epps_pulley_closed_form
# ---------------------------------------------------------------------------

def epps_pulley_closed_form(*args, **kwargs):
    raise NotImplementedError(
        "Round 3: closed-form Epps-Pulley replaced by canonical trapezoidal "
        "quadrature. Use SIGReg(...) module or sigreg(z) functional."
    )

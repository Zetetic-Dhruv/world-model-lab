"""Sanity checks for SIGReg.

Expected behavior:
  - Low on samples drawn from N(0, I).
  - High on collapsed (constant or near-constant) samples.
  - High on samples with non-Gaussian shape (Uniform, scaled).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lewm.sigreg import sigreg, epps_pulley_closed_form  # noqa: E402


def test_low_on_gaussian():
    torch.manual_seed(0)
    z = torch.randn(512, 64)
    val = sigreg(z, num_projections=128).item()
    print(f"  Gaussian N(0,I):     {val:+.4f}")
    return val


def test_high_on_collapse():
    torch.manual_seed(0)
    z = torch.zeros(512, 64) + 0.01 * torch.randn(1, 64)  # all rows equal
    val = sigreg(z, num_projections=128).item()
    print(f"  Collapsed (delta):   {val:+.4f}")
    return val


def test_high_on_uniform():
    torch.manual_seed(0)
    z = torch.empty(512, 64).uniform_(-2.0, 2.0)
    val = sigreg(z, num_projections=128).item()
    print(f"  Uniform[-2,2]:       {val:+.4f}")
    return val


def test_high_on_anisotropic():
    torch.manual_seed(0)
    z = torch.randn(512, 64)
    z[:, 0] = z[:, 0] * 5.0  # blow up first dim
    val = sigreg(z, num_projections=128).item()
    print(f"  Anisotropic Gaussian:{val:+.4f}")
    return val


def test_ep_1d_vs_2d():
    """Closed-form 1D and 2D paths must agree."""
    torch.manual_seed(0)
    h = torch.randn(256)
    v1 = epps_pulley_closed_form(h).item()
    v2 = epps_pulley_closed_form(h.unsqueeze(1)).item()
    print(f"  EP 1D vs 2D:         {v1:+.6f} vs {v2:+.6f}")
    assert abs(v1 - v2) < 1e-5, f"1D/2D mismatch: {v1} != {v2}"


if __name__ == "__main__":
    print("SIGReg sanity tests:")
    g = test_low_on_gaussian()
    c = test_high_on_collapse()
    u = test_high_on_uniform()
    a = test_high_on_anisotropic()
    print()
    test_ep_1d_vs_2d()
    print()
    print("Expected: Gaussian < others")
    if g < c and g < u and g < a:
        print("PASS: SIGReg orders cases correctly.")
    else:
        print(f"FAIL: ordering wrong. g={g} c={c} u={u} a={a}")
        sys.exit(1)

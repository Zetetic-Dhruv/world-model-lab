"""End-to-end smoke test: tiny model, tiny data, one forward+backward."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import ParticleEnv  # noqa: E402
from src.data import generate_trajectories, TrajectoryDataset  # noqa: E402
from src.lewm.model import LeWM, lewm_loss  # noqa: E402


def main():
    torch.manual_seed(0)

    # tiny env, tiny model
    print("[smoke] generating 5 episodes x 8 steps")
    obs, actions, _ = generate_trajectories(ParticleEnv, n_episodes=5, length=8, seed=0)
    print(f"        obs shape: {obs.shape}")

    ds = TrajectoryDataset(obs, actions, sub_len=4)
    print(f"        dataset windows: {len(ds)}")

    o, a = ds[0]
    print(f"        sample shapes: obs={tuple(o.shape)} act={tuple(a.shape)}")

    # tiny model — depth=2, fewer heads
    model = LeWM(
        image_size=64,
        patch_size=16,  # 4x4 = 16 patches
        in_chans=3,
        vit_dim=96,
        encoder_depth=2,
        encoder_heads=3,
        latent_dim=96,
        predictor_depth=2,
        predictor_heads=4,
        predictor_dropout=0.0,
        action_dim=2,
        max_history=4,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] tiny model params: {n_params:,}")

    # batch of 4 windows
    batch = [ds[i] for i in range(4)]
    obs_b = torch.stack([b[0] for b in batch])
    act_b = torch.stack([b[1] for b in batch])
    print(f"        batch: obs={tuple(obs_b.shape)} act={tuple(act_b.shape)}")

    z, z_pred = model(obs_b, act_b)
    print(f"        z: {tuple(z.shape)}  z_pred: {tuple(z_pred.shape)}")
    assert z.shape == z_pred.shape
    assert z.shape == (4, 4, 96)

    total, pred, sig = lewm_loss(z, z_pred, lambda_sigreg=0.1, num_projections=32)
    print(f"        L={total.item():.4f}  L_pred={pred.item():.4f}  L_sig={sig.item():.4f}")
    total.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"        params with non-zero grad: {has_grad}")
    assert has_grad > 0

    print("PASS: forward, loss, backward all run without error.")


if __name__ == "__main__":
    main()

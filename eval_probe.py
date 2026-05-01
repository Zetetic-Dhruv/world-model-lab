"""Linear probe: does the trained latent encode (x, y) particle position?

Verification target: R² > 0.9 for both x and y after training is healthy.
Also reports rollout MSE: predictor's open-loop multi-step prediction error.

Usage:
    python eval_probe.py --ckpt runs/v1/ckpt_epoch2.pt --data-dir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env import ParticleEnv
from src.data import generate_trajectories
from src.lewm.model import LeWM


def load_model(ckpt_path: str, device: torch.device) -> LeWM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    # Round 3 canonical architecture
    model = LeWM(
        image_size=64,
        patch_size=8,
        in_chans=3,
        vit_dim=192,
        encoder_depth=12,
        encoder_heads=3,
        latent_dim=192,
        proj_hidden_dim=2048,
        predictor_depth=6,
        predictor_heads=16,
        predictor_dim_head=64,
        predictor_mlp_dim=2048,
        predictor_dropout=0.1,
        action_dim=2,
        max_history=args["sub_len"],
        sigreg_num_proj=args.get("num_projections", 1024),
        sigreg_knots=args.get("sigreg_knots", 17),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def encode_trajectories(model: LeWM, obs: np.ndarray, device: torch.device) -> np.ndarray:
    """obs: (E, T, H, W, C). Returns z: (E, T, D)."""
    E, T = obs.shape[:2]
    z_all = []
    with torch.no_grad():
        for e in range(E):
            o = torch.from_numpy(obs[e]).permute(0, 3, 1, 2).float().to(device)  # (T, C, H, W)
            z = model.encoder(o)  # (T, D)
            z_all.append(z.cpu().numpy())
    return np.stack(z_all)


def linear_probe(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """z: (N, D), y: (N, k). Returns (r2_per_target, mse_per_target)."""
    from numpy.linalg import lstsq
    # add bias
    X = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
    # split 80/20
    n = X.shape[0]
    idx = np.random.RandomState(0).permutation(n)
    n_train = int(n * 0.8)
    Xtr, Xte = X[idx[:n_train]], X[idx[n_train:]]
    ytr, yte = y[idx[:n_train]], y[idx[n_train:]]
    W, *_ = lstsq(Xtr, ytr, rcond=None)
    yhat = Xte @ W
    mse = ((yhat - yte) ** 2).mean(axis=0)
    var = yte.var(axis=0)
    r2 = 1.0 - mse / (var + 1e-8)
    return r2, mse


def rollout_mse(
    model: LeWM,
    obs: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    horizon: int = 5,
    sub_len: int = 4,
) -> np.ndarray:
    """Open-loop rollout MSE per step.

    Encode the first `sub_len` frames; predictor rolls forward `horizon` steps using
    actions; compare ẑ_{t+k} vs encoder(o_{t+k}). Return mean MSE per step.
    """
    E, T = obs.shape[:2]
    if T < sub_len + horizon:
        return None
    mses = np.zeros(horizon)
    counts = 0
    with torch.no_grad():
        for e in range(E):
            o = torch.from_numpy(obs[e]).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)  # (1, T, C, H, W)
            a = torch.from_numpy(actions[e]).float().unsqueeze(0).to(device)  # (1, T, A)
            # use first sub_len frames as context (post-projector encoder output = emb)
            z_ctx = model.encoder(o[:, :sub_len])  # (1, sub_len, D)
            for k in range(horizon):
                # Round 3: predict() applies pred_proj on top of predictor output.
                # Loss in training was MSE(pred_proj(predictor(...)), encoder_emb_next).
                window_z = z_ctx[:, -sub_len:]
                window_a = a[:, k : k + sub_len]
                preds = model.predict(window_z, window_a)  # (1, sub_len, D), post-pred_proj
                next_z_pred = preds[:, -1:]  # last pos predicts next step
                # actual: encoder(next observation) — same topology as training target
                actual_o = o[:, sub_len + k : sub_len + k + 1]
                actual_z = model.encoder(actual_o)
                mses[k] += ((next_z_pred - actual_z) ** 2).mean().item()
                # advance context using the actual (un-projected-for-pred) emb
                z_ctx = torch.cat([z_ctx, actual_z], dim=1)
            counts += 1
    return mses / counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-eval-eps", type=int, default=200)
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"[eval] device={device}  ckpt={args.ckpt}")
    model = load_model(args.ckpt, device)

    print(f"[eval] generating {args.n_eval_eps} held-out episodes seed={args.seed}...")
    obs, actions, states = generate_trajectories(
        ParticleEnv, n_episodes=args.n_eval_eps, length=args.length, seed=args.seed
    )

    print("[eval] encoding...")
    z = encode_trajectories(model, obs, device)  # (E, T, D)
    print(f"        z stats: mean abs={np.abs(z).mean():.4f}  std along all={z.std():.4f}")

    # ---- Linear probe: latent -> (x, y) ----
    z_flat = z.reshape(-1, z.shape[-1])
    s_flat = states.reshape(-1, 2)
    r2, mse = linear_probe(z_flat, s_flat)
    print(f"[probe] linear latent -> (x, y):")
    print(f"        R²:  x={r2[0]:.3f}  y={r2[1]:.3f}")
    print(f"        MSE: x={mse[0]:.3f}  y={mse[1]:.3f}")

    # ---- Rollout MSE ----
    sub_len = 4
    if obs.shape[1] >= sub_len + args.horizon:
        mses = rollout_mse(model, obs, actions, device, horizon=args.horizon, sub_len=sub_len)
        print(f"[rollout] open-loop MSE per step (horizon={args.horizon}):")
        for k, m in enumerate(mses, 1):
            print(f"        step+{k}: {m:.4f}")
    else:
        print(f"[rollout] skipped (length {obs.shape[1]} < sub_len+horizon)")

    # ---- Latent geometry ----
    print(f"[geom] latent norm distribution:")
    norms = np.linalg.norm(z_flat, axis=-1)
    print(f"        ||z||: mean={norms.mean():.3f}  std={norms.std():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")


if __name__ == "__main__":
    main()

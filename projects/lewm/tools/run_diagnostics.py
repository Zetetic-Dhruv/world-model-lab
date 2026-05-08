"""Per-cell latent extraction + diagnostic_suite.

Loads a Path-C training checkpoint, samples N latent triples (Z_t, Z_{t+stride},
env_state_t) from the cell's heldout val split, runs the diagnostic_suite from
src/diagnostics.py, and saves a JSON-serializable result.

Designed to be invoked as a subprocess by tools/run_sweep.py per cell. Each
invocation is process-isolated.

Output JSON schema:
{
  "ckpt": str,
  "h5":   str,
  "n_samples":  int,
  "stride":     int,
  "image_size": int,
  "patch_size": int,
  "env":        str,
  "Z_dim":      int,
  "state_dim":  int,
  "metrics": {
    "latent_effective_rank_pr":      float,
    "latent_twonn_intrinsic_dim":    float,
    "env_state_twonn_intrinsic_dim": float,
    "env_state_effective_rank_pr":   float,
    "mi_z_envstate_nats":            float,
    "mi_z_znext_nats":               float
  }
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnostics import diagnostic_suite


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    from src.lewm.model import LeWM
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    if args.get("path_c", False):
        action_dim = 2 * args.get("path_c_stride", 5)
    else:
        action_dim = 2
    model = LeWM(
        image_size=args.get("image_size", 64),
        patch_size=args.get("patch_size", 8),
        in_chans=3,
        vit_dim=args.get("vit_dim", 192),
        encoder_depth=args.get("encoder_depth", 12),
        encoder_heads=args.get("encoder_heads", 3),
        latent_dim=args.get("latent_dim", 192),
        proj_hidden_dim=args.get("proj_hidden_dim", 2048),
        predictor_depth=args.get("predictor_depth", 6),
        predictor_heads=args.get("predictor_heads", 16),
        predictor_dim_head=args.get("predictor_dim_head", 64),
        predictor_mlp_dim=args.get("predictor_mlp_dim", 2048),
        predictor_dropout=args.get("predictor_dropout", 0.1),
        action_dim=action_dim,
        max_history=args.get("path_c_history", 3),
        sigreg_num_proj=args.get("num_projections", 1024),
        sigreg_knots=args.get("sigreg_knots", 17),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, args


def extract_latent_pairs(
    model,
    h5_path: str,
    val_eps: np.ndarray,
    stride: int,
    n_samples: int,
    device: torch.device,
    batch_size: int = 64,
    seed: int = 0,
):
    """Sample (Z_t, Z_{t+stride}, env_state_t) triples from val episodes.

    Returns: (Z, Z_next, env_state) with shapes (N, latent_dim), (N, latent_dim),
    (N, state_dim).
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        T = int(f["obs"].shape[1])
        max_s = T - stride - 1
        if max_s <= 0:
            raise ValueError(
                f"Episode length {T} too short for stride {stride}; "
                f"need at least stride+2 = {stride+2} env steps."
            )
        eps = np.asarray(val_eps)
        e_samples = rng.choice(eps, size=n_samples, replace=True)
        s_samples = rng.integers(0, max_s + 1, size=n_samples)

        # Pre-allocate
        H, W, C = f["obs"].shape[2:]
        S = f["states"].shape[2]
        obs_t = np.zeros((n_samples, H, W, C), dtype=np.float32)
        obs_next = np.zeros((n_samples, H, W, C), dtype=np.float32)
        state_t = np.zeros((n_samples, S), dtype=np.float32)

        # Pull data (one by one — h5 fancy indexing across non-contiguous
        # is slow but correct)
        for i, (e, s) in enumerate(zip(e_samples, s_samples)):
            o_t = f["obs"][int(e), int(s)]
            o_n = f["obs"][int(e), int(s) + stride]
            if o_t.dtype == np.uint8:
                obs_t[i] = o_t.astype(np.float32) / 255.0
                obs_next[i] = o_n.astype(np.float32) / 255.0
            else:
                obs_t[i] = o_t.astype(np.float32)
                obs_next[i] = o_n.astype(np.float32)
            state_t[i] = f["states"][int(e), int(s)]

    # Encode in batches: (B, H, W, C) -> (B, C, H, W) -> model.encode
    obs_t_chw = obs_t.transpose(0, 3, 1, 2)
    obs_next_chw = obs_next.transpose(0, 3, 1, 2)
    Z_list, Z_next_list = [], []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_t = torch.from_numpy(obs_t_chw[i:i + batch_size]).to(device)
            batch_n = torch.from_numpy(obs_next_chw[i:i + batch_size]).to(device)
            z = model.encode(batch_t).cpu().numpy()
            z_next = model.encode(batch_n).cpu().numpy()
            Z_list.append(z)
            Z_next_list.append(z_next)
    Z = np.concatenate(Z_list, axis=0)
    Z_next = np.concatenate(Z_next_list, axis=0)
    return Z, Z_next, state_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--h5", required=True,
                        help="Canonical HDF5 (states + obs).")
    parser.add_argument("--splits", required=True,
                        help="Path to splits.npz from training.")
    parser.add_argument("--out", required=True,
                        help="Path to write diagnostics.json.")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[diag] device={device}  ckpt={args.ckpt}")
    model, ckpt_args = load_model_from_ckpt(args.ckpt, device)

    splits = np.load(args.splits, allow_pickle=True)
    val_eps = splits["val"]
    stride = int(ckpt_args.get("path_c_stride", 5))

    print(f"[diag] extracting {args.n_samples} latent triples (stride={stride})...")
    Z, Z_next, env_state = extract_latent_pairs(
        model, args.h5, val_eps, stride, args.n_samples, device, seed=args.seed,
    )
    print(f"[diag] Z={Z.shape}  Z_next={Z_next.shape}  env_state={env_state.shape}")

    print("[diag] running diagnostic_suite...")
    metrics = diagnostic_suite(Z, env_state=env_state, Z_next=Z_next)

    output = {
        "ckpt": str(args.ckpt),
        "h5": str(args.h5),
        "n_samples": int(args.n_samples),
        "stride": stride,
        "image_size": int(ckpt_args.get("image_size", 64)),
        "patch_size": int(ckpt_args.get("patch_size", 8)),
        "env": str(ckpt_args.get("env", "unknown")),
        "Z_dim": int(Z.shape[1]),
        "state_dim": int(env_state.shape[1]),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[diag] saved → {args.out}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

"""LeWM-vanilla training — Round 3 canonical-mechanism config.

Round 3 changes vs Round 2:
  - Projection head: 2-layer MLP with BN-in-middle (matches official).
  - pred_proj: separate same-shape MLP applied to predictor outputs (asymmetric).
  - SIGReg: trapezoidal quadrature with knots=17, num_proj=1024, multiply by
    sample count B (per-time-step). Now expects (T, B, D).
  - Predictor: full DiT-style 6-param AdaLN-zero with multiplicative residual
    gates; heads=16, dim_head=64 (decoupled), mlp_dim=2048, pos_embed std=1.0.
  - Predictor outputs FULL next latent (not residual delta).
  - Optimizer: AdamW lr=5e-5, wd=1e-3.
  - Scheduler: LinearWarmupCosineAnnealingLR, 1% warmup (canonical).
  - λ_SIGReg = 0.09 (canonical, with N-scaling restored).
  - batch_size=128 if memory permits.

Held over from Round 2:
  - Threaded NaN supervisor (Fix D).
  - .contiguous() in encoder attention (Fix C).
  - Synthetic 64×64 particle environment.

Usage:
    python train.py --device cpu --no-wandb --out-dir runs/v3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env import ParticleEnv
from src.data import generate_trajectories, TrajectoryDataset
from src.lewm.model import LeWM, lewm_loss
from src.lewm.scheduler import linear_warmup_cosine_annealing
from src.lewm.trainer import NaNSupervisedTrainer


def get_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runs/v3")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--sub-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Round 3 canonical = 128. Reduce if OOM.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Round 3 canonical = 5e-5.")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="Round 3 canonical = 1e-3.")
    parser.add_argument("--lambda-sigreg", type=float, default=0.09,
                        help="Round 3 canonical = 0.09 with N-scaling.")
    parser.add_argument("--num-projections", type=int, default=1024)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--warmup-fraction", type=float, default=0.01,
                        help="Round 3 canonical = 1%.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--ring-depth", type=int, default=3)
    parser.add_argument("--nan-rate-threshold", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-project", default="lewm-vanilla")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_episodes = 50
        args.episode_length = 20
        args.epochs = 1
        args.batch_size = 16
        args.num_projections = 128

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"[init] device={device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    incident_dir = out_dir / "incidents"

    # ---------------------------------------------------------------- Data
    data_path = Path(args.data_dir) / (
        f"particle_n{args.n_episodes}_T{args.episode_length}_seed{args.seed}.npz"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        cache = np.load(data_path)
        obs, actions = cache["obs"], cache["actions"]
        print(f"[data] loaded {data_path.name}: obs={obs.shape}")
    else:
        print(f"[data] generating {args.n_episodes} eps × len {args.episode_length}")
        obs, actions, states = generate_trajectories(
            ParticleEnv,
            n_episodes=args.n_episodes,
            length=args.episode_length,
            action_dim=2,
            seed=args.seed,
        )
        np.savez_compressed(data_path, obs=obs, actions=actions, states=states)
        print(f"[data] cached → {data_path}")

    train_ds = TrajectoryDataset(obs, actions, sub_len=args.sub_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    n_batches = len(train_dl)
    total_steps = n_batches * args.epochs
    print(f"[data] dataset windows={len(train_ds)} batches/epoch={n_batches} "
          f"total_steps={total_steps}")

    # ---------------------------------------------------------------- Model
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
        max_history=args.sub_len,
        sigreg_num_proj=args.num_projections,
        sigreg_knots=args.sigreg_knots,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = linear_warmup_cosine_annealing(
        optimizer, total_steps=total_steps, warmup_fraction=args.warmup_fraction,
    )
    print(f"[opt] AdamW lr={args.lr} wd={args.weight_decay} "
          f"warmup={args.warmup_fraction:.1%} total_steps={total_steps}")

    # ---------------------------------------------------------------- Supervisor
    trainer = NaNSupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lewm_loss,
        loss_kwargs=dict(
            sigreg_module=model.sigreg,
            lambda_sigreg=args.lambda_sigreg,
        ),
        snapshot_every=args.snapshot_every,
        ring_depth=args.ring_depth,
        nan_rate_threshold=args.nan_rate_threshold,
        nan_window=100,
        grad_clip_norm=1.0,
        incident_dir=incident_dir,
    )

    # ---------------------------------------------------------------- Logging
    use_wandb = not args.no_wandb
    wb = None
    if use_wandb:
        try:
            import wandb as _wb
            wb = _wb
            wb.init(project=args.wandb_project, config=vars(args), dir=str(out_dir))
        except ImportError:
            use_wandb = False

    csv_path = out_dir / "train_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "epoch", "status", "loss", "loss_pred", "loss_sigreg",
         "z_std", "z_mean_abs", "lr", "nan_rate", "recoveries", "wallclock_s"]
    )

    # ---------------------------------------------------------------- Train
    print(f"[train] epochs={args.epochs} batch={args.batch_size} "
          f"λ={args.lambda_sigreg} M={args.num_projections}")
    start_t = time.time()

    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_t0 = time.time()
            for obs_b, act_b in train_dl:
                obs_b = obs_b.to(device, non_blocking=True)
                act_b = act_b.to(device, non_blocking=True)

                def fwd():
                    return model(obs_b, act_b)  # (emb, preds)

                result = trainer.step(fwd, batch_for_quarantine={
                    "obs_shape": tuple(obs_b.shape),
                    "act_shape": tuple(act_b.shape),
                })

                # Step the scheduler after every successful optimizer step.
                if result.status == "ok":
                    scheduler.step()

                if result.status == "aborted":
                    print(f"[FATAL] aborted at step {trainer.step_idx} (epoch {epoch})")
                    raise SystemExit(2)

                step = trainer.step_idx
                cur_lr = optimizer.param_groups[0]["lr"]

                if step > 0 and step % args.log_every == 0 and result.status == "ok":
                    stats = trainer.stats()
                    wall = time.time() - start_t
                    msg = (
                        f"[step {step:6d} ep {epoch}] "
                        f"L={result.loss:.4f} L_p={result.pred_loss:.4f} "
                        f"L_s={result.sigreg_loss:.4f} "
                        f"z_std={result.z_std:.3f} lr={cur_lr:.2e} "
                        f"nr={stats['nan_rate']:.3f} rec={stats['recovery_count']} "
                        f"({wall:.1f}s)"
                    )
                    print(msg, flush=True)
                    csv_writer.writerow(
                        [step, epoch, "ok", result.loss, result.pred_loss,
                         result.sigreg_loss, result.z_std, result.z_mean_abs, cur_lr,
                         stats['nan_rate'], stats['recovery_count'], wall]
                    )
                    csv_file.flush()
                    if use_wandb:
                        wb.log({
                            "loss/total": result.loss,
                            "loss/pred": result.pred_loss,
                            "loss/sigreg": result.sigreg_loss,
                            "diag/z_std": result.z_std,
                            "diag/z_mean_abs": result.z_mean_abs,
                            "opt/lr": cur_lr,
                            "supervisor/nan_rate": stats['nan_rate'],
                            "supervisor/recoveries": stats['recovery_count'],
                            "epoch": epoch,
                            "wallclock_s": wall,
                        }, step=step)
                elif result.status == "recovered":
                    stats = trainer.stats()
                    wall = time.time() - start_t
                    print(f"[step {trainer.step_idx} ep {epoch}] "
                          f"RECOVERED ({result.nan_kind}); "
                          f"recoveries={stats['recovery_count']} "
                          f"nan_rate={stats['nan_rate']:.3f}", flush=True)

            print(f"[epoch {epoch}] done in {time.time() - epoch_t0:.1f}s; "
                  f"stats={trainer.stats()}")
            ckpt_path = out_dir / f"ckpt_epoch{epoch}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": trainer.step_idx,
                "epoch": epoch,
                "args": vars(args),
                "supervisor_stats": trainer.stats(),
            }, ckpt_path)
            print(f"[ckpt] saved → {ckpt_path}")

    finally:
        trainer.shutdown()
        csv_file.close()
        if use_wandb:
            wb.finish()

    print(f"[done] total wallclock={time.time() - start_t:.1f}s")


if __name__ == "__main__":
    main()

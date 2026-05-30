"""Per-resolution convergence figures for the Option-D trajectory study.

For each resolution cell, plots the convergence trajectories that discriminate
H1 (converged) / H2 (under-trained) / H3 (resolution-dependent convergence):

  (a) MI(z, env_state) + MI(z, z_next) vs epoch   [sparse: kept checkpoints]
  (b) Effective rank of latents vs epoch          [sparse: kept checkpoints]
  (c) Predictive accuracy: val L_pred vs epoch     [dense: every epoch, log-y]
  (d) Predictor/identity ratio vs epoch            [dense, log-y, y=1 reference]

Data sources per cell dir runs/sweep_traj100/reacher/img{N}/:
  - diagnostics_epoch{E}.json + diagnostics.json (final) → MI, effective rank
  - train_log.csv (status=val_epoch rows) → val_L_pred, P/I ratio per epoch

"Reward" (planning success) is NOT plotted: the trajectory sweep ran with
--skip-eval (no per-checkpoint planning eval). Predictive accuracy (val L_pred,
P/I) is the per-epoch model-quality axis. To add reward-vs-epoch, run
eval_planning.py on each kept checkpoint (see --with-reward note in README).

Usage:
    python tools/plot_trajectory.py --sweep-dir runs/sweep_traj100/reacher
    # → writes runs/sweep_traj100/reacher/fig_trajectory_img{N}.png per cell
    #   and a combined fig_trajectory_overlay.png across resolutions
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_dense_val(train_log_csv: Path):
    """Return (epochs, val_L_pred, pi_ratio) from train_log.csv val_epoch rows."""
    epochs, lpred, pir = [], [], []
    with open(train_log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "val_epoch":
                continue
            try:
                e = int(row["epoch"])
                lp = float(row["val_L_pred"])
                pi = float(row["val_pred_vs_identity_ratio"])
            except (ValueError, KeyError):
                continue
            epochs.append(e)
            lpred.append(lp)
            pir.append(pi)
    # sort by epoch
    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    return ([epochs[i] for i in order],
            [lpred[i] for i in order],
            [pir[i] for i in order])


def read_sparse_diag(cell_dir: Path, final_epoch: int):
    """Return (epochs, ER, MI_state, MI_znext) from diagnostics_epoch*.json
    plus diagnostics.json mapped to final_epoch."""
    pts = []
    for f in cell_dir.glob("diagnostics_epoch*.json"):
        m = re.search(r"epoch(\d+)", f.stem)
        if not m:
            continue
        e = int(m.group(1))
        d = json.loads(f.read_text()).get("metrics", {})
        pts.append((e, d))
    final = cell_dir / "diagnostics.json"
    if final.exists():
        d = json.loads(final.read_text()).get("metrics", {})
        pts.append((final_epoch, d))
    pts.sort(key=lambda x: x[0])
    epochs = [e for e, _ in pts]
    er = [d.get("latent_effective_rank_pr", float("nan")) for _, d in pts]
    mis = [d.get("mi_z_envstate_nats", float("nan")) for _, d in pts]
    miz = [d.get("mi_z_znext_nats", float("nan")) for _, d in pts]
    return epochs, er, mis, miz


def plot_cell(cell_dir: Path, image_size: int, out_path: Path):
    train_log = cell_dir / "train_log.csv"
    if not train_log.exists():
        print(f"[plot] skip img{image_size}: no train_log.csv")
        return None
    d_epochs, d_lpred, d_pir = read_dense_val(train_log)
    final_epoch = max(d_epochs) if d_epochs else 99
    s_epochs, er, mis, miz = read_sparse_diag(cell_dir, final_epoch)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Reacher @ {image_size}×{image_size} — convergence trajectory "
                 f"(100 epochs, tiny preset, seed 42)", fontsize=13, weight="bold")

    # (a) MI vs epoch
    ax = axes[0, 0]
    ax.plot(s_epochs, mis, "o-", color="C0", label="MI(z, env_state)")
    ax.plot(s_epochs, miz, "s--", color="C1", label="MI(z, z_next)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MI [nats]")
    ax.set_title("(a) Mutual information vs epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Effective rank vs epoch
    ax = axes[0, 1]
    ax.plot(s_epochs, er, "o-", color="C2")
    ax.set_xlabel("epoch"); ax.set_ylabel("effective rank (PR) / 192")
    ax.set_title("(b) Latent effective rank vs epoch")
    ax.grid(alpha=0.3)

    # (c) Predictive accuracy: val L_pred vs epoch (log-y)
    ax = axes[1, 0]
    ax.plot(d_epochs, d_lpred, "-", color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("val L_pred (lower = better)")
    ax.set_title("(c) Predictive accuracy vs epoch")
    ax.grid(alpha=0.3, which="both")

    # (d) Predictor/identity ratio vs epoch (log-y, y=1 ref)
    ax = axes[1, 1]
    ax.plot(d_epochs, d_pir, "-", color="C4")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="beats identity (P/I=1)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("predictor / identity ratio")
    ax.set_title("(d) Predictor vs identity baseline")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return {"image_size": image_size, "epochs": s_epochs, "ER": er,
            "MI_state": mis, "dense_epochs": d_epochs, "val_lpred": d_lpred}


def plot_overlay(cells: list, out_path: Path):
    """Overlay MI(z,state) and effective rank across resolutions."""
    if not cells:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Reacher resolution sweep — convergence across image sizes",
                 fontsize=13, weight="bold")
    for c in cells:
        lbl = f"{c['image_size']}px"
        axes[0].plot(c["epochs"], c["MI_state"], "o-", label=lbl)
        axes[1].plot(c["epochs"], c["ER"], "o-", label=lbl)
        axes[2].plot(c["dense_epochs"], c["val_lpred"], "-", label=lbl)
    axes[0].set_title("MI(z, env_state) vs epoch"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MI [nats]"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Effective rank vs epoch"); axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("effective rank (PR)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].set_title("Predictive accuracy (val L_pred) vs epoch")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val L_pred")
    axes[2].set_yscale("log"); axes[2].legend(); axes[2].grid(alpha=0.3, which="both")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True,
                        help="Dir containing img{N}/ cell subdirs "
                             "(e.g. runs/sweep_traj100/reacher)")
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir)

    cells = []
    for cell_dir in sorted(sweep_dir.glob("img*"),
                           key=lambda p: int(p.name.replace("img", ""))):
        image_size = int(cell_dir.name.replace("img", ""))
        out = sweep_dir / f"fig_trajectory_img{image_size}.png"
        res = plot_cell(cell_dir, image_size, out)
        if res:
            cells.append(res)
    if len(cells) > 1:
        plot_overlay(cells, sweep_dir / "fig_trajectory_overlay.png")
    print(f"[plot] done: {len(cells)} cell figure(s)")


if __name__ == "__main__":
    main()

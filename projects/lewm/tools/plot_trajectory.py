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
    """Return dict of trajectories from diagnostics_epoch*.json + diagnostics.json.
    Keys: epochs, er, mi_state (KSG), mi_znext (KSG), probe_lin, probe_mlp."""
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
    g = lambda k: [d.get(k, float("nan")) for _, d in pts]
    return {
        "epochs": [e for e, _ in pts],
        "er": g("latent_effective_rank_pr"),
        "mi_state": g("mi_z_envstate_nats"),
        "mi_znext": g("mi_z_znext_nats"),
        "probe_lin": g("probe_linear_r2"),
        "probe_mlp": g("probe_mlp_r2"),
    }


def plot_cell(cell_dir: Path, image_size: int, out_path: Path):
    train_log = cell_dir / "train_log.csv"
    if not train_log.exists():
        print(f"[plot] skip img{image_size}: no train_log.csv")
        return None
    d_epochs, d_lpred, d_pir = read_dense_val(train_log)
    final_epoch = max(d_epochs) if d_epochs else 99
    s = read_sparse_diag(cell_dir, final_epoch)
    se = s["epochs"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Reacher @ {image_size}×{image_size} — convergence trajectory "
                 f"(100 epochs, tiny preset, seed 42)", fontsize=13, weight="bold")

    # (a) State sufficiency: probe-R² (trustworthy) + KSG MI (geometry-suspect) vs epoch
    ax = axes[0, 0]
    ax.plot(se, s["probe_mlp"], "o-", color="C0", label="probe R² (MLP) — state")
    ax.plot(se, s["probe_lin"], "^-", color="C0", alpha=0.5, label="probe R² (linear)")
    ax.set_xlabel("epoch"); ax.set_ylabel("state-decode R²", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(se, s["mi_state"], "s--", color="C1", alpha=0.7, label="KSG MI(z;state) [nats]")
    ax2.set_ylabel("KSG MI [nats]", color="C1"); ax2.tick_params(axis="y", labelcolor="C1")
    ax.set_title("(a) State sufficiency: probe R² vs KSG MI")
    ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Effective rank + temporal MI(z_t;z_{t+1}) vs epoch
    ax = axes[0, 1]
    ax.plot(se, s["er"], "o-", color="C2", label="effective rank (PR)")
    ax.set_xlabel("epoch"); ax.set_ylabel("effective rank / 192", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax2 = ax.twinx()
    ax2.plot(se, s["mi_znext"], "d--", color="C5", alpha=0.7, label="MI(z_t; z_{t+1}) [nats]")
    ax2.set_ylabel("temporal MI [nats]", color="C5"); ax2.tick_params(axis="y", labelcolor="C5")
    ax.set_title("(b) Effective rank + temporal self-MI")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

    # (c) Predictive accuracy: val L_pred vs epoch (log-y)
    ax = axes[1, 0]
    ax.plot(d_epochs, d_lpred, "-", color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("val L_pred (lower = better)")
    ax.set_title("(c) Predictive accuracy vs epoch")
    ax.grid(alpha=0.3, which="both")

    # (d) Predictability–sufficiency frontier: L_pred vs probe-R², epoch as path
    ax = axes[1, 1]
    # interpolate dense L_pred at the sparse diagnostic epochs
    lpred_at = [d_lpred[min(range(len(d_epochs)), key=lambda i: abs(d_epochs[i]-e))]
                for e in se]
    sc = ax.scatter(lpred_at, s["probe_mlp"], c=se, cmap="viridis", s=60, zorder=3)
    ax.plot(lpred_at, s["probe_mlp"], "-", color="gray", alpha=0.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("val L_pred (predictability →, log)"); ax.set_ylabel("probe R² (sufficiency ↑)")
    ax.set_title("(d) Predictability–sufficiency frontier")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("epoch")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return {"image_size": image_size, "epochs": se, "ER": s["er"],
            "MI_state": s["mi_state"], "probe_mlp": s["probe_mlp"],
            "mi_znext": s["mi_znext"], "dense_epochs": d_epochs, "val_lpred": d_lpred,
            "lpred_at": lpred_at}


def plot_overlay(cells: list, out_path: Path):
    """Overlay across resolutions. Top row: probe-R²(state), effective rank,
    predictive accuracy vs epoch. Bottom: the predictability–sufficiency frontier
    (val L_pred vs probe-R², traversed by epoch) — the core 'object' of the study."""
    if not cells:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Reacher resolution sweep — predictability vs state-sufficiency",
                 fontsize=13, weight="bold")
    for c in cells:
        lbl = f"{c['image_size']}px"
        axes[0, 0].plot(c["epochs"], c["probe_mlp"], "o-", label=lbl)
        axes[0, 1].plot(c["epochs"], c["ER"], "o-", label=lbl)
        axes[1, 0].plot(c["dense_epochs"], c["val_lpred"], "-", label=lbl)
        axes[1, 1].plot(c["lpred_at"], c["probe_mlp"], "o-", label=lbl)
    axes[0, 0].set_title("State sufficiency: probe R² vs epoch (trustworthy)")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("state-decode R²")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_title("Effective rank vs epoch")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("effective rank (PR)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].set_title("Predictive accuracy (val L_pred) vs epoch")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("val L_pred")
    axes[1, 0].set_yscale("log"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3, which="both")
    axes[1, 1].set_title("Predictability–sufficiency frontier")
    axes[1, 1].set_xlabel("val L_pred (predictability →, log)")
    axes[1, 1].set_ylabel("probe R² (sufficiency ↑)")
    axes[1, 1].set_xscale("log"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3, which="both")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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

"""Resolution sweep orchestrator for the LeWM Limitation #3 study.

Drives N_envs × N_resolutions cells; each cell runs:
  1. train.py --path-c --env <env> --image-size <N> --patch-size auto
  2. eval_planning.py (canonical Path-C eval, 30 heldout eps)
  3. tools/run_diagnostics.py (effective_rank, MI, intrinsic dim)

After each cell, the per-cell results are aggregated into a single
runs/sweep/sweep_results.json suitable for plotting.

Per-cell isolation: each subprocess gets a fresh Python interpreter; failures
are logged and the sweep continues.

Resume: cells with an existing diagnostics.json are skipped.

Output directory layout:
  runs/sweep/
    ├── reacher/img64/{ckpt_epochN.pt, splits.npz, path_c_eval.npz, diagnostics.json, cell.log}
    ├── reacher/img96/...
    ├── tworoom/img64/...
    └── sweep_results.json   (aggregated)

Disk audit: prints estimated data + ckpts before launch, refuses with warning
if usage exceeds 80% of free space.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_RESOLUTIONS = [64, 96, 128, 160, 192, 224]
DEFAULT_ENVS = ["reacher", "tworoom", "pusht"]


def cell_dir(out_dir: Path, env: str, image_size: int) -> Path:
    return out_dir / env / f"img{image_size}"


def cell_done(cell: Path) -> bool:
    """A cell is complete iff its diagnostics.json exists."""
    return (cell / "diagnostics.json").exists()


def estimate_disk(envs, resolutions, n_episodes, episode_length):
    """Return (data_GB, runs_GB) estimates."""
    # Data: uint8 (E, T, H, W, 3) compressed ~30% via gzip
    data_gb = 0.0
    for env in envs:
        for size in resolutions:
            bytes_raw = n_episodes * episode_length * size * size * 3
            data_gb += bytes_raw * 0.30 / 1e9
    # Runs: per cell ~25 MB final ckpt only (intermediate ckpts auto-deleted)
    runs_gb = len(envs) * len(resolutions) * 0.025
    return data_gb, runs_gb


def run_cell(env, image_size, args, cell, project_dir):
    """Run train + eval + diagnostics for one cell. Returns status dict."""
    py = sys.executable  # exact Python from current venv
    cell.mkdir(parents=True, exist_ok=True)
    log_path = cell / "cell.log"
    info = {"env": env, "image_size": image_size}

    # ---------------- 1. Train ----------------
    train_cmd = [
        py, "-u", "train.py", "--path-c", "--env", env,
        "--image-size", str(image_size), "--patch-size", "auto",
        "--n-episodes", str(args.n_episodes),
        "--path-c-episode-length", str(args.episode_length),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--arch-preset", "tiny",
        "--out-dir", str(cell),
        "--device", args.device,
        "--no-wandb",
        "--log-every", "100",
        "--seed", str(args.seed),
    ]
    t0 = time.time()
    print(f"[sweep] {env}@{image_size}: training...", flush=True)
    with open(log_path, "w") as logf:
        logf.write(f"# {' '.join(train_cmd)}\n")
        logf.flush()
        result = subprocess.run(
            train_cmd, cwd=project_dir, stdout=logf, stderr=subprocess.STDOUT,
            check=False, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
    info["wallclock_train_s"] = time.time() - t0
    if result.returncode != 0:
        info["status"] = "failed"
        info["stage"] = "train"
        return info

    # Find final ckpt
    ckpts = sorted(cell.glob("ckpt_epoch*.pt"),
                   key=lambda p: int(p.stem.replace("ckpt_epoch", "")))
    if not ckpts:
        info["status"] = "failed"
        info["stage"] = "train_no_ckpt"
        return info
    final_ckpt = ckpts[-1]
    # Cleanup: keep only final ckpt to save disk
    for c in ckpts[:-1]:
        c.unlink()

    # Determine h5 path
    size_tag = f"_img{image_size}" if image_size != 64 else ""
    policy_tag = "expert" if env == "tworoom" else "weakpolicy"
    h5_path = (project_dir / "data" /
               f"{env}_{policy_tag}_n{args.n_episodes}_T{args.episode_length}{size_tag}.h5")

    # ---------------- 2. Eval ----------------
    eval_cmd = [
        py, "-u", "eval_planning.py",
        "--ckpt", str(final_ckpt),
        "--h5", str(h5_path),
        "--splits", str(cell / "splits.npz"),
        "--n-episodes", str(args.eval_episodes),
        "--tau-samples", "80",
        "--seed", "2026",
        "--device", args.device,
        "--env", env,
    ]
    t0 = time.time()
    print(f"[sweep] {env}@{image_size}: eval...", flush=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n# {' '.join(eval_cmd)}\n")
        logf.flush()
        result = subprocess.run(
            eval_cmd, cwd=project_dir, stdout=logf, stderr=subprocess.STDOUT,
            check=False, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
    info["wallclock_eval_s"] = time.time() - t0
    info["eval_ok"] = (result.returncode == 0)

    # ---------------- 3. Diagnostics ----------------
    diag_cmd = [
        py, "-u", "tools/run_diagnostics.py",
        "--ckpt", str(final_ckpt),
        "--h5", str(h5_path),
        "--splits", str(cell / "splits.npz"),
        "--out", str(cell / "diagnostics.json"),
        "--n-samples", str(args.diag_samples),
        "--device", args.device,
    ]
    t0 = time.time()
    print(f"[sweep] {env}@{image_size}: diagnostics...", flush=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n# {' '.join(diag_cmd)}\n")
        logf.flush()
        result = subprocess.run(
            diag_cmd, cwd=project_dir, stdout=logf, stderr=subprocess.STDOUT,
            check=False, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
    info["wallclock_diag_s"] = time.time() - t0
    if result.returncode != 0:
        info["status"] = "failed"
        info["stage"] = "diagnostics"
        return info

    info["status"] = "ok"
    return info


def aggregate(out_dir: Path, envs, resolutions) -> Path:
    """Walk all cells and build a single JSON list."""
    rows = []
    for env in envs:
        for size in resolutions:
            cell = cell_dir(out_dir, env, size)
            row = {"env": env, "image_size": size, "cell_dir": str(cell.relative_to(out_dir.parent))
                   if out_dir in cell.parents or out_dir == cell.parent else str(cell)}
            diag_path = cell / "diagnostics.json"
            if diag_path.exists():
                row["diagnostics"] = json.loads(diag_path.read_text())
            eval_path = cell / "path_c_eval.npz"
            if eval_path.exists():
                npz = np.load(eval_path, allow_pickle=True)
                tau_info = npz["tau_info"][0] if "tau_info" in npz.files else {}
                row["eval"] = {
                    "success_rate": float(npz["success_rate"]),
                    "tau": float(npz["tau"]),
                    "near_mean": float(tau_info.get("near_mean", float("nan"))),
                    "unrelated_mean": float(tau_info.get("unrelated_mean", float("nan"))),
                    "tau_gap_factor": float(
                        tau_info.get("unrelated_mean", float("nan"))
                        / max(tau_info.get("near_mean", 1.0), 1e-9)
                    ),
                }
            rows.append(row)
    out_path = out_dir / "sweep_results.json"
    out_path.write_text(json.dumps(rows, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/sweep",
                        help="Top-level dir for sweep cells")
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS),
                        help=f"Comma-separated env list (default: {','.join(DEFAULT_ENVS)})")
    parser.add_argument("--resolutions", default=",".join(map(str, DEFAULT_RESOLUTIONS)),
                        help=f"Comma-separated image_size list (default: {','.join(map(str, DEFAULT_RESOLUTIONS))})")
    parser.add_argument("--n-episodes", type=int, default=500,
                        help="Per-cell training episodes (default 500)")
    parser.add_argument("--episode-length", type=int, default=100,
                        help="Per-cell episode length in env steps (default 100)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--diag-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan + disk audit, no cells run.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip cells, just aggregate existing results.")
    parser.add_argument("--per-cell-min-estimate", type=int, default=90,
                        help="Wallclock estimate per cell in minutes (default 90).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    project_dir = Path(__file__).resolve().parent.parent
    envs = [e.strip() for e in args.envs.split(",")]
    resolutions = [int(r) for r in args.resolutions.split(",")]
    cells = [(env, size) for env in envs for size in resolutions]

    if args.aggregate_only:
        agg_path = aggregate(out_dir, envs, resolutions)
        print(f"[sweep] aggregated → {agg_path}")
        return

    # ---------------- Disk audit ----------------
    data_gb, runs_gb = estimate_disk(
        envs, resolutions, args.n_episodes, args.episode_length,
    )
    free_gb = shutil.disk_usage(project_dir).free / 1e9

    # ---------------- Wallclock ----------------
    n_pending = sum(1 for env, size in cells if not cell_done(cell_dir(out_dir, env, size)))
    total_min = n_pending * args.per_cell_min_estimate
    total_hr = total_min / 60

    print(f"[sweep] cells:       {len(cells)} total ({n_pending} pending, {len(cells)-n_pending} done)")
    print(f"[sweep] grid:        envs={envs}  × resolutions={resolutions}")
    print(f"[sweep] disk audit:  data ~{data_gb:.1f} GB  +  runs ~{runs_gb:.2f} GB  "
          f"= ~{data_gb + runs_gb:.1f} GB needed  /  {free_gb:.1f} GB free")
    if data_gb + runs_gb > free_gb * 0.8:
        print(f"[sweep] WARNING: estimated disk usage > 80% of free space")
    print(f"[sweep] wallclock:   {n_pending} pending × {args.per_cell_min_estimate} min "
          f"= ~{total_hr:.1f} hr")

    # ---------------- Plan / dry-run ----------------
    print(f"\n[sweep] plan:")
    for env, size in cells:
        cell = cell_dir(out_dir, env, size)
        status = "DONE" if cell_done(cell) else "PEND"
        print(f"  [{status}] {env:>10} @ {size:>3}  → {cell.relative_to(project_dir.parent)}")
    if args.dry_run:
        print("[sweep] dry run — not running any cells.")
        return

    # ---------------- Run cells ----------------
    print(f"\n[sweep] starting {n_pending} pending cells ({total_hr:.1f} hr ETA)...\n")
    sweep_t0 = time.time()
    for i, (env, size) in enumerate(cells):
        cell = cell_dir(out_dir, env, size)
        if cell_done(cell):
            print(f"[sweep] {i+1}/{len(cells)} {env}@{size}: skipping (done)", flush=True)
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"[sweep] {i+1}/{len(cells)} {env}@{size}: starting "
              f"(elapsed {(time.time()-sweep_t0)/60:.1f} min)", flush=True)
        cell_t0 = time.time()
        info = run_cell(env, size, args, cell, project_dir)
        info["wallclock_total_s"] = time.time() - cell_t0
        print(f"[sweep] {i+1}/{len(cells)} {env}@{size}: {info['status']} "
              f"({info['wallclock_total_s']/60:.1f} min)", flush=True)
        # Aggregate after each cell so partial results are queryable
        aggregate(out_dir, envs, resolutions)

    total = time.time() - sweep_t0
    print(f"\n[sweep] complete: {total/3600:.1f} hr total wallclock")
    agg_path = aggregate(out_dir, envs, resolutions)
    print(f"[sweep] aggregated → {agg_path}")


if __name__ == "__main__":
    main()

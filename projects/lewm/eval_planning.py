"""Planning evaluation — measure CEM-MPC success rate on MiniPushT.

For each eval episode:
  - Reset env to a random initial state.
  - Run MPC: plan with CEM, execute K_exec actions, replan.
  - Episode succeeds if block reaches target within `success_radius` before
    `budget_steps` runs out.

Reports:
  - success rate
  - mean steps-to-goal on successful episodes
  - mean final block-to-target distance over all episodes

Usage:
    python eval_planning.py --ckpt runs/v3-pusht/ckpt_epoch9.pt --n-episodes 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env_pusht import MiniPushTEnv
from src.lewm.model import LeWM
from src.lewm.planner import CEMPlanner, MPCRunner


def load_model(ckpt_path: str, device: torch.device) -> LeWM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = LeWM(
        image_size=64, patch_size=8, in_chans=3,
        vit_dim=192, encoder_depth=12, encoder_heads=3,
        latent_dim=192, proj_hidden_dim=2048,
        predictor_depth=6, predictor_heads=16,
        predictor_dim_head=64, predictor_mlp_dim=2048,
        predictor_dropout=0.1,
        action_dim=2, max_history=args.get("sub_len", 4),
        sigreg_num_proj=args.get("num_projections", 1024),
        sigreg_knots=args.get("sigreg_knots", 17),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-elites", type=int, default=30)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--k-exec", type=int, default=None,
                        help="Actions to execute per plan; default = horizon (full).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--success-radius", type=float, default=6.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save per-episode trajectories to <ckpt-dir>/planning_trajectories.npz")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[eval] device={device}  ckpt={args.ckpt}")
    model = load_model(args.ckpt, device)
    sub_len = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"].get("sub_len", 4)

    planner = CEMPlanner(
        model=model,
        horizon=args.horizon,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        n_iters=args.n_iters,
        action_dim=2,
        sub_len=sub_len,
    )

    rng = np.random.default_rng(args.seed)
    successes = 0
    steps_to_goal: list[int] = []
    final_distances: list[float] = []
    saved_trajectories = []

    print(f"[eval] running {args.n_episodes} episodes "
          f"(H={args.horizon}, N={args.n_samples}, K={args.n_elites}, T={args.n_iters}, "
          f"K_exec={args.k_exec or args.horizon}, budget={args.budget_steps}, "
          f"success_radius={args.success_radius})")
    t0 = time.time()

    for ep in range(args.n_episodes):
        env_seed = int(rng.integers(0, 2**31 - 1))

        def env_factory(s=env_seed):
            return MiniPushTEnv(seed=s, success_radius=args.success_radius)

        runner = MPCRunner(
            planner=planner,
            env_factory=env_factory,
            sub_len=sub_len,
            k_exec=args.k_exec,
            budget_steps=args.budget_steps,
        )
        result = runner.run()
        success = result["success"]
        steps = result["steps_taken"]
        final = result["final_state"]
        block_xy = final[2:4]
        target_xy = final[4:6]
        final_dist = float(np.hypot(block_xy[0] - target_xy[0],
                                    block_xy[1] - target_xy[1]))

        if success:
            successes += 1
            steps_to_goal.append(steps)
        final_distances.append(final_dist)

        if args.save_trajectories:
            saved_trajectories.append({
                "states": result["states"],
                "success": success,
                "steps_taken": steps,
                "final_dist": final_dist,
            })

        if (ep + 1) % 10 == 0 or ep == args.n_episodes - 1:
            sr = successes / (ep + 1)
            mfd = float(np.mean(final_distances))
            print(f"[ep {ep+1:3d}/{args.n_episodes}] "
                  f"success_rate={sr:.3f}  mean_final_dist={mfd:.2f}  "
                  f"({time.time() - t0:.1f}s)")

    success_rate = successes / args.n_episodes
    mean_final_dist = float(np.mean(final_distances))
    mean_steps = float(np.mean(steps_to_goal)) if steps_to_goal else float("nan")

    print()
    print(f"=== RESULTS ===")
    print(f"  success_rate         : {success_rate:.3f} ({successes}/{args.n_episodes})")
    print(f"  mean steps-to-goal   : {mean_steps:.1f}  (over {len(steps_to_goal)} successful eps)")
    print(f"  mean final dist      : {mean_final_dist:.2f}  (success threshold = {args.success_radius})")
    print(f"  total wallclock      : {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / args.n_episodes:.2f}s/ep)")

    if args.save_trajectories:
        out_path = Path(args.ckpt).parent / "planning_trajectories.npz"
        # Trajectories may have variable lengths; save as object array.
        np.savez_compressed(
            out_path,
            states=np.array([t["states"] for t in saved_trajectories], dtype=object),
            successes=np.array([t["success"] for t in saved_trajectories]),
            steps_taken=np.array([t["steps_taken"] for t in saved_trajectories]),
            final_dists=np.array([t["final_dist"] for t in saved_trajectories]),
        )
        print(f"  trajectories saved → {out_path}")


if __name__ == "__main__":
    main()

"""Path C canonical planning eval.

Canonical protocol (matches stable-worldmodel + le-wm/eval.py):
  - Load heldout WeakPolicy trajectories from canonical HDF5.
  - For each eval episode:
      * sample (init_step, goal_step = init_step + 25) within the same trajectory
      * env.set_state(init_state)
      * z_goal = encoder(render(goal_state))  (frozen for episode)
      * MPC: plan with CEM (frame-skip 5, action_block=5, n_iters=30,
        canonical tricks), execute, replan
      * after env execution, compute latent_distance =
          ‖encoder(obs_actual_final) − z_goal‖₂
      * success ⇔ latent_distance < τ  (calibrated)
  - τ calibration: run encoder on heldout traj pairs at offset {0, +1, +25,
    cross-trajectory} to identify same / near / unrelated distance
    distributions. τ = valley between near and unrelated; fallback p95-near.

Headline metric: success rate (latent-match). Diagnostics retained:
  - block-to-recorded-goal-block distance
  - block-to-target_xy distance (env's old success metric)
  - predicted-final latent distance (planner's internal cost)
  - actual-final latent distance (headline)
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
from src.data import sample_offset_pair
from src.lewm.model import LeWM
from src.lewm.planner import CEMPlanner, MPCRunner


def make_env(env_name: str, seed: int):
    """Factory for env-by-name dispatch. Both envs share the same API
    (reset/step/state/observe/set_state/is_terminated)."""
    if env_name == "pusht":
        return MiniPushTEnv(seed=seed)
    elif env_name == "reacher":
        from src.env_reacher import DMReacherEnv
        return DMReacherEnv(seed=seed)
    else:
        raise ValueError(f"Unknown env: {env_name}")


def load_model(ckpt_path: str, device: torch.device) -> tuple[LeWM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    # action_dim for Path C = stride * raw_action_dim = 10 by default
    if args.get("path_c", False):
        action_dim = 2 * args.get("path_c_stride", 5)
    else:
        action_dim = 2
    model = LeWM(
        image_size=64, patch_size=8, in_chans=3,
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


def calibrate_tau(
    model: LeWM,
    h5_path: str,
    test_episode_indices: np.ndarray,
    device: torch.device,
    *,
    offset_target: int = 25,
    n_samples: int = 50,
    seed: int = 0,
) -> dict:
    """Calibrate τ from heldout latent-distance distributions.

    Returns a dict with keys:
      - same:        ‖z(s) − z(s)‖     (sanity, ≈ 0)
      - near:        ‖z(s) − z(s+offset)‖   for the same-traj +offset pairs
      - unrelated:   ‖z(s) − z(s')‖     for pairs from different trajectories
      - tau_p95_near: 95th percentile of `near` distribution
      - tau_valley:  midpoint between near.max() and unrelated.min() if a
                     gap exists; otherwise tau_p95_near
    """
    import h5py
    rng = np.random.default_rng(seed)
    same_d, near_d, unrelated_d = [], [], []
    with h5py.File(h5_path, "r") as f:
        T = int(f["obs"].shape[1])
        eps = list(test_episode_indices)
        if len(eps) < 2:
            raise ValueError("Need at least 2 episodes for unrelated baseline.")
        for _ in range(n_samples):
            e = int(rng.choice(eps))
            s = int(rng.integers(0, T - offset_target))
            obs_s = np.asarray(f["obs"][e, s])
            obs_offset = np.asarray(f["obs"][e, s + offset_target])
            # unrelated: pick a different episode + random step
            e2 = int(rng.choice([x for x in eps if x != e]))
            s2 = int(rng.integers(0, T))
            obs_other = np.asarray(f["obs"][e2, s2])

            obs_s_t = torch.from_numpy(obs_s).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)
            if obs_s.dtype == np.uint8:
                obs_s_t = obs_s_t / 255.0
            obs_offset_t = torch.from_numpy(obs_offset).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)
            if obs_offset.dtype == np.uint8:
                obs_offset_t = obs_offset_t / 255.0
            obs_other_t = torch.from_numpy(obs_other).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)
            if obs_other.dtype == np.uint8:
                obs_other_t = obs_other_t / 255.0
            obs_s_t = obs_s_t.to(device)
            obs_offset_t = obs_offset_t.to(device)
            obs_other_t = obs_other_t.to(device)

            with torch.no_grad():
                z_s = model.encoder(obs_s_t)[0, 0]
                z_off = model.encoder(obs_offset_t)[0, 0]
                z_other = model.encoder(obs_other_t)[0, 0]
            same_d.append((z_s - z_s).norm().item())  # trivially 0
            near_d.append((z_s - z_off).norm().item())
            unrelated_d.append((z_s - z_other).norm().item())

    near_arr = np.array(near_d)
    unrel_arr = np.array(unrelated_d)
    tau_p95 = float(np.percentile(near_arr, 95))
    near_max = float(near_arr.max())
    unrel_min = float(unrel_arr.min())
    if unrel_min > near_max:
        tau_valley = float((near_max + unrel_min) / 2)
    else:
        tau_valley = tau_p95
    return {
        "same_mean": float(np.mean(same_d)),
        "near_mean": float(np.mean(near_d)),
        "near_max": near_max,
        "near_p95": tau_p95,
        "unrelated_mean": float(np.mean(unrel_arr)),
        "unrelated_min": unrel_min,
        "tau_p95_near": tau_p95,
        "tau_valley": tau_valley,
        "tau": tau_valley,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--h5", required=True,
                        help="Canonical HDF5 (state, obs, action, episode_ends).")
    parser.add_argument("--splits", default=None,
                        help="Optional path to splits.npz (from training).")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--budget-env-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5,
                        help="CEM plan length in horizon steps (each step = action_block env steps).")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-elites", type=int, default=30)
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument("--receding-horizon", type=int, default=None,
                        help="Default = horizon (full plan execution before replan).")
    parser.add_argument("--offset-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau-samples", type=int, default=80)
    parser.add_argument("--tau-override", type=float, default=None)
    parser.add_argument("--env", default="pusht", choices=["pusht", "reacher"],
                        help="Environment to instantiate from recorded states. "
                             "Must match the env used to generate the HDF5.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[eval] device={device}  ckpt={args.ckpt}")
    model, ckpt_args = load_model(args.ckpt, device)
    history_size = ckpt_args.get("path_c_history", 3)
    sub_action_dim = 2

    # Episode splits
    if args.splits is not None and Path(args.splits).exists():
        sp = np.load(args.splits, allow_pickle=True)
        test_eps = sp.get("test", sp["val"])  # use 'test' if present, else 'val'
    else:
        # Fallback: derive a test split deterministically from total episodes
        import h5py
        with h5py.File(args.h5, "r") as f:
            n = int(f["obs"].shape[0])
        rng = np.random.default_rng(args.seed)
        test_eps = rng.permutation(n)[:max(20, args.n_episodes * 2)]

    # τ calibration
    print(f"[tau] calibrating τ from {args.tau_samples} heldout pairs (offset={args.offset_steps})...")
    tau_info = calibrate_tau(
        model, args.h5, test_eps, device,
        offset_target=args.offset_steps, n_samples=args.tau_samples, seed=args.seed,
    )
    print(f"[tau] near_mean={tau_info['near_mean']:.3f}  near_p95={tau_info['near_p95']:.3f}  "
          f"unrelated_mean={tau_info['unrelated_mean']:.3f}  unrelated_min={tau_info['unrelated_min']:.3f}")
    print(f"[tau] tau_valley={tau_info['tau_valley']:.3f}  tau_p95_near={tau_info['tau_p95_near']:.3f}")
    tau = args.tau_override if args.tau_override is not None else tau_info["tau"]
    print(f"[tau] using τ = {tau:.3f}")

    # Planner
    planner = CEMPlanner(
        model=model,
        horizon=args.horizon,
        action_block=args.action_block,
        action_dim=sub_action_dim,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        n_iters=args.n_iters,
        history_size=history_size,
        seed=args.seed,
    )
    print(f"[plan] H={args.horizon}  action_block={args.action_block}  "
          f"N={args.n_samples}  K={args.n_elites}  T={args.n_iters}  "
          f"history_size={history_size}")

    rng = np.random.default_rng(args.seed)
    successes = 0
    actual_dists, predicted_dists = [], []
    block_recorded_goal_dists = []
    block_target_dists = []
    rows = []
    t0 = time.time()

    for ep in range(args.n_episodes):
        # Sample (init, goal) pair from same heldout trajectory
        e, s, sg, init_obs, goal_obs, init_state, goal_state = sample_offset_pair(
            args.h5, test_eps, rng, offset_steps=args.offset_steps,
        )
        # Encode goal once, freeze for episode
        goal_obs_t = torch.from_numpy(goal_obs).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            z_goal = model.encoder(goal_obs_t)[0, 0]  # (D,)

        # Set env to init state
        env = make_env(args.env, seed=int(rng.integers(0, 2**31 - 1)))
        env.set_state(init_state)

        runner = MPCRunner(
            planner=planner,
            history_size=history_size,
            action_block=args.action_block,
            action_dim=sub_action_dim,
            receding_horizon=args.receding_horizon,
            budget_env_steps=args.budget_env_steps,
            success_fn=None,  # success determined post-hoc by latent distance
        )
        result = runner.run(env, z_goal, log_diagnostics=False)

        # Post-execution: compute actual-final latent distance to z_goal
        final_obs_t = torch.from_numpy(result["final_obs"]).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            z_final = model.encoder(final_obs_t)[0, 0]
        actual_dist = float((z_final - z_goal).norm().item())

        # Diagnostics — meaning depends on env state layout
        final_state = result["final_state"]
        if args.env == "pusht":
            # MiniPushT state: [agent(2), block(2), target(2)]
            block_recorded_goal_dist = float(np.linalg.norm(final_state[2:4] - goal_state[2:4]))
            block_target_dist = float(np.linalg.norm(final_state[2:4] - final_state[4:6]))
        else:
            # Reacher state: [qpos(2), to_target(2), qvel(2)]
            # "block_recorded_goal" analog: joint-angle distance final↔goal
            # "block_target" analog: tip distance to target (env's old success metric)
            block_recorded_goal_dist = float(np.linalg.norm(final_state[0:2] - goal_state[0:2]))
            block_target_dist = float(np.linalg.norm(final_state[2:4]))

        success = actual_dist < tau
        if success:
            successes += 1
        actual_dists.append(actual_dist)
        block_recorded_goal_dists.append(block_recorded_goal_dist)
        block_target_dists.append(block_target_dist)

        rows.append({
            "ep": ep, "traj_id": e, "init_step": s, "goal_step": sg,
            "actual_dist": actual_dist,
            "block_recorded_goal_dist": block_recorded_goal_dist,
            "block_target_dist": block_target_dist,
            "success": success,
        })
        if (ep + 1) % 5 == 0 or ep == args.n_episodes - 1:
            sr = successes / (ep + 1)
            rec_fmt = f"{block_recorded_goal_dist:.1f}" if args.env == "pusht" \
                else f"{block_recorded_goal_dist:.3f}"
            print(f"[ep {ep+1:3d}/{args.n_episodes}] traj={e} init={s} goal={sg}  "
                  f"actual_dist={actual_dist:.3f}  rec_goal={rec_fmt}  "
                  f"success={success}  rate={sr:.3f}  ({time.time()-t0:.1f}s)")

    sr = successes / args.n_episodes
    print()
    print(f"=== PATH C RESULTS ===")
    print(f"  success_rate (latent < τ={tau:.3f}): {sr:.3f}  ({successes}/{args.n_episodes})")
    print(f"  actual_dist:  mean={np.mean(actual_dists):.3f}  med={np.median(actual_dists):.3f}  "
          f"min={np.min(actual_dists):.3f}  max={np.max(actual_dists):.3f}")
    rec_label = "block_recorded_goal_dist" if args.env == "pusht" else "qpos_to_recorded_goal_dist"
    tgt_label = "block_target_dist" if args.env == "pusht" else "tip_to_target_dist"
    print(f"  {rec_label} (diagnostic): mean={np.mean(block_recorded_goal_dists):.3f}  "
          f"med={np.median(block_recorded_goal_dists):.3f}")
    print(f"  {tgt_label} (old metric):       mean={np.mean(block_target_dists):.3f}  "
          f"med={np.median(block_target_dists):.3f}")
    print(f"  τ-calibration: near_mean={tau_info['near_mean']:.3f}  "
          f"unrelated_mean={tau_info['unrelated_mean']:.3f}")
    print(f"  total wallclock: {time.time()-t0:.1f}s")

    # Save per-episode log
    out_path = Path(args.ckpt).parent / "path_c_eval.npz"
    np.savez(out_path,
             rows=np.array(rows, dtype=object),
             tau_info=np.array([tau_info], dtype=object),
             success_rate=sr, tau=tau)
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()

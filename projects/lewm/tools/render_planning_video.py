"""Record planning episodes to MP4 with overlays.

For each eval episode:
  - Run the trained model + CEM-MPC planner to interact with MiniPushT.
  - Render every step's frame.
  - Overlay (per frame): a yellow box showing the planned next agent position
    (according to the executed plan), and a red dashed box showing the goal
    target. Caption (encoded as a colored bar) shows episode index and success.

Usage:
    python tools/render_planning_video.py \\
        --ckpt runs/v3-pusht/ckpt_epoch9.pt \\
        --out runs/planning_demo.mp4 \\
        --n-episodes 10 --budget-steps 50 --fps 10 --upscale 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v3 as iio

from src.env_pusht import MiniPushTEnv
from src.lewm.model import LeWM
from src.lewm.planner import CEMPlanner


def load_model(ckpt_path: str, device: torch.device) -> tuple[LeWM, int]:
    """Reconstruct the model with architecture sizes read from the checkpoint's args."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    sub_len = args.get("sub_len", 4)
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
        action_dim=2, max_history=sub_len,
        sigreg_num_proj=args.get("num_projections", 1024),
        sigreg_knots=args.get("sigreg_knots", 17),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sub_len


def upscale(img: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def draw_box(img: np.ndarray, cx: float, cy: float, half_w: float, half_h: float,
             color: tuple, *, dashed: bool = False, upscale_factor: int = 1):
    """Draw a 1-px-thick box centered at (cx, cy) with given half-extents.

    All coordinates and extents are in PRE-upscale units; we scale internally.
    """
    H, W, _ = img.shape
    cx *= upscale_factor; cy *= upscale_factor
    hw = half_w * upscale_factor; hh = half_h * upscale_factor
    x0, x1 = int(round(cx - hw)), int(round(cx + hw))
    y0, y1 = int(round(cy - hh)), int(round(cy + hh))
    color_arr = np.array(color, dtype=np.uint8)

    def setpix(x, y):
        if 0 <= x < W and 0 <= y < H:
            img[y, x, :] = color_arr

    # Top + bottom horizontal lines
    for x in range(x0, x1 + 1):
        if not dashed or (x // 2) % 2 == 0:
            setpix(x, y0); setpix(x, y1)
    # Left + right vertical lines
    for y in range(y0, y1 + 1):
        if not dashed or (y // 2) % 2 == 0:
            setpix(x0, y); setpix(x1, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--budget-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-elites", type=int, default=30)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--success-radius", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gap-frames", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, sub_len = load_model(args.ckpt, device)
    planner = CEMPlanner(
        model=model, horizon=args.horizon, n_samples=args.n_samples,
        n_elites=args.n_elites, n_iters=args.n_iters,
        action_dim=2, sub_len=sub_len,
    )

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = W = 64
    Hu = H * args.upscale
    Wu = W * args.upscale
    gap = np.zeros((Hu, Wu, 3), dtype=np.uint8)
    frames: list[np.ndarray] = []

    successes = 0
    for ep in range(args.n_episodes):
        env_seed = int(rng.integers(0, 2**31 - 1))
        env = MiniPushTEnv(seed=env_seed, success_radius=args.success_radius)
        obs_history = [env.observe()] * sub_len
        states = [env.state()]
        steps_taken = 0
        success = False
        cur_plan: np.ndarray | None = None
        plan_step = 0

        while steps_taken < args.budget_steps and not env.is_success():
            # Plan if needed
            if cur_plan is None or plan_step >= args.horizon:
                ctx_imgs = np.stack(obs_history[-sub_len:])
                ctx_imgs = np.transpose(ctx_imgs, (0, 3, 1, 2))
                ctx = torch.from_numpy(ctx_imgs).float()
                goal_img = env.goal_observation()
                goal = torch.from_numpy(goal_img).permute(2, 0, 1).float()
                cur_plan = planner.plan(ctx, goal).cpu().numpy()
                plan_step = 0

            # Render current frame with overlays
            cur_obs = env.observe()
            frame_u8 = (cur_obs * 255).astype(np.uint8)
            up = upscale(frame_u8, args.upscale).copy()

            # Goal target box (red dashed) — on the target T's bounding box
            tx, ty = env.target_x, env.target_y
            draw_box(up, tx, ty, 6.0, 6.0, color=(220, 30, 30),
                     dashed=True, upscale_factor=args.upscale)

            # Planned next agent position box (yellow) — agent + plan[plan_step]
            ax, ay = env.agent_x, env.agent_y
            next_a = cur_plan[plan_step]
            nx = float(np.clip(ax + next_a[0] * env.action_scale, 0, W))
            ny = float(np.clip(ay + next_a[1] * env.action_scale, 0, H))
            draw_box(up, nx, ny, 4.0, 4.0, color=(255, 220, 0),
                     dashed=False, upscale_factor=args.upscale)

            frames.append(up)

            # Step the env using the next planned action
            env.step(next_a)
            obs_history.append(env.observe())
            states.append(env.state())
            steps_taken += 1
            plan_step += 1

            if env.is_success():
                success = True

        # Final frame for the episode
        final_obs = env.observe()
        frame_u8 = (final_obs * 255).astype(np.uint8)
        up = upscale(frame_u8, args.upscale).copy()
        # Mark success/failure with a colored top bar
        bar_h = 6 * args.upscale
        if success:
            up[:bar_h, :, :] = [40, 200, 60]   # green
        else:
            up[:bar_h, :, :] = [200, 40, 40]   # red
        # Repeat a few frames so the result stays on screen
        for _ in range(5):
            frames.append(up)
        # Episode separator
        for _ in range(args.gap_frames):
            frames.append(gap)

        if success:
            successes += 1
        print(f"[ep {ep+1:2d}/{args.n_episodes}] success={success} "
              f"steps={steps_taken} block→target={env.block_to_target_distance():.2f}")

    iio.imwrite(out_path, frames, fps=args.fps, codec="libx264",
                plugin="FFMPEG", macro_block_size=8)
    sr = successes / args.n_episodes
    print(f"\n[done] success_rate={sr:.3f} ({successes}/{args.n_episodes})  → {out_path}")


if __name__ == "__main__":
    main()

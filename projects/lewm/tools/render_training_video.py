"""Render the cached training data stream to an MP4 video.

For visual inspection: concatenates the first N episodes of a cached
trajectory dataset (npz format) into a single video, with episode-boundary
captions and per-frame action overlays.

Usage:
    python tools/render_training_video.py \\
        --npz data/pusht_n500_T30_seed42.npz \\
        --out runs/training_data.mp4 \\
        --n-episodes 30 --fps 10 --upscale 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import imageio.v3 as iio


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               color: tuple[int, int, int], thickness: int = 1) -> None:
    """Draw a `thickness`-px wide line from (x0,y0) to (x1,y1) on `img`, in-place."""
    H, W, _ = img.shape
    dx = x1 - x0
    dy = y1 - y0
    n = max(abs(dx), abs(dy), 1)
    color_arr = np.array(color, dtype=np.uint8)
    for s in range(n + 1):
        x = x0 + dx * s // n
        y = y0 + dy * s // n
        for ox in range(-thickness // 2, thickness // 2 + 1):
            for oy in range(-thickness // 2, thickness // 2 + 1):
                xx, yy = x + ox, y + oy
                if 0 <= xx < W and 0 <= yy < H:
                    img[yy, xx, :] = color_arr


def _annotate(
    frame: np.ndarray,
    *,
    action: np.ndarray | None,
    agent_xy: tuple[float, float] | None,
    upscale: int,
) -> np.ndarray:
    """Frame annotation: action arrow drawn FROM the agent's position.

    `agent_xy` are pre-upscale env coordinates (in [0, frame_size]).
    """
    H, W, _ = frame.shape
    img = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame.copy()

    # Upscale via nearest-neighbor first so annotations are crisp at higher res.
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)

    if action is not None and agent_xy is not None:
        # Arrow length: action × 10 px in env units (so a unit-magnitude action
        # extends 10 env-px from the agent). Clip to [-1, 1].
        ax = float(np.clip(action[0], -1.0, 1.0))
        ay = float(np.clip(action[1], -1.0, 1.0))
        agent_x, agent_y = agent_xy
        # Pre-upscale → upscale coordinates
        x0 = int(round(agent_x * upscale))
        y0 = int(round(agent_y * upscale))
        x1 = int(round((agent_x + ax * 10) * upscale))
        y1 = int(round((agent_y + ay * 10) * upscale))
        # Yellow shaft
        _draw_line(img, x0, y0, x1, y1, color=(255, 220, 0), thickness=2)
        # Small white dot at the arrow tip
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                xx, yy = x1 + ox, y1 + oy
                if 0 <= xx < img.shape[1] and 0 <= yy < img.shape[0]:
                    img[yy, xx, :] = [255, 255, 255]

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True,
                        help="Path to cached dataset npz (with obs, actions, states).")
    parser.add_argument("--out", required=True, help="Output mp4 path.")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--upscale", type=int, default=4,
                        help="Nearest-neighbor upscale factor (4 → 64×64 → 256×256).")
    parser.add_argument("--annotate-actions", action="store_true",
                        help="Overlay yellow arrow indicating per-frame action direction.")
    parser.add_argument("--gap-frames", type=int, default=2,
                        help="Black frames between episodes for visual separation.")
    args = parser.parse_args()

    cache = np.load(args.npz)
    obs = cache["obs"]   # (E, T, H, W, C)  float32 [0,1]
    actions = cache["actions"]  # (E, T, A)
    states = cache["states"] if "states" in cache.files else None  # (E, T, S)
    E_total, T, H, W, C = obs.shape
    n_eps = min(args.n_episodes, E_total)
    if states is not None and states.shape[-1] >= 2:
        print(f"[render] using states[:, :, 0:2] as agent positions for action arrows")
    elif args.annotate_actions:
        print(f"[render] WARN: states absent — action arrows skipped")

    print(f"[render] reading {args.npz}: {E_total} episodes × {T} frames × {H}×{W}×{C}")
    print(f"[render] writing {n_eps} episodes → {args.out} @ {args.fps} fps, upscale {args.upscale}×")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # imageio v3 writer — let it use ffmpeg via imageio-ffmpeg
    upscaled_h = H * args.upscale
    upscaled_w = W * args.upscale
    gap = np.zeros((upscaled_h, upscaled_w, 3), dtype=np.uint8)

    # Build frame stream then write at end (imageio-ffmpeg expects iterable).
    frames = []
    for ep in range(n_eps):
        for t in range(T):
            a = actions[ep, t] if args.annotate_actions else None
            agent_xy = None
            if args.annotate_actions and states is not None and states.shape[-1] >= 2:
                agent_xy = (float(states[ep, t, 0]), float(states[ep, t, 1]))
            frame = _annotate(obs[ep, t], action=a, agent_xy=agent_xy,
                              upscale=args.upscale)
            frames.append(frame)
        for _ in range(args.gap_frames):
            frames.append(gap)

    iio.imwrite(out_path, frames, fps=args.fps, codec="libx264",
                plugin="FFMPEG", macro_block_size=8)

    print(f"[render] done. duration ≈ {n_eps * (T + args.gap_frames) / args.fps:.1f}s")


if __name__ == "__main__":
    main()

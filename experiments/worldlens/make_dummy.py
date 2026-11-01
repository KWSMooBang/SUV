"""Write synthetic scene dumps in the SUV raw format (pipeline smoke without a checkpoint).

GT  = a textured road-like scene scrolling toward the camera.
pred = the same scene with a drifting offset + noise (stands in for SUV output).
Shapes/frame counts match the real rollout: 384x640, 4 history, 5 condition
(history_4 + repeat_first), 8 future frames at 2 Hz.
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.worldlens import layout


def _frame(h, w, t, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    base[: h // 2] = (0.55, 0.7, 0.9)  # sky
    road = yy >= h // 2
    stripes = (np.sin((yy - 20.0 * t - shift) / 9.0) > 0.6) & (np.abs(xx - w / 2) < 6)
    base[road] = (0.35, 0.35, 0.37)
    base[stripes & road] = (0.95, 0.95, 0.9)
    cx = w / 2 + 120 * np.sin(0.3 * t + seed) + shift
    car = (np.abs(xx - cx) < 40 + 4 * t) & (np.abs(yy - (h * 0.62 + 3 * t)) < 22 + 2 * t)
    base[car] = rng.uniform(0.1, 0.9, 3)
    return (base * 255).astype(np.uint8)


def make(raw_root, n_scenes=6, h=384, w=640, n_hist=4, n_fut=8, fps=2.0, noise=12.0):
    for s in range(n_scenes):
        token = f"dummy{s:04d}" + "0" * 7
        gt = np.stack([_frame(h, w, t, s) for t in range(n_hist + n_fut)])
        n_cond = n_hist + 1
        cond = np.concatenate([gt[:1], gt[:n_hist]])
        rng = np.random.default_rng(100 + s)
        fut = np.stack([_frame(h, w, n_hist + k, s, shift=3.0 * k) for k in range(n_fut)]).astype(np.float32)
        fut = np.clip(fut + rng.normal(0, noise, fut.shape), 0, 255).astype(np.uint8)
        rgb = np.concatenate([cond, fut])
        depth = np.repeat(np.linspace(0, 255, h, dtype=np.float32)[None, :, None, None], rgb.shape[0], 0)
        depth = np.broadcast_to(depth, rgb.shape[:3] + (1,)).repeat(3, -1).astype(np.uint8)
        seg = (rgb // 64 * 64).astype(np.uint8)
        traj = np.stack([np.linspace(0.5, 4, n_fut) * 2, np.zeros(n_fut), np.zeros(n_fut)], -1)
        meta = {
            "token": token,
            "log_name": "dummy_log",
            "dataset": "dummy",
            "fps": fps,
            "num_history_frames": n_hist,
            "num_future_frames": n_fut,
            "num_condition_frames": n_cond,
            "video_hw": [h, w],
            "driving_command": [0, 1, 0, 0],
            "pred_trajectory": traj,
            "gt_trajectory": traj,
        }
        layout.save_scene_dump(raw_root, token, {"rgb": rgb, "depth": depth, "seg": seg, "instance": seg}, gt, meta)
    print(f"wrote {n_scenes} dummy scenes to {raw_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--n-scenes", type=int, default=6)
    a = p.parse_args()
    make(a.raw_root, a.n_scenes)

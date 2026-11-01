"""Export SUV raw scene dumps into the directory layout WorldLens reads.

Runs in either env (numpy + imageio-ffmpeg only).

    python -m experiments.worldlens.export_worldlens \
        --raw-root  $OUT/raw \
        --wl-root   $OUT/wl \
        --method    suvnavsim \
        --clip      future            # or: history_future

Per scene this writes
    generated_results/<method>/video_submission/<token>_gen0/CAM_FRONT.mp4   (SUV RGB)
    generated_results/gt/video_submission/<token>_gen0/CAM_FRONT.mp4         (NAVSIM GT)
and, with --aux-streams, SUV depth/seg/instance under sibling methods
``<method>_depth`` etc. (not read by stock WorldLens; kept for custom metrics).

Clip modes
  future          8 predicted future frames (t+0.5s .. t+4.0s, 2 Hz) vs the 8 GT
                  future frames.  This is the honest "generation" clip.
  history_future  4 real history frames + 8 future frames for *both* sides
                  (history taken from GT for both, so only the future differs).
                  Gives 12-frame clips; useful for temporal metrics that need
                  more frames and for I3D (FVD).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.worldlens import layout


def export(raw_root, wl_root, method, clip="future", gen_idx=0, aux_streams=False, crf=10, overwrite=False):
    scene_dirs = layout.list_scene_dumps(raw_root)
    if not scene_dirs:
        raise FileNotFoundError(f"no scene dumps under {raw_root}")
    tokens = [d.name for d in scene_dirs]
    layout.check_method_name(wl_root, method, tokens)

    index = []
    for scene_dir in scene_dirs:
        arrays, meta = layout.load_scene_dump(scene_dir)
        token = meta["token"]
        fps = float(meta["fps"])
        gt = arrays["gt_rgb"]
        gt_future = gt[layout.future_slice(meta, "gt_rgb")]
        gt_hist = gt[layout.history_slice_gt(meta)]

        outputs = {}
        for stream in layout.STREAMS if aux_streams else ("rgb",):
            key = f"pred_{stream}"
            if key not in arrays:
                continue
            pred_future = arrays[key][layout.future_slice(meta, key)]
            if pred_future.shape[0] != gt_future.shape[0]:
                raise ValueError(
                    f"{token}: pred {key} has {pred_future.shape[0]} future frames, GT has {gt_future.shape[0]}"
                )
            if clip == "future":
                clip_pred = pred_future
            elif clip == "history_future":
                # history is always the real frames (RGB only makes sense for rgb)
                hist = gt_hist if stream == "rgb" else arrays[key][: int(meta["num_condition_frames"])][-gt_hist.shape[0]:]
                clip_pred = np.concatenate([hist, pred_future], axis=0)
            else:
                raise ValueError(f"unknown clip mode {clip!r}")
            m = method if stream == "rgb" else f"{method}_{stream}"
            outputs[m] = clip_pred

        gt_clip = gt_future if clip == "future" else np.concatenate([gt_hist, gt_future], axis=0)
        outputs["gt"] = gt_clip

        for m, video in outputs.items():
            path = layout.scene_clip_path(wl_root, m, token, gen_idx)
            if path.exists() and not overwrite:
                continue
            layout.write_mp4(path, video, fps=fps, crf=crf)

        index.append(
            {
                "token": token,
                "log_name": meta.get("log_name"),
                "num_frames": int(gt_clip.shape[0]),
                "hw": list(gt_clip.shape[1:3]),
                "fps": fps,
                "driving_command": meta.get("driving_command"),
            }
        )

    out_index = Path(wl_root) / "generated_results" / method / "suv_export_index.json"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with open(out_index, "w") as f:
        json.dump({"method": method, "clip": clip, "gen_idx": gen_idx, "scenes": index}, f, indent=2)
    print(f"exported {len(index)} scenes -> {layout.video_submission_dir(wl_root, method)} (clip={clip})")
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--wl-root", required=True)
    p.add_argument("--method", default="suvnavsim")
    p.add_argument("--clip", choices=("future", "history_future"), default="future")
    p.add_argument("--gen-idx", type=int, default=0)
    p.add_argument("--aux-streams", action="store_true")
    p.add_argument("--crf", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    export(a.raw_root, a.wl_root, a.method, a.clip, a.gen_idx, a.aux_streams, a.crf, a.overwrite)


if __name__ == "__main__":
    main()

"""On-disk formats shared by the SUV rollout side and the WorldLens side.

This module must stay importable from both environments:
  * the SUV env (python 3.10, torch 2.7) that runs ``suv_rollout.py``;
  * the WorldLens env (python 3.9, torch 2.1) that runs ``run_metrics.py``.
So: numpy + imageio only, no ``X | Y`` annotations evaluated at runtime.

Raw scene dump (written by ``suv_rollout.py`` / ``make_dummy.py``)::

    <raw_root>/<token>/
        streams.npz   uint8 arrays, each [T, H, W, 3]
                        pred_rgb, pred_depth, pred_seg, pred_instance
                            T = num_condition_frames + num_future_frames
                            (SUV decodes the condition prefix too; slicing is
                            done at export time using meta.json)
                        gt_rgb     [num_history + num_future, H, W, 3]
                            real NAVSIM cam_f0 frames, same crop/resize as the
                            SUV feature builder (crop 28 px top/bottom -> 384x640)
        meta.json     token, log_name, fps, frame counts, ego status,
                      driving command, predicted / GT trajectories, sampler cfg

WorldLens tree (written by ``export_worldlens.py``), matching what
``worldbench.videogen.generation.*`` reads::

    <wl_root>/generated_results/<method>/video_submission/<token>_gen<k>/CAM_FRONT.mp4
    <wl_root>/generated_results/gt/video_submission/<token>_gen<k>/CAM_FRONT.mp4

WorldLens locates the GT clip with ``video_path.replace(method_name, "gt")``, so
the method name must not occur anywhere else in the path (checked in
``check_method_name``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

STREAMS = ("rgb", "depth", "seg", "instance")
CAM_NAME = "CAM_FRONT"
META_VERSION = 1


# --------------------------------------------------------------------------- raw dumps
def save_scene_dump(
    raw_root: os.PathLike,
    token: str,
    pred_streams: Dict[str, np.ndarray],
    gt_rgb: np.ndarray,
    meta: dict,
) -> Path:
    """Write one scene. ``pred_streams`` keys are names from STREAMS."""
    scene_dir = Path(raw_root) / token
    scene_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, video in pred_streams.items():
        if name not in STREAMS:
            raise ValueError(f"unknown stream {name!r}; expected one of {STREAMS}")
        arrays[f"pred_{name}"] = _as_uint8_thwc(video, f"pred_{name}")
    arrays["gt_rgb"] = _as_uint8_thwc(gt_rgb, "gt_rgb")
    tmp = scene_dir / "streams.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, scene_dir / "streams.npz")

    meta = dict(meta)
    meta.setdefault("token", token)
    meta["meta_version"] = META_VERSION
    meta["streams"] = sorted(arrays)
    meta["shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    return scene_dir


def load_scene_dump(scene_dir: os.PathLike):
    scene_dir = Path(scene_dir)
    with open(scene_dir / "meta.json") as f:
        meta = json.load(f)
    with np.load(scene_dir / "streams.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, meta


def list_scene_dumps(raw_root: os.PathLike) -> List[Path]:
    root = Path(raw_root)
    return sorted(p.parent for p in root.glob("*/streams.npz"))


def future_slice(meta: dict, key: str) -> slice:
    """Index range of the future frames inside ``key`` (pred_* or gt_rgb)."""
    n_future = int(meta["num_future_frames"])
    if key.startswith("pred_"):
        start = int(meta["num_condition_frames"])
    elif key == "gt_rgb":
        start = int(meta["num_history_frames"])
    else:
        raise KeyError(key)
    return slice(start, start + n_future)


def history_slice_gt(meta: dict) -> slice:
    return slice(0, int(meta["num_history_frames"]))


# --------------------------------------------------------------------------- WorldLens tree
def video_submission_dir(wl_root: os.PathLike, method: str) -> Path:
    return Path(wl_root) / "generated_results" / method / "video_submission"


def scene_clip_path(wl_root: os.PathLike, method: str, token: str, gen_idx: int = 0, cam: str = CAM_NAME) -> Path:
    return video_submission_dir(wl_root, method) / f"{token}_gen{gen_idx}" / f"{cam}.mp4"


def check_method_name(wl_root: os.PathLike, method: str, tokens: Iterable[str] = ()) -> None:
    """WorldLens swaps ``method`` -> ``gt`` by plain string replace on the full path."""
    if method == "gt":
        raise ValueError("method name 'gt' is reserved for the ground-truth tree")
    root = str(Path(wl_root).resolve())
    if method in root:
        raise ValueError(
            f"method name {method!r} occurs in wl_root {root!r}; WorldLens would map the GT path wrongly"
        )
    for token in tokens:
        if method in token:
            raise ValueError(f"method name {method!r} occurs in scene token {token!r}")


def write_mp4(path: os.PathLike, frames_thwc: np.ndarray, fps: float, crf: int = 10) -> None:
    import imageio.v2 as imageio  # lazy: imageio-ffmpeg only needed here

    frames = _as_uint8_thwc(frames_thwc, str(path))
    h, w = frames.shape[1:3]
    if h % 2 or w % 2:
        raise ValueError(f"H and W must be even for yuv420p, got {h}x{w}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", str(int(crf))],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def read_mp4(path: os.PathLike) -> np.ndarray:
    import imageio.v2 as imageio

    return np.stack(list(imageio.mimread(str(path), memtest=False)), axis=0)


# --------------------------------------------------------------------------- helpers
def _as_uint8_thwc(video, name: str) -> np.ndarray:
    arr = np.asarray(video)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name}: expected [T, H, W, 3], got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def pil_frames_to_array(frames) -> np.ndarray:
    """SUV ``_decode_latents`` returns a list of PIL images -> [T, H, W, 3] uint8."""
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames], axis=0)


def maybe_int(x) -> Optional[int]:
    return None if x is None else int(x)

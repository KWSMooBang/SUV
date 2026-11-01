"""Run SUV on NAVSIM scenes and dump future RGB / depth / seg / instance + trajectories.

Runs in the SUV env (same one as ``experiments/navsimv1``: NAVSIM devkit,
torch 2.7, SUV checkpoint, cached text embeddings).  It does not modify the
NAVSIM agent: ``SUVWorldLensRollout`` subclasses ``SUVNavsimV1Agent`` and only
adds a ``rollout`` method that calls ``infer_action_slot(..., return_slot_videos=True)``.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO/src:$REPO \
    python -m experiments.worldlens.suv_rollout \
        --checkpoint  $CKPT/suv_navsim.pt \
        --text-embedding-cache-dir $TEXT_EMB/v1 \
        --navsim-log-path $OPENSCENE_DATA_ROOT/navsim_logs/test \
        --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/test \
        --raw-root $OUT/raw --max-scenes 8

Output format: see ``experiments/worldlens/layout.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from experiments.navsimv1.pdm_agent import SUVNavsimV1Agent, _build_history_condition_video
from experiments.navsimv1.data.features import _resize_rgb_to_tensor
from experiments.worldlens import layout

logger = logging.getLogger("suv_worldlens_rollout")


class SUVWorldLensRollout(SUVNavsimV1Agent):
    """SUV NAVSIM v1 agent that also returns the decoded slot videos."""

    @torch.no_grad()
    def rollout(self, agent_input) -> Dict[str, Any]:
        self.initialize()
        assert self.model is not None
        features = self.feature_builder.compute_features(agent_input)
        history_video = torch.as_tensor(features["video_front"], dtype=torch.float32)
        ego_status = torch.as_tensor(features["ego_status"], dtype=torch.float32)
        condition_video = _build_history_condition_video(
            history_video,
            frame_mode=self.video_frame_mode,
            padding_mode=self.history_padding_mode,
        )
        input_image = condition_video.mul(2.0).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        slot_names, slot_contexts, slot_context_masks = self._build_slot_text_contexts(ego_status)
        pred = self.model.infer_action_slot(
            input_image=input_image,
            action_horizon=self.num_future_frames,
            num_video_frames=int(condition_video.shape[0]) + self.num_future_frames,
            proprio=ego_status[-1],
            slot_contexts=slot_contexts,
            slot_context_masks=slot_context_masks,
            slot_names=slot_names,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device=self.rand_device,
            tiled=self.tiled,
            return_slot_videos=True,
        )
        if "slot_video_error" in pred:
            raise RuntimeError(f"SUV slot video decode failed: {pred['slot_video_error']}")
        streams = {name: layout.pil_frames_to_array(frames) for name, frames in pred["slot_videos"].items()}
        return {
            "action": pred["action"].numpy()[:, :3].astype(np.float32),
            "streams": streams,
            "num_condition_frames": int(condition_video.shape[0]),
            "ego_status": ego_status.numpy(),
        }

    def gt_front_frames(self, scene) -> np.ndarray:
        """All cam_f0 frames of the scene (history + future), preprocessed like the model input."""
        cfg = self.feature_config
        frames = []
        for frame in scene.frames:
            image = frame.cameras.cam_f0.image
            if image is None:
                raise RuntimeError("scene frame without cam_f0 image; load the scene with front sensors")
            t = _resize_rgb_to_tensor(image, cfg.video_size, cfg.crop_top_bottom)
            frames.append((t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))
        return np.stack(frames, axis=0)


def front_only_sensor_config(SensorConfig):
    """cam_f0 for every frame (history + future). Official navsim v1 has no build_front_only_sensors."""
    if hasattr(SensorConfig, "build_front_only_sensors"):
        return SensorConfig.build_front_only_sensors(include=True)
    return SensorConfig(
        cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )


def _poses(trajectory) -> np.ndarray:
    return np.asarray(trajectory.poses, dtype=np.float32)[:, :3]


def run(args: argparse.Namespace) -> None:
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    agent = SUVWorldLensRollout(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config_path,
        text_embedding_cache_dir=args.text_embedding_cache_dir,
        visual_conditioning="history_4",
        slot_inference=True,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    n_hist, n_fut = agent.num_history_frames, agent.num_future_frames

    tokens = None
    if args.tokens_file:
        tokens = [t.strip() for t in Path(args.tokens_file).read_text().splitlines() if t.strip()]
    log_names = args.log_names or None
    if log_names is None:
        # only logs whose camera blobs are on disk (partial downloads otherwise crash mid-run)
        blob_logs = {p.name for p in Path(args.sensor_blobs_path).iterdir() if p.is_dir()}
        all_logs = {p.stem for p in Path(args.navsim_log_path).glob("*.pkl")}
        log_names = sorted(all_logs & blob_logs)
        if len(log_names) < len(all_logs):
            logger.warning("sensor blobs present for %d/%d logs; restricting to those", len(log_names), len(all_logs))
    scene_filter = SceneFilter(
        num_history_frames=n_hist,
        num_future_frames=n_fut,
        has_route=True,
        log_names=log_names,
        tokens=tokens,
    )
    loader = SceneLoader(
        sensor_blobs_path=Path(args.sensor_blobs_path),
        data_path=Path(args.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=front_only_sensor_config(SensorConfig),
    )
    all_tokens = sorted(loader.tokens)
    if args.stride > 1:
        all_tokens = all_tokens[:: args.stride]
    if args.max_scenes > 0:
        all_tokens = all_tokens[: args.max_scenes]
    logger.info("rolling out %d scenes", len(all_tokens))

    agent.initialize()
    raw_root = Path(args.raw_root)
    timings = []
    for i, token in enumerate(all_tokens):
        if (raw_root / token / "streams.npz").exists() and not args.overwrite:
            continue
        t0 = time.time()
        agent_input = loader.get_agent_input_from_token(token)
        scene = loader.get_scene_from_token(token)
        out = agent.rollout(agent_input)
        gt_rgb = agent.gt_front_frames(scene)
        gt_traj = _poses(scene.get_future_trajectory(num_trajectory_frames=n_fut))
        last_ego = agent_input.ego_statuses[-1]
        meta = {
            "token": token,
            "log_name": loader.scene_frames_dicts[token][0]["log_name"],
            "dataset": "navsim_v1",
            "split_logs": str(args.navsim_log_path),
            "fps": float(agent.fps),
            "num_history_frames": n_hist,
            "num_future_frames": n_fut,
            "num_condition_frames": out["num_condition_frames"],
            "history_padding_mode": agent.history_padding_mode,
            "video_hw": list(agent.feature_config.video_size),
            "crop_top_bottom": agent.feature_config.crop_top_bottom,
            "driving_command": np.asarray(last_ego.driving_command).tolist(),
            "ego_status_history": out["ego_status"],
            "pred_trajectory": out["action"],
            "gt_trajectory": gt_traj,
            "ade_m": float(np.linalg.norm(out["action"][:, :2] - gt_traj[:, :2], axis=-1).mean()),
            "sampler": {"num_inference_steps": agent.num_inference_steps, "seed": agent.seed},
            "checkpoint": str(args.checkpoint),
        }
        layout.save_scene_dump(raw_root, token, out["streams"], gt_rgb, meta)
        timings.append(time.time() - t0)
        logger.info("[%d/%d] %s  %.1fs  ADE=%.2fm", i + 1, len(all_tokens), token, timings[-1], meta["ade_m"])

    if timings:
        summary = {"n": len(timings), "sec_per_scene_mean": float(np.mean(timings)), "sec_total": float(np.sum(timings))}
        (raw_root / "rollout_timing.json").write_text(json.dumps(summary, indent=2))
        logger.info("timing %s", summary)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config-path", default=None)
    p.add_argument("--text-embedding-cache-dir", required=True)
    p.add_argument("--navsim-log-path", required=True)
    p.add_argument("--sensor-blobs-path", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--tokens-file", default=None, help="one scene token per line")
    p.add_argument("--log-names", nargs="*", default=None)
    p.add_argument("--max-scenes", type=int, default=0, help="0 = all")
    p.add_argument("--stride", type=int, default=1, help="take every k-th token (sorted)")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())

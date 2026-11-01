"""Run WorldLens single-view Generation metrics on an exported SUV tree.

Runs in the WorldLens env (py3.9, torch 2.1+cu118).  Needs only the
Generation-metric deps (no mmcv / mmdet3d / bevfusion).

    CUDA_VISIBLE_DEVICES=0 \
    python experiments/worldlens/run_metrics.py \
        --wl-root $OUT/wl --method suvnavsim \
        --worldlens-root /workspace/SUV-deps/worldlens \
        --pretrained-root /data-work/SUV/worldlens/pretrained_models \
        --metrics fvd temporal_consistency subject_consistency

Metrics (protocol (a): SUV future frames vs NAVSIM GT future frames, CAM_FRONT only)
  temporal_consistency  stock WorldLens TEMPORAL_CONSISTENCY (CLIP ViT-B/32): ACM, TJI score, TS
  subject_consistency   stock WorldLens SUBJECT_CONSISTENCY (DINO ViT-B/16), but DINO weights
                        are loaded from --pretrained-root (stock code ignores local_save_path
                        and downloads from dl.fbaipublicfiles.com)
  fvd                   WorldLens videogpt-I3D features + frechet_distance, but the GT side is
                        the exported GT mp4s instead of the nuScenes 12 Hz info pkl, and clips
                        shorter than 16 frames are nearest-neighbour upsampled in time to 16
                        (I3D's temporal stride 8 + avgpool(2) needs >= 16 frames).
                        FVD on < ~400 clips is rank-deficient: report it only as a smoke number.

--random-weights builds every network with random init (no checkpoint files): numbers are
meaningless, it only proves the data path end to end.  Used by the dummy smoke test while the
pretrained_models download is blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for experiments.worldlens.layout
from experiments.worldlens import layout  # noqa: E402

GEN_METRICS = ("fvd", "temporal_consistency", "subject_consistency")


def _setup_worldlens(worldlens_root: str) -> None:
    root = str(Path(worldlens_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    # stock metrics use paths relative to the WorldLens checkout (third_party/dino, ...)
    os.chdir(root)


# --------------------------------------------------------------------------- model builders
def _clip_model(pretrained_root, random_weights, device):
    import clip
    import torch

    if random_weights:
        from clip.model import CLIP

        model = CLIP(
            embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768,
            vision_patch_size=32, context_length=77, vocab_size=49408,
            transformer_width=512, transformer_heads=8, transformer_layers=12,
        ).to(device).eval()
        return model, None
    path = Path(pretrained_root) / "clip" / "ViT-B-32.pt"
    _require(path)
    model, preprocess = clip.load(str(path), device=device)
    return model, preprocess


def _dino_model(pretrained_root, random_weights, device):
    import torch

    model = torch.hub.load("worldbench/third_party/dino", "dino_vitb16", source="local", pretrained=False)
    if not random_weights:
        path = Path(pretrained_root) / "dino" / "dino_vitbase16_pretrain.pth"
        _require(path)
        model.load_state_dict(torch.load(str(path), map_location="cpu"), strict=True)
    return model.to(device).eval()


def _i3d_model(pretrained_root, random_weights, device):
    import torch
    from worldbench.videogen.generation.perceptual_fidelity.fvd.videogpt.pytorch_i3d import InceptionI3d

    i3d = InceptionI3d(400, in_channels=3).eval()
    if not random_weights:
        path = Path(pretrained_root) / "fvd" / "i3d_pretrained_400.pt"
        _require(path)
        i3d.load_state_dict(torch.load(str(path), map_location="cpu"))
    return torch.nn.DataParallel(i3d.to(device))


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Download WorldLens pretrained_models "
            "(https://huggingface.co/datasets/worldbench/videogen/tree/main/pretrained_models) "
            "or pass --random-weights for a pipeline-only smoke test."
        )


# --------------------------------------------------------------------------- metrics
def run_stock_consistency(name, wl_root, method, pretrained_root, random_weights, gen_idx):
    """Instantiate the stock WorldLens class without its init_model, inject our model, call it."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_root = str(Path(wl_root).resolve() / "generated_results")
    if name == "temporal_consistency":
        from worldbench.videogen.generation.temporal_consistency.temporal_consistency import (
            TEMPORAL_CONSISTENCY as Cls,
        )

        class Metric(Cls):
            def init_model(self):
                return _clip_model(pretrained_root, random_weights, device)

    elif name == "subject_consistency":
        from worldbench.videogen.generation.subject_consistency.subject_consistency import (
            SUBJECT_CONSISTENCY as Cls,
        )

        class Metric(Cls):
            def init_model(self):
                return _dino_model(pretrained_root, random_weights, device)

    else:
        raise KeyError(name)

    # stock code only iterates repeat index i == gen suffix, so use repeat_times = gen_idx + 1
    metric = Metric(method_name=method, generated_data_path=gen_root, repeat_times=gen_idx + 1)
    with torch.no_grad():
        metric()
    out = Path(gen_root) / method / name / f"repeat_{gen_idx}.json"
    with open(out) as f:
        res = json.load(f)
    return {
        "n_videos": len(res["video_results"]),
        "acm_per_frame": res["temporal_consistency_per_frame"],
        "tji_score": res["tji_per_frame"],
        "ts": res["ts_per_frame"],
        "detail_json": str(out),
    }


def _read_clip_tensor(path, min_frames=16):
    """[T,C,H,W] float in [0,1] at WorldLens FVD resolution (225x400 resize, 224x400 top-centre crop)."""
    import torch
    import torch.nn.functional as F

    frames = layout.read_mp4(path)  # T,H,W,3 uint8
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    # stock: Resize((225, 400), bicubic) then top_center_crop -> (224, 400)
    x = F.interpolate(x, size=(225, 400), mode="bicubic", align_corners=False).clamp(0, 1)
    x = x[:, :, 1:, :]
    t = x.shape[0]
    if t < min_frames:
        idx = torch.floor(torch.arange(min_frames) * t / min_frames).long()
        x = x[idx]
    return x


def run_fvd(wl_root, method, pretrained_root, random_weights, gen_idx, batch_size=8):
    import torch
    from worldbench.videogen.generation.perceptual_fidelity.fvd.videogpt.fvd import frechet_distance, get_fvd_logits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    i3d = _i3d_model(pretrained_root, random_weights, device)
    gen_paths = sorted(glob(str(layout.video_submission_dir(wl_root, method) / f"*_gen{gen_idx}" / "*.mp4")))
    if not gen_paths:
        raise FileNotFoundError(f"no generated clips for {method}")
    gt_paths = [p.replace(f"/generated_results/{method}/", "/generated_results/gt/") for p in gen_paths]

    def feats(paths):
        out = []
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([_read_clip_tensor(p) for p in paths[i : i + batch_size]])  # B,T,C,H,W
            batch = batch.permute(0, 2, 1, 3, 4)  # B,C,T,H,W  (stock `trans`)
            with torch.no_grad():
                out.append(get_fvd_logits(batch, i3d=i3d, device=device).cpu())
        return torch.cat(out).double()

    f_gen, f_gt = feats(gen_paths), feats(gt_paths)
    save_dir = Path(wl_root) / "generated_results" / method / "fvd_navsim"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "gen_feats.npy", f_gen.numpy())
    np.save(save_dir / "gt_feats.npy", f_gt.numpy())
    fvd = frechet_distance(f_gt.clone(), f_gen.clone())
    return {
        "fvd": float(fvd),
        "n_clips": len(gen_paths),
        "feature_dim": int(f_gen.shape[1]),
        "rank_deficient": len(gen_paths) <= int(f_gen.shape[1]),
        "note": "GT = NAVSIM cam_f0 future clips; T<16 upsampled to 16 (nearest)",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wl-root", required=True)
    p.add_argument("--method", default="suvnavsim")
    p.add_argument("--worldlens-root", default=os.environ.get("WL_ROOT", "/workspace/SUV-deps/worldlens"))
    p.add_argument(
        "--pretrained-root",
        default=os.environ.get("WL_PRETRAINED", "/data-work/SUV/worldlens/pretrained_models"),
    )
    p.add_argument("--metrics", nargs="+", default=list(GEN_METRICS), choices=GEN_METRICS)
    p.add_argument("--gen-idx", type=int, default=0)
    p.add_argument("--random-weights", action="store_true")
    p.add_argument("--out-json", default=None)
    a = p.parse_args(argv)

    wl_root = str(Path(a.wl_root).resolve())
    layout.check_method_name(wl_root, a.method)
    _setup_worldlens(a.worldlens_root)

    results = {
        "method": a.method,
        "wl_root": wl_root,
        "random_weights": a.random_weights,
        "worldlens_root": str(Path(a.worldlens_root).resolve()),
        "metrics": {},
    }
    for name in a.metrics:
        t0 = time.time()
        if name == "fvd":
            r = run_fvd(wl_root, a.method, a.pretrained_root, a.random_weights, a.gen_idx)
        else:
            r = run_stock_consistency(name, wl_root, a.method, a.pretrained_root, a.random_weights, a.gen_idx)
        r["sec"] = round(time.time() - t0, 2)
        results["metrics"][name] = r
        print(f"[worldlens] {name}: {json.dumps({k: v for k, v in r.items() if k != 'detail_json'})}", flush=True)

    out_json = Path(a.out_json) if a.out_json else Path(wl_root) / f"metrics_{a.method}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[worldlens] wrote {out_json}")
    return results


if __name__ == "__main__":
    main()

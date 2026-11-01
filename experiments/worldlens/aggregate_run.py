"""Collect one WorldLens protocol-(a) run into a single metrics.json (either env, CPU).

    python -m experiments.worldlens.aggregate_run \
        --scenes $RUN/scenes.txt --raw-root $ROLL/raw --wl-root $ROLL/wl \
        --method suvnavsim --gt-method gtref --wall-log $RUN/gpu_wall.jsonl --out $RUN/metrics.json

Inputs
  scenes.txt           requested tokens (select_tokens.py)
  raw/<token>/         successful rollouts; raw/failures.jsonl, raw/timing.jsonl
  wl/metrics_<m>.json  run_metrics.py output for the method and for the GT-vs-GT reference
  gpu_wall.jsonl       {"stage", "sec"} per GPU stage invocation (written by run_wl_eval.sh)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _consistency(m):
    return None if m is None else {
        "ACM": m["acm_per_frame"], "TJI_score": m["tji_score"], "TS": m["ts"], "n_videos": m["n_videos"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--wl-root", required=True)
    p.add_argument("--method", default="suvnavsim")
    p.add_argument("--gt-method", default="gtref")
    p.add_argument("--wall-log", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    raw = Path(a.raw_root)
    wl = Path(a.wl_root)
    requested = [t.strip() for t in Path(a.scenes).read_text().splitlines() if t.strip()]
    done = [t for t in requested if (raw / t / "streams.npz").exists()]
    failed = sorted(set(requested) - set(done))
    last_err = {}
    for r in _jsonl(raw / "failures.jsonl"):
        last_err[r["token"]] = f'{r["stage"]}: {r["error"]}'
    timing = _jsonl(raw / "timing.jsonl")
    wall = _jsonl(a.wall_log) if a.wall_log else []

    ade = []
    for t in done:
        with open(raw / t / "meta.json") as f:
            v = json.load(f).get("ade_m")
            if v is not None:
                ade.append(v)

    def load(m):
        f = wl / f"metrics_{m}.json"
        return json.loads(f.read_text())["metrics"] if f.exists() else {}

    gen, ref = load(a.method), load(a.gt_method)
    fvd = gen.get("fvd")
    out = {
        "run_id": a.run_id,
        "protocol": "WorldLens (a): SUV future cam_f0 (8 frames, 2 Hz, 384x640) vs NAVSIM GT future, navtest",
        "scenes": {
            "requested": len(requested),
            "evaluated": len(done),
            "failed": len(failed),
            "failed_tokens": {t: last_err.get(t, "unknown") for t in failed},
        },
        "suv": {
            "G3_subject_consistency": _consistency(gen.get("subject_consistency")),
            "G5_temporal_consistency": _consistency(gen.get("temporal_consistency")),
            "G7_fvd": None if fvd is None else {
                "fvd_full": fvd["fvd"], "n_clips": fvd["n_clips"],
                "fvd_half_mean": fvd["half_split"]["gen_vs_gt_halves_mean"],
                "fvd_half_A_vs_gtB": fvd["half_split"]["gen_A_vs_gt_B"],
                "fvd_half_B_vs_gtA": fvd["half_split"]["gen_B_vs_gt_A"],
            },
            "ade_m_mean": float(np.mean(ade)) if ade else None,
        },
        "gt_reference": {
            "G3_subject_consistency": _consistency(ref.get("subject_consistency")),
            "G5_temporal_consistency": _consistency(ref.get("temporal_consistency")),
            "G7_fvd_floor_gtA_vs_gtB": None if fvd is None else fvd["half_split"]["gt_A_vs_gt_B"],
            "note": "GT vs itself: MRS = S_acm = 1 so TS = ACM; FVD floor = two disjoint halves of the GT clips",
        },
        "compute": {
            "rollout_sec_per_scene_mean": float(np.mean([r["sec"] for r in timing])) if timing else None,
            "rollout_gpu_hours_scene_sum": sum(r["sec"] for r in timing) / 3600,
            "gpu_hours_wall_by_stage": {
                s: sum(r["sec"] for r in wall if r["stage"] == s) / 3600 for s in sorted({r["stage"] for r in wall})
            },
            "gpu_hours_wall_total": sum(r["sec"] for r in wall) / 3600,
        },
        "sources": {
            "raw_root": str(raw), "wl_root": str(wl),
            "method_metrics": str(wl / f"metrics_{a.method}.json"),
            "gt_reference_metrics": str(wl / f"metrics_{a.gt_method}.json"),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("scenes", "suv", "gt_reference", "compute")}, indent=2)[:4000])


if __name__ == "__main__":
    main()

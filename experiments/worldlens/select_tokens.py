"""Pick an evenly strided subset of navtest scene tokens (CPU only, SUV env).

    python -m experiments.worldlens.select_tokens \
        --navsim-log-path $OPENSCENE_DATA_ROOT/navsim_logs/test \
        --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/test \
        --n 1000 --out $RUN/scenes.txt

Tokens = official navtest scene filter (navsim v1 ``navtest.yaml``: log names + token list, has_route)
re-windowed with SUV's 4 history / 8 future frames, sorted lexicographically; the subset is
``sorted_tokens[floor(i * N / n)]`` for i in 0..n-1 (deterministic, no RNG).  Writes the token list
plus a json with the selection provenance and how many selected tokens currently have camera blobs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger("select_tokens")


def navtest_yaml_path() -> Path:
    import navsim

    return (
        Path(navsim.__file__).parent
        / "planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--navsim-log-path", required=True)
    p.add_argument("--sensor-blobs-path", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--num-history-frames", type=int, default=4)
    p.add_argument("--num-future-frames", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    cfg_path = navtest_yaml_path()
    cfg = yaml.safe_load(cfg_path.read_text())
    scene_filter = SceneFilter(
        num_history_frames=a.num_history_frames,
        num_future_frames=a.num_future_frames,
        frame_interval=1,
        has_route=bool(cfg.get("has_route", True)),
        log_names=list(cfg["log_names"]),
        tokens=list(cfg["tokens"]),
    )
    loader = SceneLoader(
        sensor_blobs_path=Path(a.sensor_blobs_path),
        data_path=Path(a.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    all_tokens = sorted(loader.tokens)
    n_all = len(all_tokens)
    if a.n > n_all:
        raise ValueError(f"asked for {a.n} of {n_all} tokens")
    idx = [(i * n_all) // a.n for i in range(a.n)]
    chosen = [all_tokens[i] for i in idx]

    blob_logs = {q.name for q in Path(a.sensor_blobs_path).iterdir() if q.is_dir()}
    log_of = {t: loader.scene_frames_dicts[t][0]["log_name"] for t in chosen}
    missing = [t for t in chosen if log_of[t] not in blob_logs]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + "\n")
    info = {
        "rule": "sorted(navtest tokens)[floor(i*N/n)], i=0..n-1",
        "navtest_yaml": str(cfg_path),
        "navtest_yaml_tokens": len(cfg["tokens"]),
        "navtest_yaml_logs": len(cfg["log_names"]),
        "num_history_frames": a.num_history_frames,
        "num_future_frames": a.num_future_frames,
        "n_navtest_tokens": n_all,
        "n_selected": len(chosen),
        "n_logs_selected": len(set(log_of.values())),
        "n_selected_without_blobs_at_selection_time": len(missing),
        "token_log": log_of,
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    logger.info("navtest tokens=%d selected=%d logs=%d missing_blobs=%d -> %s",
                n_all, len(chosen), info["n_logs_selected"], len(missing), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

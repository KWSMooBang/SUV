"""Pick N navtest tokens whose front-camera blobs are already on disk (for smoke runs).

usage: python experiments/env/pick_smoke_tokens.py --n 20 [--split-yaml PATH]
Prints a Hydra list override value: [tok1,tok2,...] and the log names used, to stderr.
Tokens are spread round-robin across the available logs.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split-yaml", default=None, help="navtest scene filter yaml (default: installed navsim)")
    a = ap.parse_args()

    root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    if a.split_yaml is None:
        import navsim

        a.split_yaml = (
            Path(navsim.__file__).parent / "planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
        )
    cfg = yaml.safe_load(Path(a.split_yaml).read_text())
    navtest_tokens = set(cfg["tokens"])
    navtest_logs = set(cfg["log_names"])

    blob_logs = sorted(p.name for p in (root / "sensor_blobs/test").iterdir() if (p / "CAM_F0").is_dir())
    logs = [log for log in blob_logs if log in navtest_logs]

    import pickle

    per_log = {}
    for log in logs:
        frames = pickle.load(open(root / "navsim_logs/test" / f"{log}.pkl", "rb"))
        blobs = root / "sensor_blobs/test"
        def valid(idx: int) -> bool:
            # navtest token whose 4 history CAM_F0 frames (current + 3 previous) are on disk
            # mirror SceneLoader: route present (has_route) and 8 future frames available
            return (
                idx >= 3
                and idx + 8 < len(frames)
                and len(frames[idx]["roadblock_ids"]) > 0
                and frames[idx]["token"] in navtest_tokens
                and all((blobs / frames[j]["cams"]["CAM_F0"]["data_path"]).is_file() for j in range(idx - 3, idx + 1))
            )

        # Pick adjacent (previous, current) frame pairs: the v2 one-stage scorer computes
        # two-frame extended comfort over adjacent tokens and fails if no pair exists.
        per_log[log] = [
            (frames[idx - 1]["token"], frames[idx]["token"])
            for idx in range(1, len(frames))
            if valid(idx) and valid(idx - 1)
        ]

    picked, i = [], 0
    while len(picked) < a.n and any(per_log.values()):
        log = logs[i % len(logs)]
        if per_log[log]:
            picked.extend(per_log[log].pop(len(per_log[log]) // 2))
        i += 1
    picked = picked[: a.n]
    used = sorted({log for log in logs})
    print(f"logs with blobs: {used}", file=sys.stderr)
    print("[" + ",".join(picked) + "]")


if __name__ == "__main__":
    main()

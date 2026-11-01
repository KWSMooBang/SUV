#!/usr/bin/env bash
# Pipeline smoke without an SUV checkpoint: synthetic raw dumps -> export -> WorldLens metrics.
#   bash experiments/worldlens/scripts/smoke_dummy.sh [out_dir]
#   RANDOM_WEIGHTS=1 ...   # also skip pretrained_models (numbers meaningless)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WL_PY="${WL_PY:-/workspace/SUV-deps/envs/worldlens/bin/python}"
OUT="${1:-${OUT:-/data-work/SUV/exp/worldlens/smoke_dummy}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

"${WL_PY}" -m experiments.worldlens.make_dummy --raw-root "${OUT}/raw" --n-scenes "${N_SCENES:-6}"
"${WL_PY}" -m experiments.worldlens.export_worldlens --raw-root "${OUT}/raw" --wl-root "${OUT}/wl" \
  --method suvdummy --clip "${CLIP:-future}" --aux-streams --overwrite
extra=()
[[ "${RANDOM_WEIGHTS:-0}" == 1 ]] && extra+=(--random-weights)
"${WL_PY}" "${REPO_ROOT}/experiments/worldlens/run_metrics.py" --wl-root "${OUT}/wl" --method suvdummy \
  --metrics fvd temporal_consistency subject_consistency "${extra[@]}"

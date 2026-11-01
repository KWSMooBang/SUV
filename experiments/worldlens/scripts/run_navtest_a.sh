#!/usr/bin/env bash
# Protocol (a): SUV on NAVSIM navtest -> WorldLens single-view Generation metrics.
#
#   stage 1 (SUV env)       : rollout -> raw dumps (4 streams + GT frames + trajectories)
#   stage 2 (either env)    : export  -> generated_results/{suvnavsim,gt}/video_submission/<token>_gen0/CAM_FRONT.mp4
#   stage 3 (WorldLens env) : metrics -> metrics_suvnavsim.json
#
#   MAX_SCENES=8 bash experiments/worldlens/scripts/run_navtest_a.sh            # all stages
#   STAGES="export metrics" bash experiments/worldlens/scripts/run_navtest_a.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data-work/SUV}"

SUV_PY="${SUV_PY:-/workspace/envs/suv/bin/python}"
WL_PY="${WL_PY:-/workspace/SUV-deps/envs/worldlens/bin/python}"
WL_ROOT="${WL_ROOT:-/workspace/SUV-deps/worldlens}"
WL_PRETRAINED="${WL_PRETRAINED:-${DATA_ROOT}/worldlens/pretrained_models}"

CKPT="${CKPT:-${CKPT_LOCAL_DIR:-${DATA_ROOT}/SUV_ckpt}/suv_navsim.pt}"
TEXT_EMB="${TEXT_EMB:-${TEXT_EMBEDDING_CACHE_DIR_V1:-${DATA_ROOT}/text_embeds/v1}}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/navsim}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"

OUT="${OUT:-${DATA_ROOT}/exp/worldlens/navtest_a}"
METHOD="${METHOD:-suvnavsim}"
CLIP="${CLIP:-future}"
MAX_SCENES="${MAX_SCENES:-8}"
STRIDE="${STRIDE:-1}"
METRICS="${METRICS:-fvd temporal_consistency subject_consistency}"
STAGES="${STAGES:-rollout export metrics}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
# same convention as experiments/navsimv1/scripts/evaluation/run_pdm_score.sh
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-${DATA_ROOT}/SUV_ckpt}"
export OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

for s in ${STAGES}; do
  case "$s" in
  rollout)
    "${SUV_PY}" -m experiments.worldlens.suv_rollout \
      --checkpoint "${CKPT}" --text-embedding-cache-dir "${TEXT_EMB}" \
      --navsim-log-path "${NAVSIM_LOG_PATH}" --sensor-blobs-path "${SENSOR_BLOBS_PATH}" \
      --raw-root "${OUT}/raw" --max-scenes "${MAX_SCENES}" --stride "${STRIDE}" ;;
  export)
    "${WL_PY}" -m experiments.worldlens.export_worldlens \
      --raw-root "${OUT}/raw" --wl-root "${OUT}/wl" --method "${METHOD}" --clip "${CLIP}" ;;
  metrics)
    # shellcheck disable=SC2086
    "${WL_PY}" "${REPO_ROOT}/experiments/worldlens/run_metrics.py" \
      --wl-root "${OUT}/wl" --method "${METHOD}" --worldlens-root "${WL_ROOT}" \
      --pretrained-root "${WL_PRETRAINED}" --metrics ${METRICS} ${EXTRA_METRIC_ARGS:-} ;;
  *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done

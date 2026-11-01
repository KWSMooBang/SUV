#!/usr/bin/env bash
# wl-eval: WorldLens protocol (a) on an evenly strided navtest subset, resumable.
#
#   RUN_ID=wl-a-navtest1k-s0 N_SCENES=1000 SEED=0 \
#     nohup bash experiments/worldlens/scripts/run_wl_eval.sh > $RUN_DIR/run.log 2>&1 &
#
# Stages (STAGES, default all): select rollout export metrics gtref aggregate
#   select    (CPU)  scenes.txt = sorted(navtest)[floor(i*N/n)]          (skipped if scenes.txt exists)
#   rollout   (GPU)  SUV -> raw/<token>/streams.npz; skips finished scenes, logs failures, budget guard
#   export    (CPU)  raw -> wl/generated_results/{suvnavsim,gt}/video_submission/*.mp4 (skips existing mp4)
#   metrics   (GPU)  FVD (+ half-split floor), Temporal (CLIP), Subject (DINO) for SUV vs GT
#   gtref     (GPU)  Temporal / Subject for GT vs GT (gtref -> symlink to gt clips): GT ACM / TS reference
#   aggregate (CPU)  $RUN_DIR/metrics.json
# Pins the GPU explicitly (default 0); does not use /harness/launch.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data-work/SUV}"
RUN_ID="${RUN_ID:-wl-a-navtest1k-s0}"
N_SCENES="${N_SCENES:-1000}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-/outputs/SUV/runs/${RUN_ID}}"
ROLL_DIR="${ROLL_DIR:-${DATA_ROOT}/worldlens/rollouts/${RUN_ID}}"
METHOD="${METHOD:-suvnavsim}"
GT_METHOD="${GT_METHOD:-gtref}"
BUDGET_GPU_HOURS="${BUDGET_GPU_HOURS:-3.0}"
STAGES="${STAGES:-select rollout export metrics gtref aggregate}"

SUV_PY="${SUV_PY:-/workspace/envs/suv/bin/python}"
WL_PY="${WL_PY:-/workspace/SUV-deps/envs/worldlens/bin/python}"
WL_ROOT="${WL_ROOT:-/workspace/SUV-deps/worldlens}"
WL_PRETRAINED="${WL_PRETRAINED:-${DATA_ROOT}/worldlens/pretrained_models}"
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-${DATA_ROOT}/SUV_ckpt}"
CKPT="${CKPT:-${CKPT_LOCAL_DIR}/suv_navsim.pt}"
TEXT_EMB="${TEXT_EMBEDDING_CACHE_DIR:-${DATA_ROOT}/text_embeds/v1}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true

mkdir -p "${RUN_DIR}" "${ROLL_DIR}"
SCENES="${RUN_DIR}/scenes.txt"
WALL_LOG="${RUN_DIR}/gpu_wall.jsonl"
echo "[wl-eval] run=${RUN_ID} gpu=${CUDA_VISIBLE_DEVICES} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD) $(date -Is)"

timed_gpu() {  # stage cmd...  -> appends wall seconds to gpu_wall.jsonl
  local stage=$1; shift
  local t0; t0=$(date +%s)
  local rc=0
  "$@" || rc=$?
  echo "{\"stage\": \"${stage}\", \"sec\": $(( $(date +%s) - t0 )), \"rc\": ${rc}, \"end\": \"$(date -Is)\"}" >> "${WALL_LOG}"
  return ${rc}
}

for s in ${STAGES}; do
  echo "[wl-eval] stage ${s} $(date -Is)"
  case "$s" in
  select)
    if [[ -s "${SCENES}" ]]; then echo "  ${SCENES} exists, keeping it"; continue; fi
    CUDA_VISIBLE_DEVICES= "${SUV_PY}" -m experiments.worldlens.select_tokens \
      --navsim-log-path "${NAVSIM_LOG_PATH}" --sensor-blobs-path "${SENSOR_BLOBS_PATH}" \
      --n "${N_SCENES}" --out "${SCENES}" ;;
  rollout)
    timed_gpu rollout "${SUV_PY}" -m experiments.worldlens.suv_rollout \
      --checkpoint "${CKPT}" --text-embedding-cache-dir "${TEXT_EMB}" \
      --navsim-log-path "${NAVSIM_LOG_PATH}" --sensor-blobs-path "${SENSOR_BLOBS_PATH}" \
      --raw-root "${ROLL_DIR}/raw" --tokens-file "${SCENES}" --seed "${SEED}" \
      --budget-gpu-hours "${BUDGET_GPU_HOURS}" ;;
  export)
    "${WL_PY}" -m experiments.worldlens.export_worldlens \
      --raw-root "${ROLL_DIR}/raw" --wl-root "${ROLL_DIR}/wl" --method "${METHOD}" --clip future ;;
  metrics)
    timed_gpu metrics "${WL_PY}" "${REPO_ROOT}/experiments/worldlens/run_metrics.py" \
      --wl-root "${ROLL_DIR}/wl" --method "${METHOD}" --worldlens-root "${WL_ROOT}" \
      --pretrained-root "${WL_PRETRAINED}" --metrics fvd temporal_consistency subject_consistency ;;
  gtref)
    ref="${ROLL_DIR}/wl/generated_results/${GT_METHOD}"
    mkdir -p "${ref}"
    [[ -e "${ref}/video_submission" ]] || ln -s ../gt/video_submission "${ref}/video_submission"
    timed_gpu gtref "${WL_PY}" "${REPO_ROOT}/experiments/worldlens/run_metrics.py" \
      --wl-root "${ROLL_DIR}/wl" --method "${GT_METHOD}" --worldlens-root "${WL_ROOT}" \
      --pretrained-root "${WL_PRETRAINED}" --metrics temporal_consistency subject_consistency ;;
  aggregate)
    "${WL_PY}" -m experiments.worldlens.aggregate_run \
      --scenes "${SCENES}" --raw-root "${ROLL_DIR}/raw" --wl-root "${ROLL_DIR}/wl" \
      --method "${METHOD}" --gt-method "${GT_METHOD}" --wall-log "${WALL_LOG}" \
      --run-id "${RUN_ID}" --out "${RUN_DIR}/metrics.json"
    cp -f "${ROLL_DIR}/wl/metrics_${METHOD}.json" "${ROLL_DIR}/wl/metrics_${GT_METHOD}.json" "${RUN_DIR}/" 2>/dev/null || true
    cp -f "${ROLL_DIR}/raw/failures.jsonl" "${ROLL_DIR}/raw/timing.jsonl" "${RUN_DIR}/" 2>/dev/null || true ;;
  *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done
echo "[wl-eval] done $(date -Is)"

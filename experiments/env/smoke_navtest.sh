#!/usr/bin/env bash
# Small-scale smoke run on a single GPU; records wall time + peak GPU memory
# (nvidia-smi sampling).
#   bash experiments/env/smoke_navtest.sh v2 [N=20]   # navtest, N tokens with camera blobs on disk
#   bash experiments/env/smoke_navtest.sh v1 [N=20]
#   bash experiments/env/smoke_navtest.sh navhard [LOG] # navhard_two_stage restricted to one log
#                                                      # (stage-1 tokens + their stage-2 synthetic scenes)
# Outputs go to ${NAVSIM_EXP_ROOT}/smoke/<v>/ ; summary in smoke_summary.txt there.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/experiments/env/suv_paths.sh"

WHICH="${1:?usage: smoke_navtest.sh v1|v2 [N]}"
N="${2:-20}"
GPU="${CUDA_VISIBLE_DEVICES%%,*}"
SMOKE_ROOT="${NAVSIM_EXP_ROOT}/smoke/${WHICH}"
mkdir -p "${SMOKE_ROOT}"

if [[ "${WHICH}" == navhard ]]; then
  NAVHARD_LOG="${2:-2021.05.25.14.26.37_veh-27_04122_04279}"
  N="log ${NAVHARD_LOG}"
else
  TOKENS="$(PYTHONPATH="${REPO_ROOT}/experiments/navsimv2/vendor" python -W ignore \
    "${REPO_ROOT}/experiments/env/pick_smoke_tokens.py" --n "${N}" 2>/dev/null)"
  echo "${TOKENS}" > "${SMOKE_ROOT}/tokens.txt"
fi

# sample GPU memory of the chosen GPU every second
nvidia-smi -i "${GPU}" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 1 \
  > "${SMOKE_ROOT}/gpu_mem.csv" &
SMI=$!
trap 'kill ${SMI} 2>/dev/null || true' EXIT

T0=$(date +%s)
case "${WHICH}" in
  v1)
    NAVSIM_EXP_ROOT="${SMOKE_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR_V1}" METRIC_CACHE_PATH="${METRIC_CACHE_PATH_V1}" \
      bash "${REPO_ROOT}/experiments/navsimv1/scripts/evaluation/run_pdm_score.sh" \
      "train_test_split.scene_filter.tokens=${TOKENS}" ;;
  v2)
    NAVSIM_EXP_ROOT="${SMOKE_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR_V2}" METRIC_CACHE_PATH="${METRIC_CACHE_PATH_V2}" \
      bash "${REPO_ROOT}/experiments/navsimv2/scripts/evaluation/run_epdm_score.sh" \
      "train_test_split.scene_filter.tokens=${TOKENS}" ;;
  navhard)
    NAVSIM_EXP_ROOT="${SMOKE_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR_V2}" METRIC_CACHE_PATH="${METRIC_CACHE_PATH_NAVHARD}" \
      bash "${REPO_ROOT}/experiments/navsimv2/scripts/evaluation/run_navhard_epdm_score.sh" \
      "train_test_split.scene_filter.log_names=['${NAVHARD_LOG}']" ;;
  *) echo "unknown target ${WHICH}" >&2; exit 2 ;;
esac 2>&1 | tee "${SMOKE_ROOT}/run.log"
T1=$(date +%s)

PEAK=$(awk -F', ' '{if ($2+0>m) m=$2+0} END {print m}' "${SMOKE_ROOT}/gpu_mem.csv")
{
  echo "target=${WHICH} scenes=${N} gpu=${GPU}"
  echo "wall_seconds=$((T1 - T0))  (includes model load)"
  echo "peak_gpu_mem_MiB=${PEAK}  (nvidia-smi memory.used, whole GPU)"
} | tee "${SMOKE_ROOT}/smoke_summary.txt"

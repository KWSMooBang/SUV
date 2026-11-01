# Source this file before running any SUV NAVSIM preparation/evaluation script:
#   source experiments/env/suv_paths.sh
# Every value can be overridden by exporting it before sourcing.
# The evaluation scripts read the *_V1 / *_V2 / *_NAVHARD variants when the
# generic TEXT_EMBEDDING_CACHE_DIR / METRIC_CACHE_PATH are not set.

export SUV_DATA_ROOT="${SUV_DATA_ROOT:-/data-work/SUV}"
export SUV_VENV="${SUV_VENV:-/workspace/envs/suv}"

# Checkpoints: suv_navsim.pt + Wan2.2-TI2V-5B/
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-${SUV_DATA_ROOT}/SUV_ckpt}"

# NAVSIM data
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${SUV_DATA_ROOT}/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SUV_DATA_ROOT}/exp}"

# Metric caches (CPU-built)
export METRIC_CACHE_PATH_V1="${METRIC_CACHE_PATH_V1:-${SUV_DATA_ROOT}/metric_cache/navsim_v1}"
export METRIC_CACHE_PATH_V2="${METRIC_CACHE_PATH_V2:-${SUV_DATA_ROOT}/metric_cache/navsim_v2}"
export METRIC_CACHE_PATH_NAVHARD="${METRIC_CACHE_PATH_NAVHARD:-${SUV_DATA_ROOT}/metric_cache/navhard}"

# Text-embedding caches
export TEXT_EMBEDDING_CACHE_DIR_V1="${TEXT_EMBEDDING_CACHE_DIR_V1:-${SUV_DATA_ROOT}/text_embeds/v1}"
export TEXT_EMBEDDING_CACHE_DIR_V2="${TEXT_EMBEDDING_CACHE_DIR_V2:-${SUV_DATA_ROOT}/text_embeds/v2}"

# This machine: GPU 0 only (1-3 are used by other projects)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Python env (uv venv; conda is unusable in this container)
if [[ -z "${VIRTUAL_ENV:-}" && -f "${SUV_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SUV_VENV}/bin/activate"
fi

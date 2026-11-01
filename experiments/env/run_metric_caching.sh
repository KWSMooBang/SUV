#!/usr/bin/env bash
# Build the official NAVSIM metric caches used by SUV evaluation (CPU only).
#   bash experiments/env/run_metric_caching.sh v1       # navtest, NAVSIM v1 devkit (installed package)
#   bash experiments/env/run_metric_caching.sh v2       # navtest, vendored NAVSIM v2 devkit
#   bash experiments/env/run_metric_caching.sh navhard  # navhard_two_stage (wraps navsimv2/scripts/run_navhard_metric_caching.sh)
# Paths come from experiments/env/suv_paths.sh; extra Hydra overrides are forwarded.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/experiments/env/suv_paths.sh"

WHICH="${1:?usage: run_metric_caching.sh v1|v2|navhard [hydra overrides...]}"
shift
WORKERS="${WORKERS:-64}"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

case "${WHICH}" in
  v1)
    SCRIPT="$(python -c 'import navsim, pathlib; print(pathlib.Path(navsim.__file__).parent / "planning/script/run_metric_caching.py")')"
    python -u "${SCRIPT}" \
      train_test_split=navtest \
      "navsim_log_path=${OPENSCENE_DATA_ROOT}/navsim_logs/test" \
      "cache.cache_path=${METRIC_CACHE_PATH_V1}" \
      worker=single_machine_thread_pool \
      "worker.max_workers=${WORKERS}" \
      worker.use_process_pool=true \
      "$@"
    ;;
  v2)
    export PYTHONPATH="${REPO_ROOT}/experiments/navsimv2/vendor:${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
    python -u "${REPO_ROOT}/experiments/navsimv2/vendor/navsim/planning/script/run_metric_caching.py" \
      train_test_split=navtest \
      "navsim_log_path=${OPENSCENE_DATA_ROOT}/navsim_logs/test" \
      "metric_cache_path=${METRIC_CACHE_PATH_V2}" \
      worker=single_machine_thread_pool \
      "worker.max_workers=${WORKERS}" \
      worker.use_process_pool=true \
      "$@"
    ;;
  navhard)
    METRIC_CACHE_PATH="${METRIC_CACHE_PATH_NAVHARD}" WORKER_MAX_WORKERS="${WORKERS}" WORKER_USE_PROCESS_POOL=true \
      bash "${REPO_ROOT}/experiments/navsimv2/scripts/run_navhard_metric_caching.sh" "$@"
    ;;
  *) echo "unknown target ${WHICH}" >&2; exit 2 ;;
esac

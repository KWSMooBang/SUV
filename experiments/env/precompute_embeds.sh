#!/usr/bin/env bash
# Precompute SUV text-embedding caches with the paths from suv_paths.sh.
#   bash experiments/env/precompute_embeds.sh v1   # navtest prompts, NAVSIM v1
#   bash experiments/env/precompute_embeds.sh v2   # navtest + navhard_two_stage prompts, NAVSIM v2
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/experiments/env/suv_paths.sh"
WHICH="${1:?usage: precompute_embeds.sh v1|v2}"
shift
case "${WHICH}" in
  v1) TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR_V1}" \
        bash "${REPO_ROOT}/experiments/navsimv1/scripts/evaluation/precompute_suv_navsimv1_pdm_text_embeds.sh" "$@" ;;
  v2) TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR_V2}" \
        bash "${REPO_ROOT}/experiments/navsimv2/scripts/evaluation/precompute_text_embeddings.sh" "$@" ;;
  *) echo "unknown target ${WHICH}" >&2; exit 2 ;;
esac

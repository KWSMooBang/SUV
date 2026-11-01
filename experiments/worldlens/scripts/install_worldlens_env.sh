#!/usr/bin/env bash
# Build the WorldLens evaluation env without touching the system CUDA (12.4).
#
#   bash experiments/worldlens/scripts/install_worldlens_env.sh base   # conda env + nvcc 11.8
#   bash experiments/worldlens/scripts/install_worldlens_env.sh gen    # torch 2.1/cu118 + Generation-metric deps (no mmcv)
#   bash experiments/worldlens/scripts/install_worldlens_env.sh mm     # mmcv-1.6.0 ops, mmseg, mmdet, bevfusion, mmdet3d (Downstream only)
#
# Paths (override via env):
#   WL_ROOT   WorldLens clone        (default /workspace/SUV-deps/worldlens @ fa6c5ff)
#   WL_ENV    env prefix             (default /workspace/SUV-deps/envs/worldlens)
set -euo pipefail

WL_ROOT="${WL_ROOT:-/workspace/SUV-deps/worldlens}"
WL_ENV="${WL_ENV:-/workspace/SUV-deps/envs/worldlens}"
CONDA="${CONDA:-/opt/miniforge3/bin/conda}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/workspace/SUV-deps/conda-pkgs}"
PY="${WL_ENV}/bin/python"
PIP="${PY} -m pip"
stage="${1:?usage: $0 base|gen|mm}"

case "${stage}" in
base)
  "${CONDA}" create -y -p "${WL_ENV}" -c nvidia/label/cuda-11.8.0 -c conda-forge \
    python=3.9.20 cuda-toolkit=11.8 "gxx_linux-64=11" "gcc_linux-64=11" ninja
  ;;
gen)
  ${PIP} install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
  # pinned to WorldLens environment.yml versions; only what the generation/reconstruction metrics import
  ${PIP} install numpy==1.23.5 "setuptools<70" loguru==0.7.3 easydict==1.13 einops==0.8.1 \
    decord==0.6.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 moviepy==1.0.3 matplotlib==3.5.2 \
    scipy==1.13.1 scikit-image==0.19.3 scikit-learn==1.6.1 opencv-python-headless==4.10.0.84 \
    pillow==9.5.0 tqdm hydra-core==1.3.2 omegaconf==2.3.0 openai-clip==1.0.1 ftfy regex \
    timm==1.0.21 lpips==0.1.4 pyiqa==0.1.14.1 transformers==4.37.2 pyyaml
  ;;
mm)
  # compile against the env's nvcc 11.8 + gcc 11, never /usr/local/cuda (12.4)
  export CUDA_HOME="${WL_ENV}" PATH="${WL_ENV}/bin:${PATH}"
  export CC="${WL_ENV}/bin/x86_64-conda-linux-gnu-gcc" CXX="${WL_ENV}/bin/x86_64-conda-linux-gnu-g++"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" MAX_JOBS="${MAX_JOBS:-16}"
  export CPATH="${WL_ENV}/targets/x86_64-linux/include:${WL_ENV}/include:${CPATH:-}"
  export LIBRARY_PATH="${WL_ENV}/lib:${WL_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  ${PIP} install numba==0.53.0 llvmlite==0.36.0 nuscenes-devkit==1.1.10 shapely==1.8.5 pyquaternion
  (cd "${WL_ROOT}/worldbench/third_party/mmcv-1.6.0" && MMCV_WITH_OPS=1 ${PIP} install --no-build-isolation -v -e .)
  ${PIP} install https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.30.0.zip
  ${PIP} install mmdet==2.28.2
  (cd "${WL_ROOT}/worldbench/third_party/bevfusion" && ${PY} setup.py develop)
  (cd "${WL_ROOT}/worldbench/third_party/mmdetection3d-1.0.0rc6" && ${PIP} install --no-build-isolation -v -e .)
  ${PIP} install numpy==1.23.5 numba==0.53.0
  ;;
*) echo "unknown stage ${stage}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Phase 2B eval: closed-loop success rate of an SFT'd Show-o2 in LIBERO-Spatial.
set -eo pipefail

VENV=/nyx-storage1/hanliu/envs/mirage_showo_venv
MIRAGE_DIR=/home/mzh1800/MIRAGE
SHOWO2_PATH=/home/mzh1800/Show-o-repo/show-o2
LOG_DIR=${MIRAGE_DIR}/logs
mkdir -p "${LOG_DIR}"

source "${VENV}/bin/activate"

export HF_HOME=/nyx-storage1/hanliu/hf
export HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub
export TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
export TORCH_HOME=/nyx-storage1/hanliu/torch_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=${SHOWO2_PATH}:${MIRAGE_DIR}:${PYTHONPATH:-}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python3 "${MIRAGE_DIR}/scripts/phase2b_eval.py" "$@" 2>&1 | tee "${LOG_DIR}/phase2b_eval.log"

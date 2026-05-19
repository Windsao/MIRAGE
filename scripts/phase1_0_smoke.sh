#!/usr/bin/env bash
# Phase 1.0 smoke test — confirm Show-o2-1.5B loads and runs mmu_generate
# on a single image without crashing.
#
# Gate: prints a non-empty caption for the demo image. If pass, proceed to 1.1.
#
# Usage (from the cluster):
#   bash /home/mzh1800/MIRAGE/scripts/phase1_0_smoke.sh
#
# Logs to /home/mzh1800/MIRAGE/logs/phase1_0_smoke.log.

set -eo pipefail

# Project layout
SHOWO_REPO=/home/mzh1800/Show-o-repo/show-o2
MIRAGE_DIR=/home/mzh1800/MIRAGE
CONFIG=${MIRAGE_DIR}/configs/showo2_smoke.yaml
VENV=/nyx-storage1/hanliu/envs/mirage_showo_venv
DEMO_IMAGE=${SHOWO_REPO}/docs/mmu/pexels-pixabay-207983.jpg
QUESTION="Describe this image in one short sentence."
LOG_DIR=${MIRAGE_DIR}/logs
mkdir -p "${LOG_DIR}"

# Env setup
source "${VENV}/bin/activate"
export HF_HOME=/nyx-storage1/hanliu/hf
export HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub
export TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
export TORCH_HOME=/nyx-storage1/hanliu/torch_cache
# Avoid wandb network round-trip on a smoke test
export WANDB_MODE=offline
export WANDB_DIR=${LOG_DIR}

# Pick a free GPU on the current node (caller responsible for being on a GPU node)
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    # Default to GPU 0; override via CUDA_VISIBLE_DEVICES=<idx> bash phase1_0_smoke.sh
    export CUDA_VISIBLE_DEVICES=0
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Demo image: ${DEMO_IMAGE}"
echo "Question: ${QUESTION}"

# Show-o2's inference_mmu.py uses OmegaConf CLI overrides (key=value)
cd "${SHOWO_REPO}"
python3 inference_mmu.py \
    config="${CONFIG}" \
    mmu_image_path="${DEMO_IMAGE}" \
    question="${QUESTION}" \
    2>&1 | tee "${LOG_DIR}/phase1_0_smoke.log"

echo "---"
echo "Smoke test complete. Inspect ${LOG_DIR}/phase1_0_smoke.log for the model's response."

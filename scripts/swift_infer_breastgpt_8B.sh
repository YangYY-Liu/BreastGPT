#!/usr/bin/env bash
# Batch inference on BreastStage-Bench with BreastGPT-8B.
#
# Args:
#   $1 MODEL        — path to a BreastGPT checkpoint (default: ./checkpoints/BreastGPT-8B)
#   $2 MAXTOKENS    — max new tokens per response (default: 1024)
#   $3 MASTER_PORT  — torchrun rendezvous port (default: 29501)
#
# Env (overridable):
#   BENCH_DIR        — directory holding BreastStage-Bench JSONLs
#                      (default: ./datasets/BreastStage-Bench)
#   OUTPUT_DIR       — where per-dataset predictions are written
#                      (default: ./infer_outputs)
#   PLUGIN_DIR       — path to BreastGPT external plugins (default: ./model)
#   SELECT_IMAGE_NUM — # visual tokens kept per radiology image (default: 64)
#   SELECT_HISTO_NUM — # visual tokens kept per WSI (default: 128)
#   SELECT_VIDEO_NUM — # visual tokens kept per 3D volume (default: 128)

set -euo pipefail

# ---------- paths (override via env or first arg) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL=${1:-"${REPO_ROOT}/checkpoints/BreastGPT-8B"}
MAXTOKENS=${2:-1024}
MASTER_PORT=${3:-29501}

BENCH_DIR=${BENCH_DIR:-"${REPO_ROOT}/datasets/BreastStage-Bench"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/infer_outputs"}
PLUGIN_DIR=${PLUGIN_DIR:-"${REPO_ROOT}/model"}

# ---------- runtime ----------
export MASTER_PORT
export NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

# BreastGPT visual front-end limits
export MAX_H=384
export MAX_W=384
export MAX_D=48

# Qwen3-VL visual token budget
export IMAGE_MAX_TOKEN_NUM=1024
export IMAGE_MIN_TOKEN_NUM=1
export VIDEO_MIN_TOKEN_NUM=1

# Concept-based token selector budgets
export SELECT_IMAGE_NUM=${SELECT_IMAGE_NUM:-64}
export SELECT_HISTO_NUM=${SELECT_HISTO_NUM:-128}
export SELECT_VIDEO_NUM=${SELECT_VIDEO_NUM:-128}

# ---------- BreastStage-Bench splits ----------
DATASETS=(
    # BUS
    "sampled_BUS-SWIFT_RESIZED_Stage_Lesion_Closed_VQA_Test_DownSample.json"
    "sampled_BUS-SWIFT_RESIZED_Stage_Lesion_Open_VQA_Test_DownSample.json"
    "sampled_BUS-SWIFT_RESIZED_Stage_GroundCaption_Test_DownSample.json"
    # CT
    "sampled_CT-SWIFT_RESIZED_Screen_Closed_VQA_Test_NII.json"
    "sampled_CT-SWIFT_RESIZED_Screen_Open_VQA_Test_NII.json"
    "sampled_CT-SWIFT_RESIZED_Screen_GroundCaption_Test_NII.json"
    # Histopathology (WSI)
    "sampled_Histopathology-SWIFT_RESIZED_Slide_Caption_Test_H5.json"
    "sampled_Histopathology-SWIFT_RESIZED_Slide_VQA_Open_Test_H5.json"
    "sampled_Histopathology-SWIFT_RESIZED_Slide_VQA_Closed_Test_H5.json"
    # Mammography
    "sampled_Mammography-SWIFT_RESIZED_Stage_Mammo_Closed_VQA_Test.json"
    "sampled_Mammography-SWIFT_RESIZED_Stage_Mammo_GroundCaption_Test.json"
    # MRI
    "sampled_MRI-SWIFT_RESIZED_Stage_JJ_Closed_VQA_Test_FULL_NII.json"
    "sampled_MRI-SWIFT_RESIZED_Stage_JJ_Open_VQA_Test_FULL_NII.json"
    "sampled_MRI-SWIFT_RESIZED_Stage_JJ_Report_Test_FULL_NII.json"
    "sampled_MRI-SWIFT_RESIZED_Stage_JJ_GroundCaption_Test_FULL_NII.json"
)

NUM_CONFIG="img${SELECT_IMAGE_NUM}_his${SELECT_HISTO_NUM}_vid${SELECT_VIDEO_NUM}"
MODEL_TAG=$(echo "${MODEL}" | sed 's|/|_|g; s|^_||')
RUN_DIR="${OUTPUT_DIR}/${MODEL_TAG}_${NUM_CONFIG}"
mkdir -p "${RUN_DIR}"

echo "Model     : ${MODEL}"
echo "Bench dir : ${BENCH_DIR}"
echo "Output dir: ${RUN_DIR}"
echo "GPUs      : ${NPROC_PER_NODE}"

for DATASET in "${DATASETS[@]}"; do
    RESPATH="${RUN_DIR}/${DATASET}"
    if [ -e "${RESPATH}" ]; then
        echo "[skip] ${DATASET} (already exists)"
        continue
    fi
    echo "[run ] ${DATASET}"
    swift infer \
        --model "${MODEL}" \
        --infer_backend transformers \
        --external_plugins "${PLUGIN_DIR}" \
        --template breastgpt \
        --result_path "${RESPATH}" \
        --val_dataset "${BENCH_DIR}/${DATASET}" \
        --use_hf false \
        --attn_impl flash_attn \
        --remove_unused_columns false \
        --torch_dtype bfloat16 \
        --max_batch_size 16 \
        --max_new_tokens "${MAXTOKENS}"
done

echo "Predictions written to: ${RUN_DIR}"

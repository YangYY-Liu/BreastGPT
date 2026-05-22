#!/bin/bash

set -e
MODEL_NAME=$1
EXP_NAME=$2
PER_DEVICE_BS=$3  
ACC_STEPS=$4 
TrainType=${5:-"full"}
SAVESTEPS=${6:-500}
USETXT=${7:-false}

# 多机多卡 ------------
echo "MODEL_NAME $MODEL_NAME | EXP_NAME $EXP_NAME | PER_DEVICE_BS $PER_DEVICE_BS | ACC_STEPS $ACC_STEPS | TrainType $TrainType | SAVESTEPS $SAVESTEPS | GPUs=$NPROC_PER_NODE"
export NCCL_SOCKET_IFNAME=dn_bond
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

# NCCL 日志等级，方便排查 NCCL 组网问题。
export TORCH_NCCL_TRACE_BUFFER_SIZE=10485760
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# export NCCL_BLOCKING_WAIT=1  # 开启阻塞等待，方便定位哪张卡挂了
# export NCCL_TIMEOUT=3600000  # 将超时增加到 1 小时

# 4. 优化 RoCE 通信参数
export NCCL_IB_TIMEOUT=22
export NCCL_IB_TC=106
export NCCL_IB_RETRY_CNT=7

# 4. 启用多轨优化，让 8 张卡充分利用 8 个网口
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_DEBUG=WARN
# 获取 MLFlow 或平台自动分配的变量
export NNODES=$WORLD_SIZE             # 总机器数
export NODE_RANK=$RANK                # 当前机器编号
export NPROC_PER_NODE=$NPROC_PER_NODE  # 单机卡数
export MASTER_ADDR=$MASTER_ADDR       # 主节点 IP
export MASTER_PORT=$MASTER_PORT              # 通讯端口

export PYTORCH_ALLOC_CONF=expandable_segments:True 
# ---------- BreastGPT 特定 ----------
export MAX_H=384
export MAX_W=384
export MAX_D=48

# ---------- Qwen 特定 ----------
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MIN_TOKEN_NUM=1
export IMAGE_MIN_TOKEN_NUM=1
# export VIDEO_MAX_TOKEN_NUM=128
export HF_ENDPOINT=https://hf-mirror.com

cur_dir=$(dirname "$0")
cur_dir=$(realpath "$cur_dir")
export MODELSCOPE_CACHE=~/MODELSCOPE_CACHE
export OUT_DIR="./outputs/${EXP_NAME}_${TrainType}"

swift sft \
    --model "/nas/yangye.ly/breastGPT/model/architectures/saved_models/$MODEL_NAME" \
    --model_type breastgpt \
    --template breastgpt \
    --external_plugins '/nas/yangye.ly/breastGPT/model' \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --train_type "${TrainType}" \
    --torch_dtype bfloat16 \
    --output_dir "$OUT_DIR" \
    --dataset "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/BUS/SWIFT_EXTREME_RESIZED_Stage_Report_VQA_Train_DownSample.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/BUS/SWIFT_EXTREME_RESIZED_Stage_Lesion_Closed_VQA_Train_DownSample.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/BUS/SWIFT_EXTREME_RESIZED_Stage_Lesion_Open_VQA_Train_DownSample.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/BUS/SWIFT_EXTREME_RESIZED_Stage_GroundCaption_Train_DownSample.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/CT/SWIFT_EXTREME_RESIZED_Screen_Closed_VQA_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/CT/SWIFT_EXTREME_RESIZED_Screen_Open_VQA_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/CT/SWIFT_EXTREME_RESIZED_Screen_GroundCaption_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/Histopathology/SWIFT_EXTREME_RESIZED_Slide_Caption_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/Histopathology/SWIFT_EXTREME_RESIZED_Slide_VQA_Open_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/Histopathology/SWIFT_EXTREME_RESIZED_Slide_VQA_Closed_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/Mammography/SWIFT_EXTREME_RESIZED_Stage_Mammo_Closed_VQA_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/Mammography/SWIFT_EXTREME_RESIZED_Stage_Mammo_GroundCaption_Train.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/MRI/SWIFT_EXTREME_RESIZED_Stage_JJ_Closed_VQA_Train_FULL.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/MRI/SWIFT_EXTREME_RESIZED_Stage_JJ_Open_VQA_Train_FULL.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/MRI/SWIFT_EXTREME_RESIZED_Stage_JJ_Report_Train_FULL.json" \
        "/nas/yangye.ly/breastGPT/datasets/VQA_SYS/MRI/SWIFT_EXTREME_RESIZED_Stage_JJ_GroundCaption_Train_FULL.json" \
    --split_dataset_ratio 0.001 \
    --use_hf false \
    --gradient_checkpointing true \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --per_device_eval_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${ACC_STEPS}" \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --attn_impl flash_attn \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 8 \
    --save_strategy steps \
    --save_steps "${SAVESTEPS}" \
    --eval_steps "${SAVESTEPS}" \
    --logging_steps 10 \
    --max_length 10000 \
    --deepspeed zero2
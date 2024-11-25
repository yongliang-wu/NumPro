#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export BATCH_SIZE=1
export EPOCH=3
export GRADIENT_ACCU_STEPS=16
export MASTER_PORT=29502
export CPUS_PER_TASK=24
export QUOTA=reserved

export BASE_LR=1e-4
export VIT_LR=2e-6
export LORA_R=64
export LORA_ALPHA=128
export WARMUP_RATIO=0.05
export DATA_PATH=temporal_annotations/train_sft.json
export SAVE_PATH=longva_7b_dpo_NumPro_FT
export ANET_PATH=data/anet/videos_0.5FPS_number_red_40_br
export DIDEO_PATH=data/didemo/videos_0.5FPS_number_red_40_br
export INTERNVID_PATH=data/internvid/videos_0.5FPS_number_red_40_br

PYTHONPATH="$(dirname \$0)/..":$PYTHONPATH \
deepspeed --num_nodes $NNODES --num_gpus $GPUS_PER_NODE --master_addr localhost --master_port ${MASTER_PORT} \
    longva/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path  ./pretrained/LongVA-7B-DPO \
    --version qwen_1_5 \
    --data_path ${DATA_PATH} \
    --anet_video_folder ${ANET_PATH} \
    --didemo_video_folder ${DIDEO_PATH} \
    --internvid_video_folder ${INTERNVID_PATH} \
    --videobackend all \
    --vision_tower ./pretrained/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --lora_enable True \
    --unfreeze_mm_vision_tower False \
    --mm_vision_tower_lr ${VIT_LR} \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type unires \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name $SAVE_PATH
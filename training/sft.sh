#!/usr/bin/env bash
# Stage 1: curriculum-guided SFT of Qwen3-VL-4B-Instruct with LoRA.
#
# Needs   models/pretrained/Qwen3-VL-4B-Instruct
#         data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl (built by data_pipeline/; the
#         image paths inside it are relative to the repository root)
# Writes  models/training_runs/SFT/<run>/checkpoint-*  (merge_lora.sh sft picks the newest)
#
# The curriculum is the sample order of the JSONL (HBB -> OBB -> HBB-to-OBB CoT), so both
# --dataset_shuffle and --train_dataloader_shuffle stay false; shuffling would turn this
# run into the "mixed" ablation arm.  Paper setting: LoRA rank 16 / alpha 32 on all linear
# layers, vision tower and merger frozen (--freeze_vit / --freeze_aligner), lr 1e-4,
# 1 epoch, max length 4096, 2 GPUs x batch 1 x 2 accumulation steps.
# IMAGE_MAX_TOKEN_NUM (image token budget) and QWENVL_BBOX_FORMAT=new (norm1000 grounding
# format) must match rollout and evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."   # every path below is repository-root relative

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MASTER_PORT=29501 \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,2 \
swift sft \
    --model models/pretrained/Qwen3-VL-4B-Instruct \
    --dataset 'data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl' \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir models/training_runs/SFT \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --report_to swanlab \
    --swanlab_project 'SFT' \
    --swanlab_exp_name 'GeoBox-R1-SFT' \
    --swanlab_mode cloud

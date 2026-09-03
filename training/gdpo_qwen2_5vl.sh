#!/usr/bin/env bash
# GDPO on a Qwen2.5-VL policy: the gdpo.sh recipe on a different backbone.
#
# Needs   models/checkpoints/GeoBox-R1-SFT-Qwen2_5VL  (a Qwen2.5-VL SFT checkpoint merged
#         with bash training/merge_lora.sh sft after overriding BASE_MODEL/OUTPUT)
#         data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl
#         a running rollout server: bash training/rollout_vllm.sh qwen2_5  (port 7773)
# Writes  models/training_runs/GRPO-Qwen2_5VL/<run>/checkpoint-*
#         (bash training/merge_lora.sh gdpo-qwen2_5 picks the newest)
#
# Qwen2.5-VL emits pixel coordinates of its smart-resized input rather than norm1000, so
# --norm_bbox none stops ms-swift from rewriting the boxes and the rewards come from
# training/reward_plugin_qwen2_5vl.py, which maps each prediction back to original-image
# pixels before scoring.  All other hyperparameters are those of gdpo.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."   # every path below is repository-root relative

NCCL_P2P_DISABLE=1 \
MASTER_PORT=29502 \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=4 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
SWANLAB_LOG_DIR='swanlog' \
swift rlhf \
    --rlhf_type grpo \
    --model models/checkpoints/GeoBox-R1-SFT-Qwen2_5VL \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --norm_bbox none \
    --dataset 'data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl' \
    --external_plugins training/reward_plugin_qwen2_5vl.py \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --attn_impl flash_attention_2 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 7773 \
    --vllm_server_group_port 51246 \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 256 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir models/training_runs/GRPO-Qwen2_5VL \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --deepspeed zero2 \
    --log_completions true \
    --beta 0.02 \
    --num_iterations 1 \
    --report_to swanlab \
    --swanlab_project 'RL' \
    --swanlab_exp_name 'GRPO@[Qwen2.5-VL 20%Data IoU(0.5) + Adaptive_WD(0.5) GDPO C_TAU=8 max_grad_norm=1]' \
    --max_grad_norm 1 \
    --swanlab_mode cloud \
    --reward_funcs external_vg_iou_qwen2_5 external_vg_wd_adaptive_qwen2_5 \
    --reward_weights 0.5 0.5 \
    --scale_rewards gdpo

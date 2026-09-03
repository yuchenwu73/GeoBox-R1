#!/usr/bin/env bash
# Stage 2: GDPO on OBB grounding with the rotated-IoU and adaptive Wasserstein rewards.
#
# Needs   models/checkpoints/GeoBox-R1-SFT           (bash training/merge_lora.sh sft)
#         data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl  (20% OBB subset from data_pipeline/;
#         image paths relative to the repository root)
#         a running rollout server: bash training/rollout_vllm.sh  (port 7772, must match
#         --vllm_server_port below)
# Writes  models/training_runs/GRPO/<run>/checkpoint-*  (merge_lora.sh gdpo picks the newest)
#
# Rewards live in training/reward_plugin_qwen3vl.py and are mixed 0.5 / 0.5.
# --scale_rewards gdpo normalises the advantage of each reward function within its group
# separately before the weighted sum, which is what turns GRPO into GDPO.
# --num_generations 8 is the group size, --beta 0.02 the KL penalty towards the SFT policy,
# temperature 0.9 and --max_completion_length 256 bound the rollouts.
# Paper setting: lr 5e-6, 1 epoch, LoRA rank 16 / alpha 32, 4 GPUs x batch 8 x 4 steps.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."   # every path below is repository-root relative

NCCL_P2P_DISABLE=1 \
MASTER_PORT=29501 \
QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=4 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
SWANLAB_LOG_DIR='swanlog' \
swift rlhf \
    --rlhf_type grpo \
    --model models/checkpoints/GeoBox-R1-SFT \
    --model_type qwen3_vl \
    --template qwen3_vl \
    --dataset 'data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl' \
    --external_plugins training/reward_plugin_qwen3vl.py \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --attn_impl flash_attention_2 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 7772 \
    --vllm_server_group_port 51226 \
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
    --output_dir models/training_runs/GRPO \
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
    --swanlab_exp_name 'GRPO@[20%Data IoU(0.5) + Adaptive_WD(0.5) GDPO C_TAU=8 beta=0.02 max_grad_norm=1]' \
    --max_grad_norm 1 \
    --swanlab_mode cloud \
    --reward_funcs external_vg_iou external_vg_wd_adaptive \
    --reward_weights 0.5 0.5 \
    --scale_rewards gdpo

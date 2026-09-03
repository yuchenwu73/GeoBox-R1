#!/usr/bin/env bash
# Fixed-tau control for gdpo.sh: identical recipe, but the Wasserstein reward keeps a
# constant tau (external_vg_wd, tau = C_TAU = 8 in norm1000 units) instead of scaling it
# with the target size.  Compare against gdpo.sh to see why the adaptive tau matters.
#
# Needs   the same checkpoint, data and rollout server (port 7772) as gdpo.sh; since both
#         talk to the same server, run the two experiments one after the other.
# Writes  models/training_runs/GRPO-fixed-tau/<run>/checkpoint-*
#         (merge with ADAPTER_ROOT=models/training_runs/GRPO-fixed-tau bash training/merge_lora.sh gdpo)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."   # every path below is repository-root relative

MASTER_PORT=29510 \
NCCL_P2P_DISABLE=1 \
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
    --vllm_server_group_port 51236 \
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
    --output_dir models/training_runs/GRPO-fixed-tau \
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
    --swanlab_exp_name 'GRPO@[20%Data IoU(0.5) + WD(0.5) GDPO tau=8]' \
    --max_grad_norm 1 \
    --swanlab_mode cloud \
    --reward_funcs external_vg_iou external_vg_wd \
    --reward_weights 0.5 0.5 \
    --scale_rewards gdpo

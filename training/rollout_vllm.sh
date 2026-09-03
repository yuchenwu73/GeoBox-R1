#!/usr/bin/env bash
# vLLM rollout server for stage 2.  The GDPO trainer sends prompts here and scores the
# sampled completions, so this must be running (on its own GPU) before gdpo*.sh starts
# and stay up for the whole run.
#
#   bash training/rollout_vllm.sh           serves models/checkpoints/GeoBox-R1-SFT on 7772
#                                           (gdpo.sh and gdpo_fixed_tau.sh)
#   bash training/rollout_vllm.sh qwen2_5   serves the Qwen2.5-VL SFT checkpoint on 7773
#                                           (gdpo_qwen2_5vl.sh)
#
# IMAGE_MAX_TOKEN_NUM / QWENVL_BBOX_FORMAT and --norm_bbox must match the trainer, otherwise
# the sampled boxes live in a different coordinate space than the rewards expect.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."   # every path below is repository-root relative

case "${1:-qwen3}" in
  qwen3)
    MODEL='models/checkpoints/GeoBox-R1-SFT'
    MODEL_TYPE='qwen3_vl'
    PORT=7772
    NORM_ARGS=()
    ;;
  qwen2_5)
    MODEL='models/checkpoints/GeoBox-R1-SFT-Qwen2_5VL'
    MODEL_TYPE='qwen2_5_vl'
    PORT=7773
    # Qwen2.5-VL emits resized-image pixel coordinates rather than norm1000 coordinates.
    NORM_ARGS=(--norm_bbox none)
    ;;
  *)
    echo "usage: bash training/rollout_vllm.sh [qwen3|qwen2_5]" >&2
    exit 2
    ;;
esac

QWENVL_BBOX_FORMAT='new' \
IMAGE_MAX_TOKEN_NUM=1024 \
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model "$MODEL" \
    --model_type "$MODEL_TYPE" \
    "${NORM_ARGS[@]}" \
    --vllm_gpu_memory_utilization 0.85 \
    --attn_impl flash_attention_2 \
    --vllm_max_model_len 2048 \
    --port "$PORT"

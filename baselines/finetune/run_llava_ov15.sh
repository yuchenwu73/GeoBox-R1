#!/usr/bin/env bash
# "LLaVA-OV-1.5 (SFT)" row of the paper: LoRA fine-tuning of
# lmms-lab/LLaVA-OneVision-1.5-4B-Instruct on refGeo (the 8B checkpoint works unchanged).
#
# Data    baselines/finetune/data/refgeo_norm1.jsonl (OV-1.5's native 0-1 float coordinates),
#         built from data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl on first run.
# Recipe  LoRA r16 / alpha 32 / dropout 0.05 on the Qwen3 language model only (the Rice vision
#         tower uses fused qkv names and is never matched, so it stays frozen), batch 1 x
#         grad-accum 8 x 2 GPUs = effective batch 16, lr 1e-4, cosine, 1 epoch, bf16,
#         flash-attention-2, max length 4096. MAX_PIXELS=802816 caps the native-resolution
#         processor at about 1024 image tokens, the same budget as GeoBox-R1.
# Time    about 16.5 h on 2 GPUs in the original 4B run.
# Eval    python baselines/evaluate/eval_hf.py --model_path "$MODEL_PATH" --adapter_dir "$OUTPUT_DIR" \
#             --task hbb --prompt_mode trained --coord_space norm1 --max_pixels 802816   (and --task obb)
#
# Usage   CUDA_DEVICES=0,1 bash baselines/finetune/run_llava_ov15.sh
#         Every knob below can be overridden from the environment; MAX_STEPS=20 gives a smoke test.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${REPO_ROOT}"
prepare_data

MODEL_PATH="${MODEL_PATH:-models/pretrained/LLaVA-OneVision-1.5-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-models/adapters/$(basename "${MODEL_PATH}")_refgeo_lora}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT="$(find_free_port "${DEFAULT_MASTER_PORT:-29900}")"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
torchrun_cmd "${PYTHON_BIN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_hf_lora.py" \
    --model "${MODEL_PATH}" \
    --dataset "${HF_DATA_NORM1}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --gradient-accumulation "${GRAD_ACC:-8}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --max-pixels "${MAX_PIXELS:-802816}" \
    --max-steps "${MAX_STEPS:--1}" \
    --bits "${BITS:-16}" \
    --attn-implementation "${ATTN_IMPL:-flash_attention_2}" \
    --workers "${WORKERS:-4}" \
    --report-to "${REPORT_TO:-none}"

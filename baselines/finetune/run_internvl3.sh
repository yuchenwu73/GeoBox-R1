#!/usr/bin/env bash
# "InternVL3 (SFT)" rows of the paper (2B and 8B): LoRA fine-tuning of the Hugging Face
# checkpoints OpenGVLab/InternVL3-2B-hf and OpenGVLab/InternVL3-8B-hf on refGeo.
#
# Data    baselines/finetune/data/refgeo_norm1000.jsonl (InternVL's native [0, 1000] grid),
#         built from data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl on first run.
# Recipe  LoRA r16 / alpha 32 / dropout 0.05 on the language model (vision tower frozen),
#         batch 1 x grad-accum 8 x 2 GPUs = effective batch 16, lr 1e-4, cosine, 1 epoch,
#         bf16, flash-attention-2, max length 4096, dynamic tiling (--crop-to-patches).
# Time    about 9.5 h (2B) and 15.6 h (8B) on 2 GPUs in the original runs.
# Eval    python baselines/evaluate/eval_hf.py --model_path "$MODEL_PATH" --adapter_dir "$OUTPUT_DIR" \
#             --task hbb --prompt_mode trained --crop_to_patches      (and --task obb)
#
# Usage   MODEL_PATH=models/pretrained/InternVL3-8B-hf CUDA_DEVICES=0,1 bash baselines/finetune/run_internvl3.sh
#         Every knob below can be overridden from the environment; MAX_STEPS=20 gives a smoke test.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${REPO_ROOT}"
prepare_data

MODEL_PATH="${MODEL_PATH:-models/pretrained/InternVL3-8B-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-models/adapters/$(basename "${MODEL_PATH}")_refgeo_lora}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT="$(find_free_port "${DEFAULT_MASTER_PORT:-29600}")"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
torchrun_cmd "${PYTHON_BIN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_hf_lora.py" \
    --model "${MODEL_PATH}" \
    --dataset "${HF_DATA_NORM1000}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --gradient-accumulation "${GRAD_ACC:-8}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --max-steps "${MAX_STEPS:--1}" \
    --bits "${BITS:-16}" \
    --attn-implementation "${ATTN_IMPL:-flash_attention_2}" \
    --crop-to-patches \
    --workers "${WORKERS:-4}" \
    --report-to "${REPORT_TO:-none}"

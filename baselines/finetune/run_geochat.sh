#!/usr/bin/env bash
# "GeoChat (SFT)" row of the paper: LoRA fine-tuning of geochat-7B on refGeo with the
# official GeoChat training code (geochat/train/train.py, ZeRO-2).
#
# Data    baselines/finetune/data/refgeo_geochat_native_llava.json, GeoChat's own
#         {<x1><y1><x2><y2>|<angle>} format on a 0-100 grid (see prepare_geochat_data.py),
#         built from data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl on first run.
# Recipe  LoRA r16 / alpha 32 / dropout 0.05, batch 1 x grad-accum 8 x 2 GPUs = effective
#         batch 16, lr 1e-4, cosine, warmup 0.03, 1 epoch, bf16, max length 4096, CLIP-L/14-336
#         vision tower with the official mlp2x_gelu projector.
# Time    about 20.8 h on 2 GPUs in the original run.
# Eval    python baselines/evaluate/eval_geochat.py --task hbb --model_path "$MODEL_PATH" \
#             --adapter_path "$OUTPUT_DIR"                                   (and --task obb)
#
# Needs   GEOCHAT_REPO   checkout of https://github.com/mbzuai-oryx/GeoChat (for the package
#                        and scripts/zero2.json)
#         GEOCHAT_PYTHON interpreter of the GeoChat environment (transformers 4.31, peft 0.4,
#                        deepspeed). Its old bitsandbytes/deepspeed builds may need CUDA_HOME
#                        pointed at that conda env before launching.
#         PYTHON_BIN     interpreter with PIL/cv2/shapely for data preparation (default python)
#
# Usage   GEOCHAT_REPO=/path/to/GeoChat GEOCHAT_PYTHON=/path/to/envs/geochat/bin/python \
#             CUDA_DEVICES=0,1 bash baselines/finetune/run_geochat.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${REPO_ROOT}"
GEOCHAT_REPO="${GEOCHAT_REPO:?set GEOCHAT_REPO to the official GeoChat checkout}"
prepare_geochat_data

GEOCHAT_PYTHON="${GEOCHAT_PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-models/pretrained/geochat-7B}"
VISION_TOWER="${VISION_TOWER:-models/pretrained/clip-vit-large-patch14-336}"
OUTPUT_DIR="${OUTPUT_DIR:-models/adapters/geochat_7b_refgeo_lora}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NPROC="${NPROC:-2}"
MASTER_PORT="$(find_free_port "${DEFAULT_MASTER_PORT:-29800}")"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${GEOCHAT_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
torchrun_cmd "${GEOCHAT_PYTHON}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_geochat_lora.py" \
    --deepspeed "${GEOCHAT_REPO}/scripts/zero2.json" \
    --lora_enable True \
    --lora_r "${LORA_RANK:-16}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version v1 \
    --data_path "${GEOCHAT_DATA}" \
    --image_folder "${IMAGE_ROOT}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS:-1}" \
    --per_device_train_batch_size "${BATCH_SIZE:-1}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC:-8}" \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-500}" \
    --save_total_limit 2 \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers "${WORKERS:-4}" \
    --report_to "${REPORT_TO:-none}"

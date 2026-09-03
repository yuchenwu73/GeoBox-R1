#!/usr/bin/env bash
# Merge a LoRA adapter into its base model, producing a checkpoint that evaluation/, demo/
# and the next training stage can load directly.
#
#   bash training/merge_lora.sh sft            models/training_runs/SFT  -> models/checkpoints/GeoBox-R1-SFT
#   bash training/merge_lora.sh gdpo           models/training_runs/GRPO -> models/checkpoints/GeoBox-R1
#   bash training/merge_lora.sh gdpo-qwen2_5   the Qwen2.5-VL variant of the gdpo stage
#
# Without ADAPTER= the newest checkpoint below the stage's run directory is merged, so the
# script can follow a training run that produced several versioned sub-directories.
# Override any of BASE_MODEL=, ADAPTER_ROOT=, ADAPTER=, OUTPUT=, GPU=; relative overrides
# are resolved from the repository root, like every other path here.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."   # every path below is repository-root relative

STAGE="${1:-sft}"

resolve_latest_adapter() {
  # Accept either a checkpoint itself or a run directory containing several checkpoints,
  # and pick the most recently written adapter_config.json.
  local root="$1"
  local best=''
  local config
  local candidate

  if [[ -f "$root/adapter_config.json" ]]; then
    printf '%s\n' "$root"
    return 0
  fi

  while IFS= read -r -d '' config; do
    candidate="${config%/adapter_config.json}"
    if [[ -z "$best" || "$candidate" -nt "$best" ]]; then
      best="$candidate"
    fi
  done < <(find "$root" -mindepth 1 -maxdepth 3 -type f -name adapter_config.json -print0 2>/dev/null)

  if [[ -z "$best" ]]; then
    echo "no adapter checkpoint found under: $root" >&2
    return 1
  fi
  printf '%s\n' "$best"
}

case "$STAGE" in
  sft)
    # Stage 1 starts from the pretrained base.
    BASE_MODEL="${BASE_MODEL:-models/pretrained/Qwen3-VL-4B-Instruct}"
    ADAPTER_ROOT="${ADAPTER_ROOT:-models/training_runs/SFT}"
    OUTPUT="${OUTPUT:-models/checkpoints/GeoBox-R1-SFT}"
    MODEL_TYPE='qwen3_vl'
    TEMPLATE='qwen3_vl'
    ;;
  gdpo)
    # Stage 2 starts from the merged SFT checkpoint.
    BASE_MODEL="${BASE_MODEL:-models/checkpoints/GeoBox-R1-SFT}"
    ADAPTER_ROOT="${ADAPTER_ROOT:-models/training_runs/GRPO}"
    OUTPUT="${OUTPUT:-models/checkpoints/GeoBox-R1}"
    MODEL_TYPE='qwen3_vl'
    TEMPLATE='qwen3_vl'
    ;;
  gdpo-qwen2_5)
    # The Qwen2.5-VL variant keeps its own model type and template during export.
    BASE_MODEL="${BASE_MODEL:-models/checkpoints/GeoBox-R1-SFT-Qwen2_5VL}"
    ADAPTER_ROOT="${ADAPTER_ROOT:-models/training_runs/GRPO-Qwen2_5VL}"
    OUTPUT="${OUTPUT:-models/checkpoints/GeoBox-R1-Qwen2_5VL}"
    MODEL_TYPE='qwen2_5_vl'
    TEMPLATE='qwen2_5_vl'
    ;;
  *)
    echo "usage: bash training/merge_lora.sh [sft|gdpo|gdpo-qwen2_5]" >&2
    exit 1
    ;;
esac

ADAPTER="${ADAPTER:-$(resolve_latest_adapter "$ADAPTER_ROOT")}"

echo "base    : $BASE_MODEL"
echo "adapter : $ADAPTER"
echo "output  : $OUTPUT"

CUDA_VISIBLE_DEVICES="${GPU:-0}" swift export \
    --model "$BASE_MODEL" \
    --model_type "$MODEL_TYPE" \
    --template "$TEMPLATE" \
    --adapters "$ADAPTER" \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT"

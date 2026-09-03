<b>English</b> | <a href="README_zh.md">简体中文</a>

# Evaluation

Each script runs the full loop — load model, batch inference, parse boxes, compute metrics —
and reports **all three metrics in a single pass**:

| Script | Task | Metrics |
| --- | --- | --- |
| `evaluate_hbb.py` | Horizontal boxes, 7 evaluation sets | Acc@0.5 / Acc@0.7 / mIoU |
| `evaluate_obb.py` | Oriented boxes, 3 evaluation sets | Acc@0.5 / Acc@0.7 / mRIoU |

Rotated IoU is computed as a polygon intersection-over-union via shapely, so
`evaluate_obb.py` needs `pip install shapely`.

## Usage

```bash
# All evaluation sets
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_hbb.py \
    --model_path models/checkpoints/GeoBox-R1 --dataset all

# A single evaluation set
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_obb.py \
    --model_path models/checkpoints/GeoBox-R1 --dataset avvg_test

# Qwen2.5-VL checkpoints trained with norm_bbox=none
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_obb.py \
    --model_path <qwen2.5-vl-checkpoint> --model_type qwen2_5_vl \
    --coord_mode resized_absolute --dataset avvg_test
```

Every run creates a fresh timestamped directory containing:

- HBB results default to `eval_results/hbb`.
- OBB results default to `eval_results/obb`.

```text
predictions.jsonl   per-sample prediction, ground truth and IoU
summary.json        per-dataset and macro-average metrics
table*.md           markdown table ready to paste into the paper
```

## Arguments

| Flag | Meaning |
| --- | --- |
| `--model_path` | Base model |
| `--checkpoint_dir` | LoRA adapter applied on top of the base model; omit to evaluate the base model alone |
| `--model_type` | ms-swift model type. `qwen3_vl` by default; use `qwen2_5_vl` for Qwen2.5-VL baselines |
| `--dataset` | `all`, or a single split name |
| `--metainfo_dir` | Directory of test jsonl files, default `data/refGeo/metainfo` |
| `--image_dir` | Image root, default `data/refGeo/images`; glob patterns are supported |
| `--output_dir` | Result root, default `eval_results/hbb` or `eval_results/obb` |
| `--coord_mode` | Coordinate convention: `norm1000` (default, quantized to 0–1000), `absolute`, `resized_absolute` |
| `--batch_size` | Overrides the per-dataset default |
| `--attn_impl` | Attention backend: `flash_attn` (default, needs the `flash-attn` package) or `sdpa` |
| `--max_samples` | Only run the first N samples — useful as a smoke test |
| `--table_format` | `text` or `markdown` |
| `--model_name` | Name written into the result table |

Per-dataset batch sizes are preset by image size (60 for DIOR-RSVG at 800×800, 12 for the
larger AVVG images) and assume a free 40 GB GPU. Lower them with `--batch_size` on a smaller
or shared GPU. A batch that still fails (typically out of memory) is retried one sample at a
time; a sample that fails even then is kept as a zero with an `error` field in
`predictions.jsonl`, and the per-dataset summary prints a warning with the count, so the
metrics never silently drop samples.

## Coordinate conventions

`norm1000` matches training: all coordinates are quantized to `[0, 1000]`. When evaluating a
model that emits pixel coordinates of the original image, use `--coord_mode absolute`; for
pixel coordinates of the resized image, use `--coord_mode resized_absolute`. Picking the wrong
one depresses the metrics substantially. The Qwen2.5-VL baseline in this repository uses
`resized_absolute`.

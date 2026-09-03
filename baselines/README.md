<b>English</b> | <a href="README_zh.md">简体中文</a>

# Baselines

Evaluation and fine-tuning code for the models compared against in the paper.

## evaluate/ — baseline evaluation

Same protocol as the main scripts in `evaluation/`: Acc@0.5, Acc@0.7 and mIoU / mRIoU in one
pass over the 7 HBB and 3 OBB evaluation sets. `common.py` holds the shared loop (metainfo
loading, batching with per-sample retry, IoU / rotated IoU, result files); each script only
wraps one model family's inference interface, prompt and coordinate convention.

| Script | Paper rows | Model output that is parsed |
| --- | --- | --- |
| `eval_geochat.py` | GeoChat (official weights); GeoChat (SFT) with `--adapter_path` | `{<x1><y1><x2><y2>\|<angle>}` on a 0–100 grid |
| `eval_geoground.py --task hbb\|obb` | GeoGround | HBB `<box>[[x1,y1,x2,y2]]</box>` (norm1000); OBB `<obb>[[cx,cy,w,h,θ]]</obb>` (norm100) |
| `eval_internvl.py` | InternVL3-2B / 8B, zero-shot | `<box>[[x1,y1,x2,y2]]</box>` (norm1000) |
| `eval_hf.py` | LLaVA-OV-1.5-4B / 8B zero-shot; InternVL3-2B / 8B (SFT); LLaVA-OV-1.5-4B (SFT) | the JSON format of the main model, coordinate space set by `--coord_space` |
| `eval_lhrs.py` | LHRS-Bot, LHRS-Bot-Nova (HBB only) | `[x1,y1,x2,y2]` in 0–1 |

Qwen2.5-VL / Qwen3-VL zero-shot and Qwen2.5-VL (SFT) need no dedicated script — use the main
evaluation scripts with `--model_type qwen2_5_vl` or `qwen3_vl`.

The OBB protocol is strict: only a genuinely oriented output (an angle or four corners) scores;
a horizontal box returned to an OBB prompt counts as 0. This is what separates models with and
without OBB capability.

```bash
# GeoChat — official weights, then the refGeo LoRA from finetune/
python baselines/evaluate/eval_geochat.py --task hbb --model_path models/pretrained/geochat-7B \
    --output_dir eval_results/baselines/geochat
python baselines/evaluate/eval_geochat.py --task obb --model_path models/pretrained/geochat-7B \
    --adapter_path models/adapters/geochat_7b_refgeo_lora --output_dir eval_results/baselines/geochat_sft

# GeoGround
python baselines/evaluate/eval_geoground.py --task hbb --output_dir eval_results/baselines/geoground
python baselines/evaluate/eval_geoground.py --task obb --output_dir eval_results/baselines/geoground

# InternVL3 zero-shot (official GitHub-format weights, ms-swift engine)
python baselines/evaluate/eval_internvl.py --task hbb --model_path models/pretrained/InternVL3-8B \
    --output_dir eval_results/baselines/internvl3_8b

# LLaVA-OV-1.5 zero-shot, and the two fine-tuned families (adapters from finetune/)
python baselines/evaluate/eval_hf.py --model_path models/pretrained/LLaVA-OneVision-1.5-4B-Instruct \
    --task hbb --prompt_mode zeroshot --zeroshot_style qwenvl
python baselines/evaluate/eval_hf.py --model_path models/pretrained/InternVL3-8B-hf \
    --adapter_dir models/adapters/InternVL3-8B-hf_refgeo_lora --task hbb --prompt_mode trained --crop_to_patches
python baselines/evaluate/eval_hf.py --model_path models/pretrained/LLaVA-OneVision-1.5-4B-Instruct \
    --adapter_dir models/adapters/LLaVA-OneVision-1.5-4B-Instruct_refgeo_lora \
    --task hbb --prompt_mode trained --coord_space norm1 --max_pixels 802816

# LHRS-Bot-Nova
python baselines/evaluate/eval_lhrs.py --model lhrs-nova --output_dir eval_results/baselines/lhrs_nova
```

Run everything from the repository root; `--task obb` works the same way. Every run writes
`<split>_<task>_predictions.jsonl` (raw model output, parsed box, ground truth, IoU), a
`summary_<task>.json` and a markdown table into a timestamped directory under `--output_dir`.

Each baseline needs its official inference environment: GeoChat the `geochat` package
(transformers 4.31), GeoGround the `llava` package shipped with its repository, LHRS the `lhrs`
package. `eval_internvl.py` runs in the main ms-swift environment, `eval_hf.py` needs only
transformers and peft.

## finetune/ — baseline fine-tuning

Produces the "(SFT)" rows: every baseline is trained on the same curriculum SFT set as GeoBox-R1
(`data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl`, built by [`data_pipeline/`](../data_pipeline/))
with the same recipe — LoRA rank 16 / alpha 32, learning rate 1e-4, 1 epoch, effective batch 16
on 2 GPUs. Only the coordinate text is rewritten into each backbone's native convention.

```text
prepare_hf_data.py       → refgeo_norm1000.jsonl (InternVL3, 0–1000 grid) and refgeo_norm1.jsonl (LLaVA-OV-1.5, 0–1 floats)
prepare_geochat_data.py  → refgeo_geochat_native_llava.json ({<x1><y1><x2><y2>|<θ>} on a 0–100 grid)
train_hf_lora.py         LoRA SFT for Hugging Face checkpoints (Transformers + PEFT)
train_geochat_lora.py    thin wrapper around the official GeoChat trainer
run_internvl3.sh         launchers with the paper hyperparameters; they build their data file on first use
run_llava_ov15.sh
run_geochat.sh
common.sh                shared paths and helpers
```

```bash
MODEL_PATH=models/pretrained/InternVL3-8B-hf CUDA_DEVICES=0,1 bash baselines/finetune/run_internvl3.sh
MODEL_PATH=models/pretrained/LLaVA-OneVision-1.5-4B-Instruct CUDA_DEVICES=0,1 bash baselines/finetune/run_llava_ov15.sh
GEOCHAT_REPO=/path/to/GeoChat GEOCHAT_PYTHON=/path/to/envs/geochat/bin/python \
    CUDA_DEVICES=0,1 bash baselines/finetune/run_geochat.sh
```

Every setting is an environment variable (`BATCH_SIZE`, `GRAD_ACC`, `LEARNING_RATE`,
`OUTPUT_DIR`, ...; `MAX_STEPS=20` gives a smoke test). Adapters land in
`models/adapters/<checkpoint>_refgeo_lora`; evaluate them with the `eval_hf.py` /
`eval_geochat.py --adapter_path` commands above. The original runs took about 9.5 h (InternVL3-2B),
15.6 h (InternVL3-8B), 16.5 h (LLaVA-OV-1.5-4B) and 20.8 h (GeoChat-7B) on 2 GPUs.

The HF trainer's dependencies are listed in `finetune/requirements.txt` and are independent of
the main environment; GeoChat trains in its own environment (transformers 4.31, peft 0.4,
deepspeed) with the official repository on `PYTHONPATH`.

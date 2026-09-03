<div align="center">

<b>English</b> | <a href="README_zh.md">简体中文</a>

# GeoBox-R1: Curriculum-Guided SFT and Geometric RL for Unified Box-Level Remote Sensing Visual Grounding

Chenxi Lan\*, Yuchen Wu\*, Minghang Zhou, Tianyu Li, Zhihao Qiu, Guoqing Wang<sup>†</sup>

University of Electronic Science and Technology of China

<sup>\*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author

*Under review at AAAI 2027*

[![Project Page](https://img.shields.io/badge/Project-Page-1d4ed8)](https://yuchenwu73.github.io/geobox-r1/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-GeoBox--R1-ffc107)](https://huggingface.co/yuchenwu73/GeoBox-R1)

</div>

---

## Overview

Remote sensing visual grounding (RSVG) localizes the object referred to by a natural-language
expression in aerial or satellite imagery. Multimodal large language models (MLLMs) make
*unified* box-level grounding feasible — one model, both horizontal (HBB) and oriented (OBB)
boxes — but two problems stand in the way:

- General-purpose MLLMs **do not reliably produce OBBs**. Prompted explicitly for an oriented
  box, they often fall back to a horizontal one that ignores target orientation.
- Training a single model for both tasks is hard, because HBB and OBB demand **different levels
  of semantic grounding and geometric precision**.

**GeoBox-R1** addresses both with a two-stage recipe on top of Qwen3-VL-4B-Instruct:

| Stage | What it does |
| --- | --- |
| **1. Curriculum-guided SFT** | Presents training data easy-to-hard — HBB grounding → OBB grounding → HBB-to-OBB Chain-of-Thought — instead of shuffling. Establishes coarse-to-fine grounding capability in a single run. |
| **2. Geometric RL (GDPO)** | Group Reward-Decoupled Normalization Policy Optimization, driven by two rule-based geometric rewards: **Rotated IoU** and an **adaptive Wasserstein distance** that stays informative when RIoU saturates near zero. No learned reward model. |

With **4B parameters**, GeoBox-R1 obtains the best macro average on every metric over 7 HBB
and 3 OBB benchmarks, ahead of the 7B–8B state of the art, with the largest margins at the
stricter Acc@0.7 threshold. On OBB it leads every metric on every set. On HBB the per-benchmark
picture is complementary: GeoGround leads most metrics on DIOR-RSVG and GeoChat\*, GeoBox-R1
leads every metric on RSVG and AVVG and both accuracy thresholds on VRSBench\*.

## Results

Macro averages over all evaluation sets, with the strongest baselines of each group; the
complete tables with every baseline are on the [project page](https://yuchenwu73.github.io/geobox-r1/).

**HBB grounding** (7 evaluation sets)

| Model | Params | Acc@0.5 | Acc@0.7 | mIoU |
| --- | :-: | :-: | :-: | :-: |
| Qwen3-VL | 8B | 43.70 | 28.53 | 39.19 |
| Qwen2.5-VL (SFT) | 3B | 49.43 | 32.57 | 43.08 |
| GeoGround | 7B | 52.35 | 35.02 | 46.80 |
| InternVL3 (SFT) | 8B | 55.10 | 37.33 | 47.78 |
| GeoBox-R1 (SFT) | 4B | 57.17 | 40.33 | 48.82 |
| **GeoBox-R1 (SFT + GDPO)** | **4B** | **58.78** | **42.22** | **50.39** |

**OBB grounding** (3 evaluation sets)

| Model | Params | Acc@0.5 | Acc@0.7 | mRIoU |
| --- | :-: | :-: | :-: | :-: |
| Qwen2.5-VL (SFT) | 3B | 35.72 | 17.61 | 31.86 |
| InternVL3 (SFT) | 8B | 35.74 | 17.05 | 32.58 |
| GeoGround | 7B | 41.96 | 19.81 | 36.96 |
| GeoBox-R1 (SFT) | 4B | 43.11 | 24.40 | 36.85 |
| **GeoBox-R1 (SFT + GDPO)** | **4B** | **47.32** | **27.55** | **39.85** |

GDPO trains on OBB samples only, yet lifts **both** tasks — +1.61 HBB and +4.21 OBB Acc@0.5
over the SFT checkpoint.

**GeoBox-R1 (SFT + GDPO) per evaluation set** — the numbers `evaluation/evaluate_hbb.py` and
`evaluate_obb.py` print for the released checkpoint.

| Evaluation set | Task | Acc@0.5 | Acc@0.7 | mIoU / mRIoU |
| --- | :-: | :-: | :-: | :-: |
| DIOR-RSVG test | HBB | 76.61 | 64.78 | 67.50 |
| DIOR-RSVG val | HBB | 75.11 | 63.13 | 66.57 |
| RSVG test | HBB | 51.26 | 32.60 | 42.99 |
| RSVG val | HBB | 48.38 | 31.97 | 42.00 |
| GeoChat\* | HBB | 61.13 | 35.49 | 51.25 |
| VRSBench\* | HBB | 66.84 | 41.16 | 55.74 |
| AVVG | HBB | 32.14 | 26.39 | 26.68 |
| GeoChat\* | OBB | 60.56 | 35.19 | 48.92 |
| VRSBench\* | OBB | 56.61 | 30.55 | 49.43 |
| AVVG | OBB | 24.79 | 16.89 | 21.18 |

\* The refGeo re-splits of GeoChat and VRSBench (overlaps with DIOR-RSVG val/test images removed),
as in GeoGround. All baselines were re-evaluated with model-specific grounding prompts and parsers.

## Repository structure

```text
GeoBox-R1/
├── evaluation/          # Evaluation — Acc@0.5, Acc@0.7 and mIoU/mRIoU in one pass
│   ├── evaluate_hbb.py
│   └── evaluate_obb.py
├── training/            # SFT, GDPO, vLLM rollout, LoRA merging
│   ├── sft.sh
│   ├── gdpo.sh
│   ├── gdpo_fixed_tau.sh          # fixed-τ control
│   ├── gdpo_qwen2_5vl.sh          # same recipe on a Qwen2.5-VL backbone
│   ├── rollout_vllm.sh
│   ├── merge_lora.sh
│   ├── reward_plugin_qwen3vl.py   # RIoU + adaptive Wasserstein rewards
│   └── reward_plugin_qwen2_5vl.py # same rewards, Qwen2.5-VL coordinate convention
├── data_pipeline/       # Building the refGeo SFT / RL / ablation splits
├── visualization/       # Qualitative figures and τ analysis
├── demo/                # Gradio demo
├── baselines/           # Baseline evaluation and fine-tuning
└── tests/               # Unit tests, no GPU needed
```

Every directory carries its own README, in English and Chinese.

Two directories are expected but not tracked (see [Setup](#setup)):

```text
data/
├── refGeo/            metainfo/ and images/ — evaluation, and the source of the training sets
└── GeoBox-R1-Data/    sft/ and rl/ training sets, built locally by data_pipeline/
models/
├── pretrained/        Qwen3-VL-4B-Instruct
└── checkpoints/       GeoBox-R1 (released); GeoBox-R1-SFT is produced by Stage 1
```

## Setup

```bash
git clone https://github.com/yuchenwu73/GeoBox-R1.git
cd GeoBox-R1

conda create -n geobox-r1 python=3.10 -y
conda activate geobox-r1
pip install -r requirements.txt
# Flash-Attention 2 is used by every launcher and evaluation script. Pick the wheel that
# matches your torch / CUDA / Python from https://github.com/Dao-AILab/flash-attention/releases;
# this one matches the pinned torch 2.8 / CUDA 12 / Python 3.10:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

python -m unittest discover -s tests   # checks the scripts; no GPU or model needed
```

Training and evaluation run on [ms-swift](https://github.com/modelscope/ms-swift) 4.0.2.
The pinned set was verified to install into a fresh Python 3.10 environment, after which the
unit tests, the data pipeline, the evaluation scripts and the demo all ran. Without
`flash-attn`, evaluate with `--attn_impl sdpa` and run the demo with `GEOBOX_ATTN=sdpa`; the
training launchers rely on it for `--packing` / `--padding_free`.

### Data

The training sets are not distributed. Build them from the refGeo annotations with the
[data pipeline](data_pipeline/): it is deterministic (seed 42) and regenerates the files the
released model was trained on, laid out as:

```text
data/GeoBox-R1-Data/
├── sft/
│   ├── sft_curriculum_cot.jsonl      # main training set — curriculum order, with CoT
│   ├── sft_curriculum_no_cot.jsonl
│   ├── sft_mixed_cot.jsonl
│   └── sft_mixed_no_cot.jsonl        # the three above are the ablation arms
└── rl/
    └── rl_obb_20pct.jsonl            # 20% OBB subset for GDPO
```

Images come from the original benchmarks and are not redistributed here; follow
[refGeo](https://github.com/zytx121/GeoGround) to assemble them, then arrange as:

```text
data/refGeo/
├── metainfo/          # *.jsonl test splits
└── images/
    ├── DIOR-RSVG/
    ├── RSVG/
    ├── GeoChat/
    ├── VRSBench/
    └── AVVG/
```

Override the location with `--metainfo_dir` / `--image_dir`, or the `REFGEO_ROOT`
environment variable where supported. To rebuild the training splits from scratch, see
[`data_pipeline/`](data_pipeline/README.md).

### Models

| Checkpoint | Description |
| --- | --- |
| [`GeoBox-R1`](https://huggingface.co/yuchenwu73/GeoBox-R1) | Final model — curriculum-guided SFT + GDPO. Use this to reproduce the reported numbers. |

The Stage-1 checkpoint is not distributed: `training/sft.sh` followed by
`training/merge_lora.sh sft` produces `models/checkpoints/GeoBox-R1-SFT`, which Stage 2 starts from.

## Evaluation

Both scripts report **Acc@0.5, Acc@0.7 and mIoU (HBB) / mRIoU (OBB) in a single pass**, and
write `predictions.jsonl` plus a `summary.json` into a timestamped run directory.

```bash
# HBB — 7 evaluation sets
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_hbb.py \
    --model_path models/checkpoints/GeoBox-R1 \
    --dataset all

# OBB — 3 evaluation sets
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_obb.py \
    --model_path models/checkpoints/GeoBox-R1 \
    --dataset all
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--dataset` | `all`, or a single split (`dior_rsvg_test`, `rsvg_test`, `geochat_test`, `vrsbench_test`, `avvg_test`, …) |
| `--checkpoint_dir` | LoRA adapter to apply on top of `--model_path` |
| `--model_type` | ms-swift model type; `qwen3_vl` (default) or `qwen2_5_vl` for baselines |
| `--coord_mode` | Coordinate convention: `norm1000` (default), `absolute`, `resized_absolute` |
| `--batch_size` | Overrides the per-dataset default |
| `--table_format` | `text` or `markdown` |

For this repository's Qwen2.5-VL baseline, pass both `--model_type qwen2_5_vl` and
`--coord_mode resized_absolute`.

## Training

```bash
# Stage 1 — curriculum-guided SFT
bash training/sft.sh

# Stage 2 — GDPO. Start the vLLM rollout server first, then launch training.
bash training/rollout_vllm.sh
bash training/gdpo.sh

# Merge the LoRA adapters into standalone checkpoints
bash training/merge_lora.sh sft     # → GeoBox-R1-SFT
bash training/merge_lora.sh gdpo    # → GeoBox-R1
```

The curriculum lives in the data order, not in the training script — `sft.sh` consumes
`sft_curriculum_cot.jsonl` and must not shuffle it. Shuffling turns the run into the Mixed
baseline of the ablation.

`training/reward_plugin_qwen3vl.py` implements the two geometric rewards. The adaptive scale is
$\tau = \tau_c \cdot \sqrt{\mathrm{Tr}(\Sigma_g)}$, which normalizes reward sensitivity with respect to target size;
`training/gdpo_fixed_tau.sh` is the fixed-τ control. `reward_plugin_qwen2_5vl.py` is the same
reward set for a Qwen2.5-VL policy, which uses a different coordinate convention; launch it
with `training/gdpo_qwen2_5vl.sh`.

Stage 1 uses LoRA rank 16 / alpha 32 with the vision encoder and merger frozen, lr `1e-4`,
1 epoch on 2× RTX 4090. Stage 2 runs on a 20% OBB subset (19,357 of 96,791 samples).

## Demo

```bash
cd demo
bash run_demo.sh
```

Upload an aerial image, type a description, and pick HBB or OBB. Five curated examples ship
with the demo. See [`demo/README.md`](demo/README.md) for GPU/port options.

## Citation

```bibtex
@misc{geoboxr1,
  title  = {GeoBox-R1: Curriculum-Guided SFT and Geometric RL for
            Unified Box-Level Remote Sensing Visual Grounding},
  author = {Lan, Chenxi and Wu, Yuchen and Zhou, Minghang and
            Li, Tianyu and Qiu, Zhihao and Wang, Guoqing},
  year   = {2026},
  url    = {https://yuchenwu73.github.io/geobox-r1/},
  note   = {Preprint}
}
```

## Acknowledgements

Built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and
[ms-swift](https://github.com/modelscope/ms-swift). The refGeo benchmark setting follows
[GeoGround](https://github.com/zytx121/GeoGround). We thank the authors of DIOR-RSVG, RSVG,
GeoChat, VRSBench and AVVG for releasing their data.

## License

Code is released under the MIT License. The model weights follow CC BY-NC 4.0; the training
sets you build inherit the licenses of the underlying benchmarks.

<b>English</b> | <a href="README_zh.md">简体中文</a>

# Training

Two stages, both on [ms-swift](https://github.com/modelscope/ms-swift) 4.0.2.

## Stage 1 — Curriculum-guided SFT

```bash
bash training/sft.sh
```

The curriculum lives in the **data order**, not in the script: training samples are arranged
HBB grounding → OBB grounding → HBB-to-OBB CoT and must not be shuffled. `sft.sh` therefore
sets both dataset and dataloader shuffling to `false`.

Configuration: LoRA rank 16 / alpha 32, vision encoder and merger frozen, learning rate `1e-4`,
1 epoch, 2 × RTX 4090 (24 GB). Data is 161,692 samples (32,638 HBB + 96,791 OBB + 32,263 CoT).

## Stage 2 — GDPO

Start the vLLM rollout server first, then launch training (two terminals):

```bash
bash training/rollout_vllm.sh   # terminal 1
bash training/gdpo.sh           # terminal 2
```

`rollout_vllm.sh` listens on port 7772; the rollout address in `gdpo.sh` must match.

Rewards are implemented in `reward_plugin_qwen3vl.py`:

- **Rotated IoU** — direct overlap supervision for oriented boxes.
- **Adaptive Wasserstein distance** — treats predicted and ground-truth boxes as 2D Gaussians,
  measures their 2-Wasserstein distance, and maps it to a bounded reward `r = τ / (τ + W₂)`.
  The scale `τ = τ_c · sqrt(Tr(Σ_g))` adapts to target size; a fixed τ systematically
  under-rewards large targets. This channel stays informative when RIoU saturates near zero
  and stops providing gradient.

Equal weights (0.5 each), trained on a 20% OBB subset (19,357 of 96,791 samples).

`gdpo_fixed_tau.sh` is the fixed-τ control; it runs against the same rollout server, so run
the two one after the other. `visualization/analyze_tau_fixed.py` and
`analyze_tau_adaptive.py` were used to settle on τ_c.

### Qwen2.5-VL

`reward_plugin_qwen2_5vl.py` handles Qwen2.5-VL's resized-image pixel coordinates; the Qwen3-VL
plugin instead expects `norm1000`. Its reward names use a `*_qwen2_5` suffix to avoid collisions.

```bash
bash training/rollout_vllm.sh qwen2_5
bash training/gdpo_qwen2_5vl.sh
```

The Qwen2.5-VL rollout server uses port 7773; the GDPO launcher uses the same port.

## Merging LoRA

```bash
bash training/merge_lora.sh sft    # SFT adapter  → GeoBox-R1-SFT
bash training/merge_lora.sh gdpo   # GDPO adapter → GeoBox-R1
bash training/merge_lora.sh gdpo-qwen2_5
```

The merge script selects the newest adapter beneath each versioned run directory. Override
`ADAPTER_ROOT=`, `ADAPTER=`, or `OUTPUT=` when your layout differs. Evaluation and the demo both
load the merged checkpoint.

## Paths

Every launcher first changes to the **repository root** and uses paths relative to it
(`models/...`, `data/GeoBox-R1-Data/...`, `training/reward_plugin_*.py`), so the commands above
work from any current directory. This matters for the data: the training JSONL files store image
paths that are also repository-root relative (`data/refGeo/images/<Subset>/<file>`, as written by
`data_pipeline/`), and ms-swift opens them relative to the working directory. `--dataset` in
`sft.sh` and `gdpo*.sh` points at those pipeline outputs; if you assemble a JSONL yourself, keep
its image paths valid from the repository root.

The `merge_lora.sh` overrides (`ADAPTER_ROOT=`, `ADAPTER=`, `OUTPUT=`, `BASE_MODEL=`) follow the
same rule: relative values resolve from the repository root.

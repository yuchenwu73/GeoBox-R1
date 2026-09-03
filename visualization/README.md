<b>English</b> | <a href="README_zh.md">简体中文</a>

# Visualization and analysis

## Qualitative figures

| Script | Purpose |
| --- | --- |
| `compare_testsets.py` | Batch multi-model comparison across evaluation sets; used for Figures 3 and 4 |
| `export_sample.py` | Export per-model predictions for a single sample; reuses the cache written by `compare_testsets.py` |
| `plot_obb_boxes.py` | Boxes only, ground truth vs prediction — good for inspecting fit up close |
| `infer_and_plot.py` | Runs inference and plots the result; supports HBB and OBB |

Coordinates are de-normalized from `norm1000` back to the original image size, matching the
evaluation scripts. Run everything from the repository root; outputs go to `output/visualizations/`.

```bash
# One image, HBB and OBB; --gpu is optional and only then touches CUDA_VISIBLE_DEVICES
python visualization/infer_and_plot.py --image path/to/scene.png --query "the ship at the pier" --mode both --gpu 0

# Inference for every model on every evaluation set, then the five-panel comparison figures
python visualization/compare_testsets.py --task obb --models all

# One sample from that cache (defaults to the first sample of the task's default split)
python visualization/export_sample.py --task obb --dataset vrsbench_test --question-id 12

# Boxes only, from explicit coordinates
python visualization/plot_obb_boxes.py --image-path scene.png --gt-obb "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]" --pred-obb "[[...]]"
```

## Choosing τ

The adaptive Wasserstein reward scales as `τ = τ_c · sqrt(Tr(Σ_g))`, which leaves `τ_c` to pick:

- `analyze_tau_adaptive.py` profiles the size and aspect-ratio distribution of OBBs in the RL
  training data and suggests a range for `τ_c`.
- `analyze_tau_fixed.py` contrasts this with a fixed τ, showing why the scale has to follow
  target size — under a fixed τ, the same relative localization error earns systematically
  lower reward on large targets.

The result feeds `training/gdpo.sh`; `training/gdpo_fixed_tau.sh` keeps the fixed-τ control.

Run either analysis script from any working directory. Both read
`data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl` relative to the repository root.

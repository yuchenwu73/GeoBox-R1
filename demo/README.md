<b>English</b> | <a href="README_zh.md">简体中文</a>

# GeoBox-R1 demo

A Gradio demo: upload an aerial image, describe the target in natural language, and the model
returns a horizontal (HBB) or oriented (OBB) box. The interface is bilingual (English / 中文).

## Running

```bash
cd demo
bash run_demo.sh
```

Defaults to GPU 0 and port 7860, loading `models/checkpoints/GeoBox-R1` from the repository root.

```bash
# GPU / port
GEOBOX_GPU=1 GEOBOX_PORT=8000 bash run_demo.sh

# Different model: a local path, or a Hugging Face repo id
GEOBOX_MODEL=yuchenwu73/GeoBox-R1 bash run_demo.sh

# Temporary public link (https://<random>.gradio.live)
GEOBOX_SHARE=1 bash run_demo.sh

# Without flash-attn; and a specific interpreter (default: python on PATH)
GEOBOX_ATTN=sdpa GEOBOX_PYTHON=/path/to/envs/geobox-r1/bin/python bash run_demo.sh
```

On a remote server, SSH port forwarding is more reliable and gives a stable address:

```bash
ssh -N -L 7860:localhost:7860 <user>@<server>
# then open http://localhost:7860 locally
```

Gradio's free `gradio.live` links are randomly assigned on each start and cannot be fixed.

## Bundled examples

Five curated examples ship with the demo — two HBB and three OBB tasks:

| # | Dataset | Task | What it shows |
|---|---------|------|---------------|
| 1 | DIOR-RSVG | HBB | Tennis court upper-left of the red vehicle — spatial relation reasoning |
| 2 | VRSBench | HBB | Rightmost airplane, clean target |
| 3 | GeoChat | OBB | Cargo ship at roughly 45°, box hugging the hull |
| 4 | AVVG | OBB | Slanted vehicle, box aligned to the body |
| 5 | VRSBench | OBB | Vehicle at roughly 45°, where the oriented box differs clearly from a horizontal one |

## Files

```text
demo/
├── app.py                  # Gradio interface
├── inference.py            # GeoBox-R1 inference wrapper (prompts identical to the eval scripts)
├── visualize.py            # GT / prediction boxes with label anti-overlap
├── prepare_examples.py     # regenerate the examples from refGeo
├── selftest.py             # batch self-check
├── examples_manifest.json  # example metadata
├── examples/               # example images
└── run_demo.sh             # launcher
```

## Self-check

Runs all five examples headlessly, printing IoU / RIoU and writing visualizations to
`selftest_out/`:

```bash
CUDA_VISIBLE_DEVICES=0 python selftest.py
```

## Regenerating examples

`prepare_examples.py` picks samples from the refGeo test splits and copies the images. Set up
`data/refGeo/` first as described in the [root README](../README.md#data):

```bash
REFGEO_ROOT=../data/refGeo python prepare_examples.py
```

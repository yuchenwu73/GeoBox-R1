<b>English</b> | <a href="README_zh.md">简体中文</a>

# Data pipeline

Builds the GeoBox-R1 training data from the refGeo annotations in four steps. Run every
command from the repository root: the scripts read `data/refGeo/metainfo/*_train.jsonl`,
keep intermediate parts next to it (`data/refGeo/SFT`, `data/refGeo/RL`) and write the final
training sets to `data/GeoBox-R1-Data/`. The sets are not distributed; the pipeline is
deterministic (seed 42) and reproduces the files the released model was trained on.

## Flow

```text
data/refGeo/metainfo/*_train.jsonl
  │
  ├─ 1. build_hbb.py      RSVG, DIOR-RSVG          →  SFT/{RSVG,DIOR-RSVG}_HBB_train.jsonl
  └─ 2. build_obb_cot.py  GeoChat, VRSBench, AVVG  →  SFT/<Subset>_OBB_train.jsonl   75% of each split
                                                      SFT/<Subset>_CoT_train.jsonl   25%, HBB-to-OBB CoT
          │
          ├─ 3. build_sft.py   HBB + OBB + CoT        →  data/GeoBox-R1-Data/sft/sft_<config>.jsonl
          └─ 4. build_rl.py    OBB parts → RL format  →  data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl
```

```bash
python data_pipeline/build_hbb.py
python data_pipeline/build_obb_cot.py
python data_pipeline/build_sft.py --config all   # default: curriculum_cot only
python data_pipeline/build_rl.py
python data_pipeline/check_format.py             # optional: print a few encoded samples
```

Every script takes `--refgeo_root` (default `data/refGeo`); steps 3 and 4 also take
`--output_dir` (default `data/GeoBox-R1-Data`) and `--seed` (default 42). Records store image
paths as `<refgeo_root>/images/<Subset>/<file>`, which the training launchers open relative to
the repository root. `check_format.py` needs ms-swift and the base model; the other scripts
only need Pillow and tqdm.

## Which file trains the main model

`sft_curriculum_cot.jsonl`. `build_sft.py` writes the same 161,692 samples in four
arrangements — the paper's 2×2 ablation over data organization and CoT supervision:


| Config              | Order                           | CoT | Corresponds to                                                   |
| --------------------- | --------------------------------- | :---: | ------------------------------------------------------------------ |
| `curriculum_cot`    | HBB → OBB → CoT, not shuffled | ✓ | **Curriculum-guided + CoT — the main training set**             |
| `curriculum_no_cot` | HBB → OBB, not shuffled        | ✗ | Curriculum-guided, no CoT (CoT samples become plain OBB samples) |
| `mixed_cot`         | globally shuffled               | ✓ | Mixed + CoT                                                      |
| `mixed_no_cot`      | globally shuffled               | ✗ | Mixed, no CoT                                                    |

The curriculum is the file order itself, so `training/sft.sh` disables dataset and
dataloader shuffling. The `mixed_*` files are ablation arms only.

In the `*_no_cot` arms the converted CoT records keep the prompt text of the original
training files verbatim (it carries one extra blank line before the closing code fence), so
`--config all` reproduces all four original files record for record.

The RL stage trains on `rl_obb_20pct.jsonl`: 20% of each OBB part, drawn with a fixed seed
in the order AVVG, GeoChat, VRSBench. RL records add `oriented_bbox`, `image_width` and
`image_height`, which the reward functions use to map the ground truth into the model's
`norm1000` coordinate space.

## Record format

```json
{"messages": [{"role": "user", "content": "<image>Locate the instance that matches the description: [<ref-object>]. ..."},
              {"role": "assistant", "content": "```json\n[\n\t{\"horizontal_bbox\": <bbox>}\n]\n```"}],
 "images": ["data/refGeo/images/DIOR-RSVG/00001.jpg"],
 "objects": {"ref": ["the tennis court on the upper left"], "bbox": [[287, 146, 408, 398]]},
 "origin_dataset": "DIOR-RSVG"}
```

`<ref-object>` and `<bbox>` are ms-swift placeholders: the expression and the box stay in
`objects` and are rendered into the text at training time (`QWENVL_BBOX_FORMAT=new` gives
norm1000 coordinates). OBB records carry four corners in `objects.bbox`; CoT records carry
`[hbb, p1, p2, p3, p4]` because their answer consumes the horizontal box first. All
coordinates are original-image pixels, rounded half-up.

## Data volume


| Dataset   |        HBB |        OBB |        CoT |       Total |
| ----------- | -----------: | -----------: | -----------: | ------------: |
| DIOR-RSVG |     27,133 |          0 |          0 |      27,133 |
| RSVG      |      5,505 |          0 |          0 |       5,505 |
| GeoChat   |          0 |     47,912 |     15,971 |      63,883 |
| VRSBench  |          0 |     29,017 |      9,672 |      38,689 |
| AVVG      |          0 |     19,862 |      6,620 |      26,482 |
| **Total** | **32,638** | **96,791** | **32,263** | **161,692** |

The RL subset keeps 9,582 GeoChat, 5,803 VRSBench and 3,972 AVVG samples — 19,357 in total.

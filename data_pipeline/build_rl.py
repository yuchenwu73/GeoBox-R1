"""Step 4: build the GDPO training set from the OBB SFT samples.

Reads   <refgeo_root>/SFT/{AVVG,GeoChat,VRSBench}_OBB_train.jsonl   from build_obb_cot.py
Writes  <refgeo_root>/RL/<Subset>_OBB_train.jsonl        the same records in RL format
        <output_dir>/rl/rl_obb_<percent>pct.jsonl        the sampled subset used for training

RL format = the SFT record plus three columns that the reward functions read
(`training/reward_plugin_qwen3vl.py`):
    oriented_bbox               ground-truth corners in original-image pixels (= objects.bbox)
    image_width, image_height   original image size, needed to map the GT into the norm1000
                                space in which the policy predicts boxes

The subset is drawn per dataset (20% of each) with one seeded generator in the fixed order
AVVG, GeoChat, VRSBench; the reported 19,357-sample set is 3,972 + 9,582 + 5,803.

Usage (from the repository root, after step 2):
    python data_pipeline/build_rl.py                       # convert, then sample 20%
    python data_pipeline/build_rl.py --skip_convert        # reuse existing <root>/RL parts
"""
import argparse
import json
import os
import random

from PIL import Image
from tqdm import tqdm

# Sampling order matters: one random generator is shared across the three files.
OBB_SUBSETS = ["AVVG", "GeoChat", "VRSBench"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def convert(refgeo_root):
    """SFT OBB records -> RL records, adding the GT corners and the image size."""
    sft_dir = os.path.join(refgeo_root, "SFT")
    rl_dir = os.path.join(refgeo_root, "RL")
    os.makedirs(rl_dir, exist_ok=True)
    size_cache = {}
    for subset in OBB_SUBSETS:
        records = load_jsonl(os.path.join(sft_dir, f"{subset}_OBB_train.jsonl"))
        for record in tqdm(records, desc=f"{subset} -> RL"):
            image = record["images"][0]
            if image not in size_cache:
                with Image.open(image) as im:
                    size_cache[image] = im.size
            record["oriented_bbox"] = record["objects"]["bbox"]
            record["image_width"], record["image_height"] = size_cache[image]
        out_path = os.path.join(rl_dir, f"{subset}_OBB_train.jsonl")
        save_jsonl(records, out_path)
        print(f"{subset}: {len(records)} RL samples -> {out_path}")


def sample(refgeo_root, output_dir, percent, seed):
    rng = random.Random(seed)
    sampled = []
    for subset in OBB_SUBSETS:
        records = load_jsonl(os.path.join(refgeo_root, "RL", f"{subset}_OBB_train.jsonl"))
        count = int(len(records) * percent / 100)
        sampled.extend(rng.sample(records, count))
        print(f"{subset}: {len(records)} -> {count}")
    tag = f"{int(percent)}" if float(percent).is_integer() else f"{percent:g}"
    out_path = os.path.join(output_dir, "rl", f"rl_obb_{tag}pct.jsonl")
    save_jsonl(sampled, out_path)
    print(f"RL subset: {len(sampled)} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build the GDPO training subset.")
    parser.add_argument("--refgeo_root", default="data/refGeo",
                        help="refGeo root; OBB parts are read from <root>/SFT, RL parts written to <root>/RL")
    parser.add_argument("--output_dir", default="data/GeoBox-R1-Data", help="the subset is written to <output_dir>/rl")
    parser.add_argument("--percent", type=float, default=20, help="share of each dataset to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_convert", action="store_true", help="reuse the RL parts already in <root>/RL")
    args = parser.parse_args()
    if not 0 < args.percent <= 100:
        raise ValueError("--percent must be in (0, 100]")

    if not args.skip_convert:
        convert(args.refgeo_root)
    sample(args.refgeo_root, args.output_dir, args.percent, args.seed)


if __name__ == "__main__":
    main()

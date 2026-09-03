"""Step 2: build the OBB and HBB-to-OBB CoT samples from GeoChat, VRSBench and AVVG.

Reads   <refgeo_root>/metainfo/{geochat,vrsbench,avvg}_train.jsonl
Writes  <refgeo_root>/SFT/<Subset>_OBB_train.jsonl   75% of each split: plain OBB grounding
        <refgeo_root>/SFT/<Subset>_CoT_train.jsonl   25% of each split: coarse HBB, then the OBB

These three benchmarks carry both a horizontal box (`bbox`) and a four-vertex oriented box
(`poly`), which is what makes the two-step CoT supervision possible. Each split is shuffled
with a fixed seed and cut 75/25 per dataset, so every source contributes to both parts in
the same proportion.

Record layout (ms-swift conversation with placeholders, see build_hbb.py):
    OBB   objects.bbox = [p1, p2, p3, p4]            answer: {"oriented_bbox": [<bbox> x4]}
    CoT   objects.bbox = [hbb, p1, p2, p3, p4]       answer: horizontal_bbox first, then the
                                                     oriented_bbox (one of three phrasings)
The placeholders are consumed in order, which is why the CoT record stores the horizontal
box as the first entry of `objects.bbox`.

Usage (from the repository root):
    python data_pipeline/build_obb_cot.py
    python data_pipeline/build_obb_cot.py --refgeo_root /path/to/refGeo --seed 42
"""
import argparse
import json
import os
import random
from decimal import ROUND_HALF_UP, Decimal

from tqdm import tqdm

# (subset name used in paths and `origin_dataset`, metainfo file)
DATASETS = [
    ("AVVG", "avvg_train.jsonl"),
    ("VRSBench", "vrsbench_train.jsonl"),
    ("GeoChat", "geochat_train.jsonl"),
]

OBB_PROMPT = """<image>Locate the instance that matches the description: [<ref-object>]. Report oriented bbox coordinates in following JSON format:
```json
[
\t{"oriented_bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}
]
```"""

OBB_RESPONSE = """```json
[
\t{"oriented_bbox": [<bbox>, <bbox>, <bbox>, <bbox>]}
]
```"""

COT_PROMPT = "<image>Locate the instance that matches the description: [<ref-object>] in a stepwise manner."

# Three phrasings of the same two-step answer, picked at random per sample.
COT_RESPONSES = [
    '''The instance is located roughly at\n```json\n[\n\t{\"horizontal_bbox\": <bbox>}\n]\n```\nRefining this region, its oriented bbox is\n```json\n[\n\t{\"oriented_bbox\": [<bbox>, <bbox>, <bbox>, <bbox>]}\n]\n```''',
    '''I can see the instance in the area defined by\n```json\n[\n\t{\"horizontal_bbox\": <bbox>}\n]\n```\nTaking its rotation into account, it precisely occupies\n```json\n[\n\t{\"oriented_bbox\": [<bbox>, <bbox>, <bbox>, <bbox>]}\n]\n```''',
    '''Step 1, coarse horizontal location:\n```json\n[\n\t{\"horizontal_bbox\": <bbox>}\n]\n```\nStep 2, fine-grained oriented detection:\n```json\n[\n\t{\"oriented_bbox\": [<bbox>, <bbox>, <bbox>, <bbox>]}\n]\n```''',
]

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def round_half_up(value, ndigits=0):
    """Round like a human would (2.5 -> 3), not banker's rounding as `round()` does."""
    quantize_exp = Decimal(1) if ndigits == 0 else Decimal(1).scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(quantize_exp, rounding=ROUND_HALF_UP))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_image(images_dir, subset, image_id):
    """Locate the image, trying the other common extensions (GeoChat metainfo says .jpg,
    the released images are .png). Returns None when nothing exists."""
    path = os.path.join(images_dir, subset, image_id)
    if os.path.exists(path):
        return path
    stem = os.path.splitext(path)[0]
    for ext in IMAGE_EXTENSIONS:
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return None


def int_point(point):
    return [int(round_half_up(point[0])), int(round_half_up(point[1]))]


def obb_record(item, image, subset):
    return {
        "messages": [
            {"role": "user", "content": OBB_PROMPT},
            {"role": "assistant", "content": OBB_RESPONSE},
        ],
        "images": [image],
        "objects": {"ref": [item["question"]], "bbox": [int_point(p) for p in item["poly"]]},
        "origin_dataset": subset,
    }


def cot_record(item, image, subset, response):
    hbb = [int(round_half_up(v)) for v in item["bbox"]]
    return {
        "messages": [
            {"role": "user", "content": COT_PROMPT},
            {"role": "assistant", "content": response},
        ],
        "images": [image],
        # Horizontal box first: the CoT answer consumes the placeholders in this order.
        "objects": {"ref": [item["question"]], "bbox": [hbb] + [int_point(p) for p in item["poly"]]},
        "origin_dataset": subset,
    }


def build_subset(subset, metainfo_file, refgeo_root, sft_dir, obb_ratio, seed):
    items = load_jsonl(os.path.join(refgeo_root, "metainfo", metainfo_file))
    images_dir = os.path.join(refgeo_root, "images")

    # Seeded shuffle, then a per-dataset cut. The CoT templates below draw from the same
    # global `random` state right after the shuffle; keep this order to reproduce the data.
    random.seed(seed)
    random.shuffle(items)
    split = int(round_half_up(len(items) * obb_ratio))
    obb_items, cot_items = items[:split], items[split:]
    print(f"\n{subset}: {len(items)} annotations -> {len(obb_items)} OBB + {len(cot_items)} CoT")

    obb_path = os.path.join(sft_dir, f"{subset}_OBB_train.jsonl")
    cot_path = os.path.join(sft_dir, f"{subset}_CoT_train.jsonl")
    counts = {"obb": 0, "cot": 0, "missing": 0}

    with open(obb_path, "w", encoding="utf-8") as out:
        for item in tqdm(obb_items, desc=f"{subset} OBB"):
            image = resolve_image(images_dir, subset, item["image_id"])
            if image is None:  # samples without an image are dropped
                counts["missing"] += 1
                continue
            out.write(json.dumps(obb_record(item, image, subset), ensure_ascii=False) + "\n")
            counts["obb"] += 1

    with open(cot_path, "w", encoding="utf-8") as out:
        for item in tqdm(cot_items, desc=f"{subset} CoT"):
            image = resolve_image(images_dir, subset, item["image_id"])
            if image is None:
                counts["missing"] += 1
                continue
            response = random.choice(COT_RESPONSES)
            out.write(json.dumps(cot_record(item, image, subset, response), ensure_ascii=False) + "\n")
            counts["cot"] += 1

    print(f"{subset}: wrote {counts['obb']} OBB -> {obb_path}")
    print(f"{subset}: wrote {counts['cot']} CoT -> {cot_path}")
    if counts["missing"]:
        print(f"{subset}: skipped {counts['missing']} samples with no image on disk")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Build the OBB and CoT SFT samples from refGeo.")
    parser.add_argument("--refgeo_root", default="data/refGeo",
                        help="refGeo root with metainfo/ and images/; SFT parts go to <root>/SFT")
    parser.add_argument("--obb_ratio", type=float, default=0.75, help="share of each split used as plain OBB samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sft_dir = os.path.join(args.refgeo_root, "SFT")
    os.makedirs(sft_dir, exist_ok=True)
    total_obb = total_cot = 0
    for subset, meta in DATASETS:
        counts = build_subset(subset, meta, args.refgeo_root, sft_dir, args.obb_ratio, args.seed)
        total_obb += counts["obb"]
        total_cot += counts["cot"]
    print(f"\nTotal: {total_obb} OBB + {total_cot} CoT = {total_obb + total_cot} samples")


if __name__ == "__main__":
    main()

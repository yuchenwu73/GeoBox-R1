"""Step 1: build the HBB grounding samples from the RSVG and DIOR-RSVG train splits.

Reads   <refgeo_root>/metainfo/{rsvg,dior_rsvg}_train.jsonl
Writes  <refgeo_root>/SFT/{RSVG,DIOR-RSVG}_HBB_train.jsonl   (one record per referring expression)

Every record is an ms-swift multimodal conversation. The referring expression and the box
stay in `objects` and are spliced into the text by ms-swift at training time through the
`<ref-object>` / `<bbox>` placeholders (with `QWENVL_BBOX_FORMAT=new` the box is emitted as
norm1000 coordinates):

    {"messages": [{"role": "user", "content": "<image>Locate ... [<ref-object>] ..."},
                  {"role": "assistant", "content": "```json ... {\"horizontal_bbox\": <bbox>} ..."}],
     "images": ["data/refGeo/images/DIOR-RSVG/00001.jpg"],
     "objects": {"ref": ["<referring expression>"], "bbox": [[x1, y1, x2, y2]]},
     "origin_dataset": "DIOR-RSVG"}

Boxes are kept in original-image pixels, rounded half-up to integers. Image paths are
written relative to the repository root (or as given via --refgeo_root), which is how the
training launchers resolve them.

Usage (from the repository root):
    python data_pipeline/build_hbb.py
    python data_pipeline/build_hbb.py --refgeo_root /path/to/refGeo
"""
import argparse
import json
import os
from decimal import ROUND_HALF_UP, Decimal

from tqdm import tqdm

# (subset name used in paths and `origin_dataset`, metainfo file)
DATASETS = [
    ("RSVG", "rsvg_train.jsonl"),
    ("DIOR-RSVG", "dior_rsvg_train.jsonl"),
]

HBB_PROMPT = """<image>Locate the instance that matches the description: [<ref-object>]. Report horizontal bbox coordinates in following JSON format:
```json
[
\t{"horizontal_bbox": [x1, y1, x2, y2]}
]
```"""

HBB_RESPONSE = """```json
[
\t{"horizontal_bbox": <bbox>}
]
```"""


def round_half_up(value, ndigits=0):
    """Round like a human would (2.5 -> 3), not banker's rounding as `round()` does."""
    quantize_exp = Decimal(1) if ndigits == 0 else Decimal(1).scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(quantize_exp, rounding=ROUND_HALF_UP))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def image_path(images_dir, subset, image_id):
    """Path recorded for an image; swaps .jpg <-> .png once when the file is not found.

    The record is written even when the image is still missing, matching the original
    build. Missing images surface at training time instead of silently shrinking the set.
    """
    path = os.path.join(images_dir, subset, image_id)
    if not os.path.exists(path):
        stem, ext = os.path.splitext(path)
        if ext == ".jpg":
            path = stem + ".png"
        elif ext == ".png":
            path = stem + ".jpg"
    return path


def build_subset(subset, metainfo_file, refgeo_root, sft_dir):
    records = load_jsonl(os.path.join(refgeo_root, "metainfo", metainfo_file))
    images_dir = os.path.join(refgeo_root, "images")
    out_path = os.path.join(sft_dir, f"{subset}_HBB_train.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for item in tqdm(records, desc=f"{subset} HBB"):
            x1, y1, x2, y2 = (int(round_half_up(v)) for v in item["bbox"])
            record = {
                "messages": [
                    {"role": "user", "content": HBB_PROMPT},
                    {"role": "assistant", "content": HBB_RESPONSE},
                ],
                "images": [image_path(images_dir, subset, item["image_id"])],
                "objects": {"ref": [item["question"]], "bbox": [[x1, y1, x2, y2]]},
                "origin_dataset": subset,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{subset}: {len(records)} HBB samples -> {out_path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Build the HBB SFT samples from refGeo.")
    parser.add_argument("--refgeo_root", default="data/refGeo",
                        help="refGeo root with metainfo/ and images/; SFT parts go to <root>/SFT")
    args = parser.parse_args()

    sft_dir = os.path.join(args.refgeo_root, "SFT")
    os.makedirs(sft_dir, exist_ok=True)
    total = sum(build_subset(subset, meta, args.refgeo_root, sft_dir) for subset, meta in DATASETS)
    print(f"Total HBB samples: {total}")


if __name__ == "__main__":
    main()

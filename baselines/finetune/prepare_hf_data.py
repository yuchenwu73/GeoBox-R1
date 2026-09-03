#!/usr/bin/env python3
"""Materialize the refGeo SFT set for the HF-trainer baselines (InternVL3, LLaVA-OV-1.5).

The main model is trained by ms-swift, which expands the ``<ref-object>`` / ``<bbox>``
placeholders and normalizes coordinates on the fly. ``train_hf_lora.py`` uses plain
Transformers, so this script writes the final conversation text. Two copies are produced
from the same source file; they differ only in how the box numbers are written:

  refgeo_norm1000.jsonl   InternVL3:      round(x / W * 1000) integers, clamped to 0..1000
  refgeo_norm1.jsonl      LLaVA-OV-1.5:   round(x / W, 3) three-decimal floats, clamped to 0..1

x-coordinates are scaled by the image width and y-coordinates by the height, matching each
model's native grounding convention. The prompt/answer templates are kept verbatim, so the
baselines see exactly the same instructions and CoT text as GeoBox-R1.

Output record (one per line):
  {"id", "image": "<Subset>/<file>" (relative to --image-root), "conversations":
   [{"role": "user", ...}, {"role": "assistant", ...}], "origin_dataset"}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def fmt1000(v: float, dim: float) -> str:
    return str(max(0, min(1000, round(v / dim * 1000))))


def fmt1(v: float, dim: float) -> str:
    return f"{max(0.0, min(1.0, round(v / dim, 3))):.3f}"


def box_text(box, w, h, fmt) -> str:
    """Text for one ``<bbox>``: a 4-value HBB ``[x1, y1, x2, y2]`` or a 2-value corner ``[x, y]``."""
    if len(box) == 4:
        return f"[{fmt(box[0], w)}, {fmt(box[1], h)}, {fmt(box[2], w)}, {fmt(box[3], h)}]"
    return f"[{fmt(box[0], w)}, {fmt(box[1], h)}]"


def expand(text: str, boxes, w, h, fmt) -> str:
    """Replace the ``<bbox>`` placeholders in order; the count must match objects.bbox."""
    for box in boxes:
        if "<bbox>" not in text:
            raise ValueError("more objects.bbox entries than <bbox> placeholders")
        text = text.replace("<bbox>", box_text(box, w, h, fmt), 1)
    if "<bbox>" in text:
        raise ValueError("unconsumed <bbox> placeholder")
    return text


def image_relpath(path: str) -> str:
    """Return ``<Subset>/<file>`` regardless of how the source file recorded the image root."""
    normalized = path.replace("\\", "/")
    marker = "images/"
    idx = normalized.rfind(marker)
    return normalized[idx + len(marker):] if idx >= 0 else normalized


def sample_type(answer_template: str) -> str:
    if "horizontal_bbox" in answer_template and "oriented_bbox" in answer_template:
        return "CoT"
    return "OBB" if "oriented_bbox" in answer_template else "HBB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "GeoBox-R1-Data" / "sft" / "sft_curriculum_cot.jsonl",
                        help="ms-swift SFT JSONL with <ref-object>/<bbox> placeholders")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "data" / "refGeo" / "images",
                        help="Directory containing the RSVG/, DIOR-RSVG/, GeoChat/, VRSBench/, AVVG/ image folders")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "baselines" / "finetune" / "data")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out1000 = args.output_dir / "refgeo_norm1000.jsonl"
    out1 = args.output_dir / "refgeo_norm1.jsonl"

    n, skipped = 0, 0
    n_type = {"HBB": 0, "OBB": 0, "CoT": 0}
    size_cache: dict[str, tuple[int, int]] = {}
    with (args.input.open(encoding="utf-8") as src,
          out1000.open("w", encoding="utf-8") as f1000,
          out1.open("w", encoding="utf-8") as f1):
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rel = image_relpath(record["images"][0])
            image_path = args.image_root / rel
            try:
                if rel not in size_cache:
                    with Image.open(image_path) as im:
                        size_cache[rel] = im.size
            except OSError:
                skipped += 1
                continue
            w, h = size_cache[rel]

            question = record["objects"]["ref"][0]
            boxes = record["objects"]["bbox"]
            user = record["messages"][0]["content"].replace("<ref-object>", question)
            answer_template = record["messages"][1]["content"]
            n_type[sample_type(answer_template)] += 1

            for handle, fmt in ((f1000, fmt1000), (f1, fmt1)):
                out = {
                    "id": f"refgeo-norm-{n:08d}",
                    "image": rel,
                    "conversations": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": expand(answer_template, boxes, w, h, fmt)},
                    ],
                    "origin_dataset": record.get("origin_dataset"),
                }
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"converted={n} skipped_missing_image={skipped} by_type={n_type}")
    print(f"norm1000: {out1000}")
    print(f"norm1:    {out1}")


if __name__ == "__main__":
    main()

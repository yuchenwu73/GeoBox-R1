#!/usr/bin/env python3
"""Materialize the refGeo SFT set in GeoChat's native grounding format (LLaVA JSON).

GeoChat resizes every input to 504x504 and cannot recover the original image size, so it
must be trained in the coordinate format it was pretrained on rather than in the pixel-space
JSON used by GeoBox-R1. The samples are the same as ``sft_curriculum_cot.jsonl``; only the
answer text changes:

  answer      {<x1><y1><x2><y2>|<theta>}   integers on a 0..100 grid plus a rotation in degrees
  HBB         theta = 0
  OBB         polygon -> cv2.minAreaRect -> unrotated box + theta, with theta folded into [0, 90]
  CoT         two-step sentence: coarse HBB, then the refined OBB (mirrors the JSON CoT)
  prompt      "[refer] Give me the location of <p> {q} </p>", identical to the evaluation
              prompt in baselines/evaluate/eval_geochat.py; CoT appends " in a stepwise manner."

Angle convention: theta rotates the unrotated box clockwise about its center in image
coordinates (y down), which is exactly what eval_geochat.rotate_rect undoes. Because the
same rectangle can be written as (w, h, theta) or (h, w, theta +- 90), each OBB tries the three
equivalent parameterizations and keeps the one whose quantized text round-trips to the best
IoU with the original polygon. The mean round-trip IoU is printed as a sanity check.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
REFER_PROMPT = "[refer] Give me the location of <p> {q} </p>"


def rotate_rect(x1, y1, x2, y2, angle):
    """Corners of the axis-aligned box rotated by ``angle`` degrees (same as eval_geochat.rotate_rect)."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return [[cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c]
            for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]


def poly_iou(a, b) -> float:
    from shapely.geometry import Polygon

    def order(pts):
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    pa, pb = Polygon(order([tuple(p) for p in a])), Polygon(order([tuple(p) for p in b]))
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    union = pa.union(pb).area
    return pa.intersection(pb).area / union if union else 0.0


def clamp100(v: float) -> int:
    return max(0, min(100, round(v)))


def hbb_native(bbox, w, h) -> str:
    """Pixel ``[x1, y1, x2, y2]`` -> ``{<..>|<0>}``."""
    x1, y1, x2, y2 = bbox
    return (f"{{<{clamp100(x1 / w * 100)}><{clamp100(y1 / h * 100)}>"
            f"<{clamp100(x2 / w * 100)}><{clamp100(y2 / h * 100)}>|<0>}}")


def obb_native(poly, w, h):
    """Pixel polygon -> (``{<..>|<theta>}`` text, round-trip IoU).

    The IoU is None for zero-area polygons (degenerate source annotations). Those samples are
    still written, so every baseline trains on the same records, but they are excluded from the
    round-trip statistic.
    """
    pts = np.array(poly, dtype=np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    degenerate = rw < 1e-3 or rh < 1e-3
    best_txt, best_iou = None, -1.0
    for ww, hh, th in ((rw, rh, ang), (rh, rw, ang + 90), (rh, rw, ang - 90)):
        ti = round(th) % 180
        if ti > 90:  # equivalent box: swap the sides and take theta - 90
            ww, hh, ti = hh, ww, ti - 90
        x1 = clamp100((cx - ww / 2) / w * 100)
        y1 = clamp100((cy - hh / 2) / h * 100)
        x2 = clamp100((cx + ww / 2) / w * 100)
        y2 = clamp100((cy + hh / 2) / h * 100)
        txt = f"{{<{x1}><{y1}><{x2}><{y2}>|<{ti}>}}"
        if degenerate:
            return txt, None
        rec = rotate_rect(x1 / 100 * w, y1 / 100 * h, x2 / 100 * w, y2 / 100 * h, ti)
        iou = poly_iou(rec, [list(p) for p in poly])
        if iou > best_iou:
            best_iou, best_txt = iou, txt
    return best_txt, best_iou


def image_relpath(path: str) -> str:
    """Return ``<Subset>/<file>`` regardless of how the source file recorded the image root."""
    normalized = path.replace("\\", "/")
    marker = "images/"
    idx = normalized.rfind(marker)
    return normalized[idx + len(marker):] if idx >= 0 else normalized


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
    out_path = args.output_dir / "refgeo_geochat_native_llava.json"

    samples, ious = [], []
    skipped = n_degenerate = 0
    n_type = {"HBB": 0, "OBB": 0, "CoT": 0}
    size_cache: dict[str, tuple[int, int]] = {}
    with args.input.open(encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rel = image_relpath(record["images"][0])
            try:
                if rel not in size_cache:
                    with Image.open(args.image_root / rel) as im:
                        size_cache[rel] = im.size
            except OSError:
                skipped += 1
                continue
            w, h = size_cache[rel]

            question = record["objects"]["ref"][0]
            boxes = record["objects"]["bbox"]
            answer_template = record["messages"][1]["content"]
            is_cot = "horizontal_bbox" in answer_template and "oriented_bbox" in answer_template
            is_obb = (not is_cot) and "oriented_bbox" in answer_template
            prompt = REFER_PROMPT.format(q=question)

            if is_cot:
                # objects.bbox = [hbb (4 values), corner1, corner2, corner3, corner4]
                hbb_txt = hbb_native(boxes[0], w, h)
                obb_txt, iou = obb_native(boxes[1:5], w, h)
                user = f"<image>\n{prompt} in a stepwise manner."
                gpt = (f"The instance is located roughly at {hbb_txt}. "
                       f"Refining this region, its oriented location is {obb_txt}.")
                n_type["CoT"] += 1
            elif is_obb:
                obb_txt, iou = obb_native(boxes[:4], w, h)
                user = f"<image>\n{prompt}"
                gpt = obb_txt
                n_type["OBB"] += 1
            else:
                iou = None
                user = f"<image>\n{prompt}"
                gpt = hbb_native(boxes[0], w, h)
                n_type["HBB"] += 1
            if is_cot or is_obb:
                if iou is None:
                    n_degenerate += 1
                else:
                    ious.append(iou)

            samples.append({
                "id": f"geochat-native-{len(samples):08d}",
                "image": rel,
                "conversations": [{"from": "human", "value": user}, {"from": "gpt", "value": gpt}],
            })

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

    print(f"converted={len(samples)} skipped_missing_image={skipped} by_type={n_type}")
    print(f"degenerate_obb_gt={n_degenerate} (written, excluded from the round-trip statistic)")
    if ious:
        arr = np.array(ious)
        print(f"round-trip IoU of the quantized OBB text: mean={arr.mean():.4f} "
              f"p5={np.percentile(arr, 5):.4f} below_0.8={float((arr < 0.8).mean()) * 100:.2f}%")
        if arr.mean() < 0.8:
            print("WARNING: mean round-trip IoU below 0.8; check the angle convention")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()

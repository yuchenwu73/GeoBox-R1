#!/usr/bin/env python3
"""InternVL3 (2B / 8B) zero-shot baseline through ms-swift's TransformersEngine.

Reproduces the zero-shot "InternVL3" rows. The HBB prompt is the official InternVL
grounding prompt plus a format suffix; the suffix raises the share of answers that
contain a box (the official prompt alone often digresses) and is the prompt behind
the reported numbers. InternVL has no OBB training, so under the strict OBB parser
(only 5 values with an angle or 8 corner values count) its OBB scores are ~0 by
design: that row documents the missing capability.

Output format: `<box>[[x1, y1, x2, y2]]</box>`, normalized to [0, 1000] and mapped back
with the original image size (dynamic tiling happens inside the model).

Environment: ms-swift (GitHub-format weights use model_type internvl3_5, `-HF`
directories use internvl_hf). Usage, from the repository root:
    python baselines/evaluate/eval_internvl.py --task hbb --model_path models/pretrained/InternVL3-8B --output_dir eval_results/baselines/internvl3_8b
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional

from common import add_common_args, clip_hbb, run_evaluation, select_datasets

HBB_PROMPT = ("Please provide the bounding box coordinate of the region this sentence describes: <ref>{question}</ref>"
              " Answer with only the bounding box in the format <box>[[x1,y1,x2,y2]]</box>.")
OBB_PROMPT = ("Locate the region described by: <ref>{question}</ref>. Return only its oriented bounding box as four "
              "clockwise corner points normalized to [0,1000]: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].")


def prompt(task: str, question: str) -> str:
    return (HBB_PROMPT if task == "hbb" else OBB_PROMPT).format(question=question)


# --------------------------------------------------------------------------- parsing
def nums(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def parse_hbb(text: str, size) -> Optional[List[float]]:
    """First bracketed fragment with 4+ numbers (whole text as fallback), norm1000 -> pixels, clipped."""
    candidates = re.findall(r"\[\s*\[?\s*-?\d[\d\s,\.\-]*\]?\s*\]", text)
    for candidate in candidates + [text]:
        values = nums(candidate)
        if len(values) >= 4:
            w, h = size
            x1, y1, x2, y2 = values[:4]
            return clip_hbb([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], size)
    return None


def _poly_area(pts) -> float:
    """Shoelace area after polar ordering; detects degenerate polygons (e.g. a box repeated twice)."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return abs(sum(ordered[i][0] * ordered[(i + 1) % 4][1] - ordered[(i + 1) % 4][0] * ordered[i][1] for i in range(4))) / 2


def parse_obb(text: str, size) -> Optional[List[List[float]]]:
    """Strict OBB parser: 8 values are four corners (must be non-degenerate), 5 values are
    x1, y1, x2, y2 plus an angle in degrees (rotated about the center, clockwise in image
    coordinates). A plain 4-value horizontal box scores nothing."""
    w, h = size
    candidates = re.findall(r"\[\s*\[?[\d\s,\.\-\]\[]+\]?\s*\]", text)
    for candidate in candidates + [text]:
        values = nums(candidate)
        if len(values) >= 8:
            pts = [[values[i] / 1000 * w, values[i + 1] / 1000 * h] for i in range(0, 8, 2)]
            if _poly_area(pts) > 1e-6:
                return pts
            continue
        if len(values) == 5:
            x1, x2 = sorted((values[0] / 1000 * w, values[2] / 1000 * w))
            y1, y2 = sorted((values[1] / 1000 * h, values[3] / 1000 * h))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rad = math.radians(values[4])
            c, s = math.cos(rad), math.sin(rad)
            return [[cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c]
                    for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    return None


def response_text(response) -> str:
    """Generated text of a swift response (choices[0].message.content); never str() the whole object,
    whose usage/token fields would leak numbers into the coordinate parser."""
    if isinstance(response, str):
        return response
    if hasattr(response, "choices"):
        try:
            return response.choices[0].message.content
        except Exception:
            pass
    if isinstance(response, dict):
        try:
            return response["choices"][0]["message"]["content"]
        except Exception:
            pass
    for name in ("response", "text", "content"):
        value = getattr(response, name, None)
        if isinstance(value, str):
            return value
    return str(response)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="InternVL3 zero-shot baseline evaluation")
    parser.add_argument("--task", choices=["hbb", "obb"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="InternVL3", help="Model label in the result table")
    parser.add_argument("--model_format", choices=["auto", "github", "hf"], default="auto",
                        help="auto: hf when the directory name ends with -HF, github otherwise")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    add_common_args(parser, batch_size=1)
    args = parser.parse_args()

    import torch
    from swift.infer_engine import TransformersEngine
    from swift.infer_engine.protocol import RequestConfig

    model_format = args.model_format
    if model_format == "auto":
        model_format = "hf" if Path(args.model_path).name.upper().endswith("-HF") else "github"
    model_type = "internvl_hf" if model_format == "hf" else "internvl3_5"
    engine = TransformersEngine(model=args.model_path, model_type=model_type, torch_dtype=torch.bfloat16,
                                max_batch_size=args.batch_size, attn_impl="flash_attn")
    request_config = RequestConfig(max_tokens=args.max_new_tokens)

    def infer_batch(paths: List[str], prompts: List[str]) -> List[str]:
        requests = [{"messages": [{"role": "user", "content": "<image>" + p}], "images": [path]}
                    for path, p in zip(paths, prompts)]
        return [response_text(r) for r in engine.infer(requests, request_config=request_config, use_tqdm=False)]

    parse = parse_hbb if args.task == "hbb" else parse_obb
    run_evaluation(
        args.task, select_datasets(args.task, args.dataset), infer_batch,
        lambda question: prompt(args.task, question), parse, args, args.model_name,
        summary_extra={"model_path": args.model_path, "model_format": model_format, "model_type": model_type,
                       "coordinate_mode": "norm1000_original_image"},
    )


if __name__ == "__main__":
    main()

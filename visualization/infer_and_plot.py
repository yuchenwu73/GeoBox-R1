#!/usr/bin/env python3
"""Single-image GeoBox-R1 inference with an annotated output image (transformers path).

Loads the merged checkpoint, or a base model plus a LoRA adapter, with
transformers + peft + qwen_vl_utils instead of ms-swift, sends the HBB and/or OBB
grounding prompt for one image and query, parses the JSON answer and draws the
prediction (and optional ground-truth boxes) with OpenCV. --interactive loops
over image/query pairs typed on the terminal.

Model answers are norm1000 coordinates (see denorm_hbb); they are mapped to
original-image pixels before drawing, and --gt_hbb / --gt_obb are expected in
original-image pixels, as in the refGeo metainfo. --gpu sets CUDA_VISIBLE_DEVICES
before torch is imported; without it the caller's setting is respected.

Usage, from the repository root
    python visualization/infer_and_plot.py --image img.jpg --query "the yellow car" --mode both --gpu 0
    python visualization/infer_and_plot.py --interactive --gpu 0
"""

import ast
import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Configuration (before the CUDA imports)
# ---------------------------------------------------------------------------
def _parse_gpu_early(argv=None):
    """Return the --gpu value from argv (also --gpu=N), or None when it is not given.

    Runs before argparse so CUDA_VISIBLE_DEVICES can be set before torch is imported;
    without --gpu the caller's CUDA_VISIBLE_DEVICES is left untouched.
    """
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == '--gpu' and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith('--gpu='):
            return arg.split('=', 1)[1]
    return None


_GPU_OVERRIDE = _parse_gpu_early()
if _GPU_OVERRIDE is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_OVERRIDE

# Deliberately imported after the GPU selection above (torch reads CUDA_VISIBLE_DEVICES on import).
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # noqa: E402
from peft import PeftModel  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "checkpoints" / "GeoBox-R1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "visualizations" / "grounding_results"

# OpenCV colors are BGR tuples.
COLORS = {
    'HBB': (0, 0, 180),
    'OBB': (180, 0, 0),
    'GT_HBB': (120, 0, 120),
    'GT_OBB': (0, 100, 180),
    'FAIL': (0, 0, 255),
}

LINE_WIDTH = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4


# ---------------------------------------------------------------------------
# Prompts (byte-identical to evaluation/evaluate_hbb.py and evaluate_obb.py)
# ---------------------------------------------------------------------------
# The image is passed to the chat template as its own content item, so the text
# carries no <image> placeholder (that tag is an ms-swift convention).
def hbb_prompt(query: str) -> str:
    """HBB prompt, identical to evaluation/evaluate_hbb.get_prompt (a test asserts this)."""
    return f"""Locate the instance that matches the description: [{query}]. Report horizontal bbox coordinates in following JSON format:
```json
[
\t{{"horizontal_bbox": [x1, y1, x2, y2]}}
]
```"""


def obb_prompt(query: str) -> str:
    """OBB prompt, identical to evaluation/evaluate_obb.get_obb_prompt (a test asserts this)."""
    return f"""Locate the instance that matches the description: [{query}]. Report oriented bbox coordinates in following JSON format:
```json
[
\t{{"oriented_bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}}
]
```"""


# ---------------------------------------------------------------------------
# Answer parsing (ast.literal_eval only, never eval)
# ---------------------------------------------------------------------------
def parse_hbb(hbb_str: str) -> Optional[List[int]]:
    """Parse an HBB from a flat or singly nested coordinate list."""
    if not hbb_str:
        return None
    s = hbb_str.strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
            coords = parsed[0]
        elif isinstance(parsed, list) and len(parsed) == 4:
            coords = parsed
        else:
            raise ValueError(f"Unsupported HBB format: {hbb_str}")
        if len(coords) != 4:
            raise ValueError(f"HBB requires 4 coordinates, got {len(coords)}")
        return [int(c) for c in coords]
    except (SyntaxError, ValueError, TypeError):
        pass
    coords = [int(float(x.strip())) for x in s.replace('[', '').replace(']', '').split(',')]
    if len(coords) != 4:
        raise ValueError(f"HBB requires 4 coordinates, got {len(coords)}")
    return coords


def parse_obb(obb_str: str) -> Optional[List[List[int]]]:
    """Parse an OBB from nested point pairs or eight flat coordinates."""
    if not obb_str:
        return None
    s = obb_str.strip()

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(p, list) and len(p) == 2 for p in parsed):
            return [[int(p[0]), int(p[1])] for p in parsed]
        elif isinstance(parsed, list) and len(parsed) == 8:
            return [[int(parsed[i]), int(parsed[i+1])] for i in range(0, 8, 2)]
    except (SyntaxError, ValueError, TypeError):
        pass

    try:
        parsed = ast.literal_eval("[" + s + "]")
        if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(p, list) and len(p) == 2 for p in parsed):
            return [[int(p[0]), int(p[1])] for p in parsed]
    except (SyntaxError, ValueError, TypeError):
        pass

    coords = [int(float(x.strip())) for x in s.replace('[', '').replace(']', '').split(',')]
    if len(coords) != 8:
        raise ValueError(f"OBB requires 8 coordinates, got {len(coords)}")
    return [[coords[i], coords[i+1]] for i in range(0, 8, 2)]


def extract_hbb_from_response(response: str) -> Optional[List[int]]:
    """HBB from a model answer: the last "horizontal_bbox" (drafts precede the final
    answer in CoT outputs), else a flat four-number "oriented_bbox"."""
    pattern = r'"horizontal_bbox"\s*:\s*\[([^\]]+)\]'
    # Reasoning traces may contain drafts; the last valid box is the final answer.
    for match in reversed(list(re.finditer(pattern, response))):
        try:
            coords = [int(float(x.strip())) for x in match.group(1).split(',')]
        except ValueError:
            continue
        if len(coords) == 4:
            return coords

    pattern2 = r'"oriented_bbox"\s*:\s*\[(\s*\d+[\d\s,]*)\]'
    for match in reversed(list(re.finditer(pattern2, response))):
        try:
            coords = [int(float(x.strip())) for x in match.group(1).split(',')]
        except ValueError:
            continue
        if len(coords) == 4:
            return coords

    return None


def extract_obb_from_response(response: str) -> Optional[List[List[int]]]:
    """Last four-corner "oriented_bbox" in a model answer, or None."""
    pattern = r'"oriented_bbox"\s*:\s*\[((?:\s*\[[^\]]+\]\s*,?\s*)+)\]'
    for match in reversed(list(re.finditer(pattern, response))):
        coords_str = "[" + match.group(1) + "]"
        try:
            coords = ast.literal_eval(coords_str)
            if len(coords) == 4 and all(isinstance(c, list) and len(c) == 2 for c in coords):
                return [[int(c[0]), int(c[1])] for c in coords]
        except (SyntaxError, ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------
# GeoBox-R1 was trained with ms-swift under QWENVL_BBOX_FORMAT=new and norm_bbox=norm1000:
# every box in the training text lies on a 0-1000 grid relative to the whole image. The
# weights therefore emit norm1000 numbers no matter which loader runs them (transformers
# here, ms-swift in evaluation/ and demo/), so answers are mapped to original-image pixels
# with the same formula as evaluation/evaluate_hbb.py (x / 1000 * W, y / 1000 * H).
def denorm_hbb(coords: Sequence[float], size: Tuple[int, int]) -> List[float]:
    """norm1000 [x1, y1, x2, y2] -> original-image pixels for an image of size (W, H)."""
    w, h = size
    return [coords[0] / 1000.0 * w, coords[1] / 1000.0 * h, coords[2] / 1000.0 * w, coords[3] / 1000.0 * h]


def denorm_obb(coords: Sequence[Sequence[float]], size: Tuple[int, int]) -> List[List[float]]:
    """norm1000 corners -> original-image pixels for an image of size (W, H)."""
    w, h = size
    return [[pt[0] / 1000.0 * w, pt[1] / 1000.0 * h] for pt in coords]


# ---------------------------------------------------------------------------
# Drawing (OpenCV, colours are BGR)
# ---------------------------------------------------------------------------
def draw_label_with_bg(img, text, pos, font, scale, bg_color, text_color=(255, 255, 255), thickness=1):
    """Draw a text label on an opaque background."""
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - text_h - 4), (x + text_w + 4, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, text_color, thickness)
    return text_h + 8


# Boxes are drawn in the coordinate space they arrive in; no scaling is applied here.
def draw_hbb(img: np.ndarray, coords: List[int], label: str, color: tuple) -> np.ndarray:
    """Draw an HBB with a compact label."""
    x1, y1, x2, y2 = coords
    cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_WIDTH)

    text_x = x1
    text_y = y1 - 5 if y1 > 25 else y2 + 20

    draw_label_with_bg(img, label, (text_x, text_y), FONT, FONT_SCALE, color)
    return img


def draw_obb(img: np.ndarray, coords: List[List[int]], label: str, color: tuple) -> np.ndarray:
    """Draw an OBB with a compact label."""
    pts = np.array(coords, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=LINE_WIDTH)

    text_x = coords[0][0]
    text_y = coords[0][1] - 5 if coords[0][1] > 25 else coords[0][1] + 20

    draw_label_with_bg(img, label, (text_x, text_y), FONT, FONT_SCALE, color)
    return img


def draw_info_panel(img: np.ndarray, info_list: List[Tuple[str, str, tuple]]) -> np.ndarray:
    """Draw prediction details in the upper-left corner."""
    x, y = 10, 10
    for label, coord_str, color in info_list:
        text = f"{label}: {coord_str}"
        h = draw_label_with_bg(img, text, (x, y + 15), FONT, FONT_SCALE, color)
        y += h + 2
    return img


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------
def load_model(base_model: str, lora_path: Optional[str] = None):
    """Load the base model and an optional LoRA adapter."""
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path:
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print("No LoRA adapter specified; using the base model")
    model.eval()
    print("Model loaded")
    return model, processor


def infer(model, processor: AutoProcessor,
          image_path: str, prompt: str, max_tokens: int = 512) -> str:
    """One greedy generation through the HF chat template; returns only the newly generated text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def visualize(
    model,
    processor: AutoProcessor,
    image_path: str,
    query: str,
    mode: str,
    output_path: str,
    gt_hbb: Optional[str] = None,
    gt_obb: Optional[str] = None,
):
    """Run the requested modes on one image, draw GT and predictions, and save the result.

    Model answers are de-normalized from norm1000 to pixels before drawing; gt_hbb and
    gt_obb are already pixel coordinates and are drawn as given. In OBB mode an answer
    that only contains an HBB is drawn in red as "WRONG FORMAT" so format failures stay
    visible; unparsable answers are listed in the info panel.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read image {image_path}")
        return

    size = (img.shape[1], img.shape[0])  # (W, H) for the norm1000 -> pixel mapping
    print(f"Image size: {size[0]} x {size[1]}")
    print(f"Query: {query}")
    print(f"Mode: {mode}")

    drawn = []
    info_list = []

    if gt_hbb:
        gt_hbb_coords = parse_hbb(gt_hbb)
        if gt_hbb_coords:
            img = draw_hbb(img, gt_hbb_coords, "GT_HBB", COLORS['GT_HBB'])
            drawn.append("GT_HBB")
            info_list.append(("GT_HBB", str(gt_hbb_coords), COLORS['GT_HBB']))
            print(f"GT_HBB: {gt_hbb_coords}")

    if gt_obb:
        gt_obb_coords = parse_obb(gt_obb)
        if gt_obb_coords:
            img = draw_obb(img, gt_obb_coords, "GT_OBB", COLORS['GT_OBB'])
            drawn.append("GT_OBB")
            info_list.append(("GT_OBB", str(gt_obb_coords), COLORS['GT_OBB']))
            print(f"GT_OBB: {gt_obb_coords}")

    if mode in ['hbb', 'both']:
        print("\nRunning HBB inference...")
        response = infer(model, processor, image_path, hbb_prompt(query))
        print(f"Model response:\n{response}")

        hbb_norm = extract_hbb_from_response(response)
        if hbb_norm:
            hbb_coords = [int(round(v)) for v in denorm_hbb(hbb_norm, size)]
            img = draw_hbb(img, hbb_coords, "HBB", COLORS['HBB'])
            drawn.append("HBB")
            info_list.append(("HBB", str(hbb_coords), COLORS['HBB']))
            print(f"Parsed HBB: norm1000 {hbb_norm} -> pixels {hbb_coords}")
        else:
            print("Warning: no valid HBB found in the response")
            fail_text = response.strip()[:80].replace('\n', ' ')
            info_list.append(("HBB-FAIL", fail_text, COLORS['FAIL']))
            drawn.append("HBB-FAIL")

    if mode in ['obb', 'both']:
        print("\nRunning OBB inference...")
        response = infer(model, processor, image_path, obb_prompt(query))
        print(f"Model response:\n{response}")

        obb_norm = extract_obb_from_response(response)
        if obb_norm:
            obb_coords = [[int(round(x)), int(round(y))] for x, y in denorm_obb(obb_norm, size)]
            img = draw_obb(img, obb_coords, "OBB", COLORS['OBB'])
            drawn.append("OBB")
            info_list.append(("OBB", str(obb_coords), COLORS['OBB']))
            print(f"Parsed OBB: norm1000 {obb_norm} -> pixels {obb_coords}")
        else:
            print("Warning: no valid OBB found in the response")
            fallback_norm = extract_hbb_from_response(response)
            if fallback_norm:
                fallback_hbb = [int(round(v)) for v in denorm_hbb(fallback_norm, size)]
                img = draw_hbb(img, fallback_hbb, "WRONG FORMAT", COLORS['FAIL'])
                drawn.append("WRONG FORMAT")
                info_list.append(("Asked OBB, Got HBB", str(fallback_hbb), COLORS['FAIL']))
                print(f"Received an HBB instead: {fallback_hbb} (shown in red)")
            else:
                fail_text = response.strip()[:80].replace('\n', ' ')
                info_list.append(("OBB-FAIL", fail_text, COLORS['FAIL']))
                drawn.append("OBB-FAIL")

    if info_list:
        img = draw_info_panel(img, info_list)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, img):
        raise OSError(f"Failed to write image: {output_path}")
    print(f"\nDrawn: {', '.join(drawn) if drawn else 'none'}")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# Interactive mode and CLI
# ---------------------------------------------------------------------------
def interactive_mode(model, processor: AutoProcessor):
    """Run the interactive prompt loop."""
    output_dir = str(DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("Grounding visualizer - interactive mode")
    print(f"Default output directory: {output_dir}")
    print("=" * 50)

    while True:
        print("\n" + "-" * 40)
        image_path = input("Image path (q to quit): ").strip()
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("Bye")
            break

        if not os.path.exists(image_path):
            print(f"Error: image not found: {image_path}")
            continue

        query = input("Query (for example, 'the yellow car'): ").strip()
        if not query:
            print("Error: query cannot be empty")
            continue

        mode = input("Mode [hbb/obb/both] (default: both): ").strip().lower()
        if mode not in ['hbb', 'obb', 'both']:
            mode = 'both'
            print("Using default mode: both")

        gt_hbb = input("GT HBB pixel coordinates (optional): ").strip() or None
        gt_obb = input("GT OBB pixel coordinates (optional): ").strip() or None

        output_name = input("Output filename (default: output.png): ").strip()
        if not output_name:
            output_name = "output.png"
            print("Using default filename: output.png")

        output_path = os.path.join(output_dir, output_name)

        visualize(model, processor, image_path, query, mode, output_path, gt_hbb, gt_obb)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    parser = argparse.ArgumentParser(
        description="Run grounding inference and draw HBB/OBB predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_and_plot.py --image img.jpg --query "the yellow car" --mode hbb
  python infer_and_plot.py --image img.jpg --query "the yellow car" --mode obb
  python infer_and_plot.py --image img.jpg --query "the yellow car" --mode both
  python infer_and_plot.py --image img.jpg --query "the car" --mode hbb --gt_hbb "[[100,100,200,200]]"
  python infer_and_plot.py --interactive --gpu 0
        """
    )

    parser.add_argument('--image', '-i', type=str, help='Input image path')
    parser.add_argument('--query', '-q', type=str, help='Grounding query')
    parser.add_argument('--mode', '-m', type=str, choices=['hbb', 'obb', 'both'], default='both',
                        help='Inference mode (default: both)')
    parser.add_argument('--gt_hbb', type=str, help='Optional GT HBB in original-image pixels, e.g. "[100,100,200,200]"')
    parser.add_argument('--gt_obb', type=str, help='Optional GT OBB in original-image pixels, e.g. "[[x1,y1],...,[x4,y4]]"')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output image path')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Directory for automatically named outputs')
    parser.add_argument('--interactive', action='store_true', help='Run interactively')

    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU id exposed through CUDA_VISIBLE_DEVICES; default keeps the caller\'s setting')
    parser.add_argument('--base_model', type=str,
                        default=str(DEFAULT_MODEL),
                        help='Base model path')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Optional LoRA adapter path')
    return parser


def main():
    """CLI; without --output the file is auto-numbered <model>[_lora-<adapter>]_<NNN>.png in --output_dir."""
    parser = build_parser()
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES was already applied by _parse_gpu_early before torch was imported.

    model, processor = load_model(args.base_model, args.lora_path)

    if args.interactive:
        interactive_mode(model, processor)
    else:
        if not args.image or not args.query:
            parser.print_help()
            print("\nError: provide --image and --query, or use --interactive")
            return

        if args.output:
            output_path = args.output
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            model_name = os.path.basename(args.base_model.rstrip('/'))
            if args.lora_path:
                lora_name = os.path.basename(args.lora_path.rstrip('/'))
                prefix = f"{model_name}_lora-{lora_name}"
            else:
                prefix = f"{model_name}_base"
            pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.(?:png|jpg)$')
            existing = [f for f in os.listdir(args.output_dir) if pattern.match(f)]
            if existing:
                max_num = max(int(pattern.match(f).group(1)) for f in existing)
                next_num = max_num + 1
            else:
                next_num = 1
            output_path = os.path.join(args.output_dir, f"{prefix}_{next_num:03d}.png")

        visualize(model, processor, args.image, args.query, args.mode, output_path,
                  args.gt_hbb, args.gt_obb)


if __name__ == "__main__":
    main()

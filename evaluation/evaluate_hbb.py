#!/usr/bin/env python3
"""Evaluate horizontal-box (HBB) visual grounding with Acc@0.5, Acc@0.7 and mIoU.

Per evaluation split the script
  1. reads the refGeo metainfo JSONL (question, image_id, pixel-space ``bbox``);
  2. resolves the image under ``--image_dir/<subdir>``, tolerating extension mismatches;
  3. runs batched inference through ms-swift's TransformersEngine with the training prompt;
  4. parses the JSON answer ``[{"horizontal_bbox": [x1, y1, x2, y2]}]``;
  5. maps the box into original-image pixels according to ``--coord_mode``;
  6. scores axis-aligned IoU against the GT and derives Acc@0.5 / Acc@0.7 / mIoU;
  7. writes ``<split>_predictions.jsonl``, a summary JSON and a markdown table into a
     timestamped run directory, then prints the macro average over splits.

Coordinate modes (``--coord_mode``):
  norm1000          Qwen3-VL's training convention and the default: 0-1000 relative to
                    the original image, so x / 1000 * width gives pixels.
  absolute          the model already emits original-image pixels.
  resized_absolute  pixels of the smart-resized input that Qwen2.5-VL sees; the resize is
                    recomputed by ``_qwen_resized_size`` (factor 28, pixel budget from
                    IMAGE_MAX_TOKEN_NUM) so the box can be rescaled to the original image.

Scoring rules: only a schema-conforming ``horizontal_bbox`` scores, anything else is a 0;
missing images and failed batches stay in the results as zero-IoU rows, so the denominator
is always the full split; the headline number is the macro average over splits, not
weighted by split size.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_hbb.py \\
        --model_path models/checkpoints/GeoBox-R1 --dataset all
"""

# Set before importing the inference stack: the image token budget must equal the one used
# for training and rollout, and QWENVL_BBOX_FORMAT=new selects ms-swift's norm1000
# grounding format.
import os
os.environ["IMAGE_MAX_TOKEN_NUM"] = "1024"
os.environ["QWENVL_BBOX_FORMAT"] = "new"

import argparse
import glob
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

METRIC_KEYS = ("acc@0.5", "acc@0.7", "mIoU")
SCRIPT_TITLE = "HBB Evaluation · Acc@0.5 / Acc@0.7 / mIoU"
DEFAULT_TABLE_FILE = "table2_acc05_acc07_miou.md"
SUMMARY_FILE = "evaluation_hbb_acc05_acc07_miou_summary.json"
DEFAULT_COORD_MODE = "norm1000"
COORD_MODES = ("norm1000", "absolute", "resized_absolute")
QWEN_IMAGE_FACTOR = 28

# One entry per evaluation split: metainfo file, image sub-directory and a default batch
# size picked by image resolution (800x800 DIOR-RSVG tiles run at 60, 4000x2250 AVVG at 12).
DATASET_CONFIGS = {
    "dior_rsvg_test": {
        "test_file": "dior_rsvg_test.jsonl",
        "image_subdir": "DIOR-RSVG",
        "description": "DIOR-RSVG test split",
        "default_batch_size": 60,
    },
    "dior_rsvg_val": {
        "test_file": "dior_rsvg_val.jsonl",
        "image_subdir": "DIOR-RSVG",
        "description": "DIOR-RSVG validation split",
        "default_batch_size": 60,
    },
    "rsvg_test": {
        "test_file": "rsvg_test.jsonl",
        "image_subdir": "RSVG",
        "description": "RSVG test split",
        "default_batch_size": 60,
    },
    "rsvg_val": {
        "test_file": "rsvg_val.jsonl",
        "image_subdir": "RSVG",
        "description": "RSVG validation split",
        "default_batch_size": 60,
    },
    "geochat_test": {
        "test_file": "geochat_test.jsonl",
        "image_subdir": "GeoChat",
        "description": "GeoChat test split",
        "default_batch_size": 60,
    },
    "vrsbench_test": {
        "test_file": "vrsbench_test.jsonl",
        "image_subdir": "VRSBench",
        "description": "VRSBench test split",
        "default_batch_size": 96,
    },
    "avvg_test": {
        "test_file": "avvg_test.jsonl",
        "image_subdir": "AVVG",
        "description": "AVVG test split",
        "default_batch_size": 12,
    },
}

# Keep reporting order stable across runs so generated tables remain comparable.
TABLE_COLUMNS: Sequence[Tuple[str, str]] = [
    ("dior_rsvg_test", "DIOR-RSVG-Test"),
    ("dior_rsvg_val", "DIOR-RSVG-Val"),
    ("rsvg_test", "RSVG-Test"),
    ("rsvg_val", "RSVG-Val"),
    ("geochat_test", "GeoChat*"),
    ("vrsbench_test", "VRSBench*"),
    ("avvg_test", "AVVG"),
]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Axis-aligned IoU of two [x1, y1, x2, y2] boxes; a degenerate or empty union gives 0."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def parse_bbox_from_output(output: str) -> Optional[List[float]]:
    """Extract the first schema-conforming HBB from the raw model text.

    Code fences are stripped, the first ``[ {...} ]`` fragment is located with a regex and
    decoded with ``json.loads`` (never ``eval``), and the result only counts if it is a list
    whose first item is a dict carrying a four-number ``horizontal_bbox``. Anything else
    returns None and is scored as 0. The first fragment is the right one here: a plain HBB
    answer has a single block, and in a CoT-style answer the first block is the horizontal
    box.
    """
    try:
        output = output.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]", output)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict) and "horizontal_bbox" in item:
                    bbox = item["horizontal_bbox"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        return [float(x) for x in bbox]
    except Exception:
        pass
    return None


def get_prompt(question: str) -> str:
    """The HBB prompt, byte-identical to the training data and to demo/inference.py (the
    tests assert this); changing a character shifts the model's output distribution."""
    return f"""Locate the instance that matches the description: [{question}]. Report horizontal bbox coordinates in following JSON format:
```json
[
\t{{"horizontal_bbox": [x1, y1, x2, y2]}}
]
```"""


def load_test_data(test_file: str) -> List[dict]:
    """Load JSONL records, including records wrapped across physical lines."""
    data: List[dict] = []
    buffer = ""
    with open(test_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            buffer = f"{buffer} {line}" if buffer else line
            try:
                data.append(json.loads(buffer))
                buffer = ""
            except json.JSONDecodeError:
                continue
    if buffer:
        try:
            data.append(json.loads(buffer))
        except json.JSONDecodeError:
            print(f"Warning: ignored an incomplete JSON fragment at the end of {test_file}")
    return data


def _coord_mode_description(coord_mode: str) -> str:
    """Human-readable label for the coordinate mode, written to logs and summary files."""
    if coord_mode == "norm1000":
        return "norm1000 coordinates scaled to the original image"
    if coord_mode == "resized_absolute":
        return "absolute coordinates on the resized Qwen2.5-VL input image"
    return "absolute coordinates on the original image"


# The three helpers below replicate the rounding of qwen_vl_utils' smart_resize so that
# _qwen_resized_size reproduces the exact input resolution Qwen2.5-VL was given.
def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    import math
    return math.ceil(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    import math
    return math.floor(value / factor) * factor


def _qwen_resized_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    """(width, height) of the smart-resized Qwen-VL input for an original image size.

    Both sides are rounded to multiples of 28 and the area is squeezed into the token
    budget given by IMAGE_MIN_TOKEN_NUM / IMAGE_MAX_TOKEN_NUM. Only the
    ``resized_absolute`` coordinate mode needs this.
    """
    import math

    width, height = image_size
    factor = QWEN_IMAGE_FACTOR
    min_pixels = int(os.environ.get("IMAGE_MIN_TOKEN_NUM", "4")) * factor * factor
    max_pixels = int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1024")) * factor * factor
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return w_bar, h_bar


def _maybe_denormalize_bbox(
    bbox: List[float], image_size: Tuple[int, int], coord_mode: str
) -> List[float]:
    """Map model coordinates into original-image pixels before computing IoU.

    norm1000 divides by 1000 and multiplies by the original size; resized_absolute rescales
    from the smart-resized input; absolute is returned unchanged.
    """
    if not bbox or len(bbox) < 4:
        return bbox
    width, height = image_size
    bbox = bbox[:4]
    if coord_mode == "absolute":
        return [float(x) for x in bbox]
    if coord_mode == "resized_absolute":
        resized_width, resized_height = _qwen_resized_size(image_size)
        return [
            float(bbox[0]) / resized_width * width,
            float(bbox[1]) / resized_height * height,
            float(bbox[2]) / resized_width * width,
            float(bbox[3]) / resized_height * height,
        ]
    return [
        float(bbox[0]) / 1000.0 * width,
        float(bbox[1]) / 1000.0 * height,
        float(bbox[2]) / 1000.0 * width,
        float(bbox[3]) / 1000.0 * height,
    ]


def _extract_response_text(response) -> str:
    """Normalize the response shapes returned by supported inference engines."""
    if isinstance(response, list) and response:
        response = response[0]
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return str(response)


def _resolve_image_dir(image_base_dir: str, image_subdir: str) -> str:
    """Join the image root and the split sub-directory; the root may be a glob pattern."""
    candidate = os.path.join(image_base_dir, image_subdir)
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(candidate)
    if matches:
        return matches[0]
    return candidate


# Result-table helpers. The table has one row (this model) and, per split, three metric
# columns followed by the macro average; it is printed, saved per run and appended to a
# shared file so successive runs can be compared.
def _format_percent(value: Optional[float], decimals: int) -> str:
    """Fraction -> percent string; a missing metric renders as '-'."""
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}"


def _metric_title(metric_key: str) -> str:
    """Column title for a metric key."""
    return {"acc@0.5": "Acc@0.5", "acc@0.7": "Acc@0.7", "mIoU": "mIoU"}[metric_key]


def _metric_value(result: Dict, metric_key: str) -> Optional[float]:
    """Read one metric out of a per-split result dict, or None when absent."""
    metrics = result.get("metrics", {}) if result else {}
    return metrics.get(metric_key)


def _html_escape(text: str) -> str:
    """Escape model names and headers before they go into HTML table cells."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_table_row_values(
    results: List[Dict],
    columns: Sequence[Tuple[str, str]],
    model_name: str,
    decimals: int,
) -> List[str]:
    """One table row: model name, three metrics per split, then the macro averages."""
    result_map = {item.get("dataset"): item for item in results if item}
    row = [_html_escape(model_name)]
    avg_values: Dict[str, List[float]] = {metric: [] for metric in METRIC_KEYS}

    for dataset_key, _ in columns:
        item = result_map.get(dataset_key)
        for metric_key in METRIC_KEYS:
            val = _metric_value(item, metric_key) if item else None
            if val is not None:
                avg_values[metric_key].append(val)
            row.append(_format_percent(val, decimals))

    for metric_key in METRIC_KEYS:
        vals = avg_values[metric_key]
        row.append(_format_percent(sum(vals) / len(vals) if vals else None, decimals))
    return row


def _build_table_row_line(
    results: List[Dict],
    columns: Sequence[Tuple[str, str]],
    model_name: str,
    decimals: int,
    markdown: bool,
) -> str:
    """Render the row as an HTML <tr> (markdown mode) or as double-space separated text."""
    row = _build_table_row_values(results, columns, model_name, decimals)
    if markdown:
        cells = [f"<td>{row[0]}</td>"] + [f'<td align="right">{v}</td>' for v in row[1:]]
        return "  <tr>" + "".join(cells) + "</tr>"
    return "  ".join(row)


def _build_table_lines(
    results: List[Dict],
    columns: Sequence[Tuple[str, str]],
    model_name: str,
    decimals: int,
    markdown: bool,
) -> List[str]:
    """Build a two-level table grouped by dataset and metric."""
    metrics_per_group = len(METRIC_KEYS)
    groups = list(columns) + [("avg", "AVG")]

    if markdown:
        top = ['  <tr><th rowspan="2">Model</th>']
        for _, dataset_name in groups:
            top.append(f'<th colspan="{metrics_per_group}">{_html_escape(dataset_name)}</th>')
        top.append("</tr>")

        second = ["  <tr>"]
        for _ in groups:
            for metric_key in METRIC_KEYS:
                second.append(f"<th>{_html_escape(_metric_title(metric_key))}</th>")
        second.append("</tr>")

        return [
            f"# {SCRIPT_TITLE}",
            "",
            "<table>",
            "<thead>",
            "".join(top),
            "".join(second),
            "</thead>",
            "<tbody>",
            _build_table_row_line(results, columns, model_name, decimals, markdown=True),
            "</tbody>",
            "</table>",
        ]

    header_top = ["Model"]
    for _, dataset_name in groups:
        header_top.extend([dataset_name] + [""] * (metrics_per_group - 1))
    header_second = [""]
    for _ in groups:
        for metric_key in METRIC_KEYS:
            header_second.append(_metric_title(metric_key))
    return [
        SCRIPT_TITLE,
        "  ".join(header_top),
        "  ".join(header_second),
        _build_table_row_line(results, columns, model_name, decimals, markdown=False),
    ]

def _resolve_table_out(output_path: Optional[str], default_filename: str) -> Optional[str]:
    """Accept a file path or a directory; directories get the default table filename."""
    if not output_path:
        return None
    if output_path.endswith(os.sep):
        os.makedirs(output_path, exist_ok=True)
        return os.path.join(output_path, default_filename)
    if os.path.isdir(output_path):
        return os.path.join(output_path, default_filename)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return output_path


def _write_table_file(lines: List[str], output_path: str, default_filename: str) -> Optional[str]:
    """Append a timestamped result table to the target file."""
    resolved = _resolve_table_out(output_path, default_filename)
    if resolved is None:
        return None
    with open(resolved, "a", encoding="utf-8") as f:
        f.write(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for line in lines:
            f.write(line + "\n")
    return resolved


def _write_run_markdown(lines: List[str], run_output_dir: Optional[str]) -> Optional[str]:
    """Write this run's table next to its predictions (one file per run)."""
    if not run_output_dir:
        return None
    path = os.path.join(run_output_dir, "evaluation_metrics.md")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def _make_run_dir(base_dir: Optional[str]) -> Optional[str]:
    """Create a timestamped run directory so repeated runs never overwrite each other."""
    if base_dir is None:
        return None
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _metrics_from_ious(ious: List[float]) -> Dict[str, float]:
    """Acc@0.5, Acc@0.7 and mIoU from the per-sample IoUs of one split."""
    total = len(ious)
    if total == 0:
        return {metric: 0.0 for metric in METRIC_KEYS}
    all_metrics = {
        "acc@0.5": sum(1 for x in ious if x >= 0.5) / total,
        "acc@0.7": sum(1 for x in ious if x >= 0.7) / total,
        "mIoU": sum(ious) / total,
    }
    return {metric: all_metrics[metric] for metric in METRIC_KEYS}


def evaluate_dataset(
    engine,
    dataset_name: str,
    metainfo_dir: str,
    image_base_dir: str,
    output_dir: Optional[str],
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    current_index: Optional[int] = None,
    total_count: Optional[int] = None,
    coord_mode: str = DEFAULT_COORD_MODE,
) -> Dict:
    """Evaluate one split while preserving a result row for every valid annotation.

    Returns the per-split result dict (sample count, metrics, coordinate mode) and, when
    ``output_dir`` is set, writes ``<split>_predictions.jsonl`` with the raw output, the
    parsed box in original-image pixels, the GT and the IoU of every sample.
    """
    from swift.infer_engine.protocol import RequestConfig
    if coord_mode not in COORD_MODES:
        raise ValueError(f"coord_mode must be one of: {', '.join(COORD_MODES)}")

    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        print(f"Unknown dataset: {dataset_name}")
        return {}

    test_file = os.path.join(metainfo_dir, config["test_file"])
    image_dir = _resolve_image_dir(image_base_dir, config["image_subdir"])

    print(f"\n{'=' * 60}")
    progress = f" ({current_index}/{total_count})" if current_index and total_count else ""
    print(f"Evaluating: {config['description']}{progress}")
    print(f"Metadata: {test_file}")
    print(f"Images: {image_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Coordinate mode: {_coord_mode_description(coord_mode)}")
    print(f"Metrics: {', '.join(_metric_title(m) for m in METRIC_KEYS)}")
    print(f"{'=' * 60}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Metadata file not found: {test_file}")

    test_data = load_test_data(test_file)
    if max_samples is not None:
        test_data = test_data[:max_samples]
    print(f"Loaded {len(test_data)} samples")

    predictions: List[Dict] = []
    ious: List[float] = []
    total = 0

    # Missing images count as failed predictions instead of silently shrinking the denominator.
    valid_items: List[Dict] = []
    for item in test_data:
        question_id = item.get("question_id", item.get("id"))
        image_id = item["image_id"]
        question = item["question"]
        gt_bbox = item.get("bbox")
        if gt_bbox is None or len(gt_bbox) != 4:
            continue

        image_path = os.path.join(image_dir, image_id)
        if not os.path.exists(image_path):
            base_name = os.path.splitext(image_id)[0]
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                test_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break

        if not os.path.exists(image_path):
            predictions.append({
                "question_id": question_id,
                "output": "",
                "pred_bbox": None,
                "pred_bbox_display": None,
                "gt_bbox": gt_bbox,
                "iou": 0,
                "iou_display": f"{0:.4f}",
                "error": "Image not found",
            })
            ious.append(0.0)
            total += 1
            continue

        valid_items.append({
            "question_id": question_id,
            "image_path": image_path,
            "question": question,
            "gt_bbox": gt_bbox,
            "image_size": Image.open(image_path).size,
        })

    # Batched inference: one request per sample with the image passed by path and the
    # training prompt verbatim. 512 new tokens is ample for a single JSON box.
    request_config = RequestConfig(max_tokens=512)
    failed = 0
    for i in tqdm(range(0, len(valid_items), batch_size), desc=f"Evaluate {dataset_name} (batch={batch_size})"):
        batch_items = valid_items[i:i + batch_size]
        batch_requests = []
        for item in batch_items:
            prompt = get_prompt(item["question"])
            batch_requests.append({
                "messages": [{"role": "user", "content": f"<image>{prompt}"}],
                "images": [item["image_path"]],
            })

        try:
            responses = engine.infer(batch_requests, request_config=request_config, use_tqdm=False)
            outputs = [_extract_response_text(responses[j] if isinstance(responses, list) else responses)
                       for j in range(len(batch_items))]
            errors = [None] * len(batch_items)
        except Exception as e:
            # Usually an OOM on a shared GPU: retry one sample at a time instead of zeroing the batch.
            print(f"[warning] {dataset_name}: batch of {len(batch_items)} failed ({type(e).__name__}: {str(e)[:120]}); "
                  "retrying one sample at a time", flush=True)
            outputs, errors = [], []
            for request in batch_requests:
                try:
                    response = engine.infer([request], request_config=request_config, use_tqdm=False)
                    outputs.append(_extract_response_text(response[0] if isinstance(response, list) else response))
                    errors.append(None)
                except Exception as e2:
                    outputs.append("")
                    errors.append(f"{type(e2).__name__}: {e2}")

        for item, output, error in zip(batch_items, outputs, errors):
            pred_bbox = parse_bbox_from_output(output) if error is None else None
            if pred_bbox is not None:
                pred_bbox = _maybe_denormalize_bbox(pred_bbox, item["image_size"], coord_mode)
                iou = calculate_iou(pred_bbox, item["gt_bbox"])
            else:
                iou = 0.0
            record = {
                "question_id": item["question_id"],
                "output": output,
                "pred_bbox": pred_bbox,
                "pred_bbox_display": [round(x, 2) for x in pred_bbox] if pred_bbox is not None else None,
                "gt_bbox": item["gt_bbox"],
                "iou": iou,
                "iou_display": f"{iou:.4f}",
            }
            if error is not None:
                # Kept in the denominator as a zero so the metrics stay aligned with the input set.
                record["error"] = error
                failed += 1
            predictions.append(record)
            ious.append(iou)
            total += 1

    metrics = _metrics_from_ious(ious)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, f"{dataset_name}_predictions.jsonl")
        with open(pred_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {pred_file}")

    result = {
        "dataset": dataset_name,
        "total": total,
        "coord_mode": coord_mode,
        "coord_mode_description": _coord_mode_description(coord_mode),
        "metrics": metrics,
        "failed": failed,
        # Keep the legacy summary fields for downstream result readers.
        "correct": int(sum(1 for x in ious if x >= (0.5 if "acc@0.5" in METRIC_KEYS else 0.7))),
        "accuracy": metrics.get("acc@0.5", metrics.get("acc@0.7", 0.0)),
        "acc_percent": _format_percent(metrics.get("acc@0.5", metrics.get("acc@0.7", 0.0)), 2),
    }

    print(f"\nResults for {dataset_name}:")
    print(f"  Samples: {total}")
    if failed:
        print(f"  [warning] {failed} sample(s) failed even after per-sample retry and count as zero; see the 'error' field in the predictions")
    for metric in METRIC_KEYS:
        print(f"  {_metric_title(metric)}: {_format_percent(metrics.get(metric), 2)}%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=SCRIPT_TITLE)
    parser.add_argument("--model_path", type=str, default="models/pretrained/Qwen3-VL-4B-Instruct", help="Base model path")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Optional LoRA adapter checkpoint")
    parser.add_argument("--model_type", type=str, default="qwen3_vl", help="ms-swift model type")
    parser.add_argument("--attn_impl", type=str, default="flash_attn", help="Attention backend passed to ms-swift: flash_attn (needs the flash-attn package) or sdpa")
    parser.add_argument("--metainfo_dir", type=str, default="data/refGeo/metainfo", help="Directory containing evaluation JSONL files")
    parser.add_argument("--image_dir", type=str, default="data/refGeo/images", help="Evaluation image root")
    parser.add_argument("--output_dir", type=str, default="eval_results/hbb", help="Result directory")
    parser.add_argument("--dataset", type=str, default="all", choices=["all"] + list(DATASET_CONFIGS.keys()), help="Evaluation split, or all")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples per split; use 0 for a setup-only smoke test")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the per-split batch size")
    parser.add_argument(
        "--coord_mode",
        type=str,
        default=DEFAULT_COORD_MODE,
        choices=COORD_MODES,
        help="Output coordinates: norm1000, absolute, or resized_absolute",
    )
    parser.add_argument("--no-print-table", dest="print_table", action="store_false", help="Skip table output")
    parser.set_defaults(print_table=True)
    parser.add_argument("--table_format", type=str, default="markdown", choices=["text", "markdown"], help="Table format")
    parser.add_argument("--table_decimals", type=int, default=2, help="Decimal places in tables")
    parser.add_argument("--model_name", type=str, default="GeoBox-R1", help="Model label in tables")
    parser.add_argument("--table_out", type=str, default=None, help="File or directory for appended tables")
    args = parser.parse_args()
    coord_mode = args.coord_mode

    base_output_dir = args.output_dir
    run_output_dir = _make_run_dir(base_output_dir)

    print("=" * 60)
    print(SCRIPT_TITLE)
    print("=" * 60)
    print(f"Base model: {args.model_path}")
    print(f"Adapter: {args.checkpoint_dir if args.checkpoint_dir else 'none'}")
    print(f"Coordinate mode: {_coord_mode_description(coord_mode)}")
    print(f"Metrics: {', '.join(_metric_title(m) for m in METRIC_KEYS)}")
    if run_output_dir:
        print(f"Output directory: {run_output_dir}")
    print("=" * 60)

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    max_batch_size = args.batch_size if args.batch_size is not None else max(DATASET_CONFIGS[d].get("default_batch_size", 1) for d in datasets)

    print("\nLoading model...")
    import torch
    from swift.infer_engine import TransformersEngine

    # --checkpoint_dir applies a LoRA adapter on top of --model_path, so an un-merged
    # training run and a merged checkpoint are evaluated with the same command.
    engine = TransformersEngine(
        model=args.model_path,
        adapters=[args.checkpoint_dir] if args.checkpoint_dir else None,
        model_type=args.model_type if args.model_type else None,
        torch_dtype=torch.bfloat16,
        max_batch_size=max_batch_size,
        attn_impl=args.attn_impl,
    )

    all_results: List[Dict] = []
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        config = DATASET_CONFIGS[dataset_name]
        batch_size = args.batch_size if args.batch_size is not None else config.get("default_batch_size", 1)
        result = evaluate_dataset(
            engine=engine,
            dataset_name=dataset_name,
            metainfo_dir=args.metainfo_dir,
            image_base_dir=args.image_dir,
            output_dir=run_output_dir,
            max_samples=args.max_samples,
            coord_mode=coord_mode,
            batch_size=batch_size,
            current_index=idx,
            total_count=total_datasets,
        )
        all_results.append(result)

    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    header = f"{'Dataset':<20} {'Samples':>8} " + " ".join(f"{_metric_title(m):>10}" for m in METRIC_KEYS)
    print(header)
    print("-" * max(60, len(header)))
    # The headline average is macro-averaged across datasets, not weighted by split size.
    avg_metrics: Dict[str, float] = {}
    for metric in METRIC_KEYS:
        vals = [_metric_value(r, metric) for r in all_results if r]
        vals = [v for v in vals if v is not None]
        avg_metrics[metric] = sum(vals) / len(vals) if vals else 0.0
    for r in all_results:
        if not r:
            continue
        metric_text = " ".join(f"{_format_percent(_metric_value(r, m), 2) + '%':>10}" for m in METRIC_KEYS)
        print(f"{r['dataset']:<20} {r['total']:>8} {metric_text}")
    print("-" * max(60, len(header)))
    avg_text = " ".join(f"{_format_percent(avg_metrics.get(m), 2) + '%':>10}" for m in METRIC_KEYS)
    total_samples = sum(r.get("total", 0) for r in all_results if r)
    print(f"{'AVG (macro)':<20} {total_samples:>8} {avg_text}")
    print("=" * 60)

    if run_output_dir:
        summary_file = os.path.join(run_output_dir, SUMMARY_FILE)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": "HBB",
                "coord_mode": coord_mode,
                "coord_mode_description": _coord_mode_description(coord_mode),
                "metrics": list(METRIC_KEYS),
                "results": all_results,
                "average_metrics_unweighted_by_dataset": avg_metrics,
                "total_samples": total_samples,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to: {summary_file}")

    if args.print_table:
        lines = _build_table_lines(
            results=all_results,
            columns=TABLE_COLUMNS,
            model_name=args.model_name,
            decimals=args.table_decimals,
            markdown=(args.table_format == "markdown"),
        )
        print("\n" + "\n".join(lines))
        if args.table_out is None:
            args.table_out = base_output_dir
        saved_path = _write_table_file(lines, args.table_out, DEFAULT_TABLE_FILE)
        if saved_path:
            print(f"Appended the result table to: {saved_path}")
        run_md = _write_run_markdown(lines, run_output_dir)
        if run_md:
            print(f"Saved this run's table to: {run_md}")


if __name__ == "__main__":
    main()

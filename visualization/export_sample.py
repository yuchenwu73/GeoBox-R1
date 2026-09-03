#!/usr/bin/env python3
"""Export one cached sample as a per-model image gallery (full view plus a shared zoom).

Reads the caches written by compare_testsets.py (no inference runs here), picks
one sample by --question-id or --uid, and writes two images for GT and for each
selected model: the full image and a crop shared by all models, each showing the
GT polygon and that model's prediction with box labels only. This is the style
of the per-model qualitative examples.

Inputs
    <source-output-root>/<task>/cache/<model>/<dataset>.jsonl (compare_testsets.py output),
    data/refGeo/metainfo/<split>.jsonl and data/refGeo/images/<subset>/.

Outputs, under --output-root/<TASK>/<dataset>__<image>__<uid>__<query slug>__<run tag>/
    00_GT_OBB/, 01_Qwen3VL_OBB/, 02_GEOGROUND_OBB/, 03_SFT_OBB/, 04_SFT_GDPO_OBB/
    (suffix _HBB for HBB tasks), each with full_image/ and zoom_region/
    sample_record.json with --write-record

Label placement
    A label is attached to a corner of its polygon from the outside
    (build_touching_corner_candidates) and chosen by choose_label_rect so that it
    stays on the canvas, does not overlap the other label and covers as little of
    the polygons as possible. compute_scene_label_layouts tries GT-above /
    prediction-below, the reverse, and free placement, and keeps the best; the
    zoom view reuses the placement chosen for the full view when it still fits.

Usage, from the repository root
    python visualization/export_sample.py --task obb                       # first sample of geochat_test_filtered
    python visualization/export_sample.py --task obb --dataset geochat_test_filtered --question-id 4208
    python visualization/export_sample.py --task hbb --dataset rsvg_test --question-id 12 --models gt,sft,gdpo
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from compare_testsets import (
    HBB_DATASETS,
    IMAGE_BASE_DIR,
    OBB_DATASETS,
    compute_crop,
    ensure_parent,
    hbb_to_poly,
    load_cache_map,
    load_jsonl_multiline,
    poly_to_bbox,
    resolve_exact_image_path,
    resolve_metainfo_file,
    sample_uid,
    union_bbox,
)


# ---------------------------------------------------------------------------
# Constants and model specs
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent.parent
SOURCE_OUTPUT_ROOT = REPO_DIR / 'output' / 'visualizations' / 'visual_compare_all_testsets'
DEFAULT_OUTPUT_ROOT = REPO_DIR / 'output' / 'visualizations' / 'single_bbox_model_gallery'

GT_COLOR = (30, 200, 80)
QWEN_COLOR = (235, 80, 80)
GEOGROUND_COLOR = (180, 80, 255)
SFT_COLOR = (255, 170, 40)
GDPO_COLOR = (50, 150, 255)


@dataclass(frozen=True)
class ModelVizSpec:
    """One gallery column: model key, output folder, label, colour, and whether it has predictions."""

    key: str
    folder_name: str
    display_label: str
    color: Tuple[int, int, int]
    use_prediction: bool


GT_KEY = 'gt'
DEFAULT_MODEL_LABELS: Dict[str, str] = {
    'gt': 'GT',
    'qwen': 'Qwen3-VL-4B',
    'geoground': 'GeoGround',
    'sft': 'SFT',
    'gdpo': 'SFT+GDPO',
}

MODEL_SPECS: List[ModelVizSpec] = [
    ModelVizSpec('gt', '00_GT_OBB', DEFAULT_MODEL_LABELS['gt'], GT_COLOR, False),
    ModelVizSpec('qwen', '01_Qwen3VL_OBB', DEFAULT_MODEL_LABELS['qwen'], QWEN_COLOR, True),
    ModelVizSpec('geoground', '02_GEOGROUND_OBB', DEFAULT_MODEL_LABELS['geoground'], GEOGROUND_COLOR, True),
    ModelVizSpec('sft', '03_SFT_OBB', DEFAULT_MODEL_LABELS['sft'], SFT_COLOR, True),
    ModelVizSpec('gdpo', '04_SFT_GDPO_OBB', DEFAULT_MODEL_LABELS['gdpo'], GDPO_COLOR, True),
]
MODEL_SPEC_KEYS = tuple(spec.key for spec in MODEL_SPECS)
MODEL_KEYS = [spec.key for spec in MODEL_SPECS if spec.use_prediction]
# Corner preference orders for compute_label_rect / compute_scene_label_layouts.
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
CORNER_POSITION_ORDER = (
    'corner_top_left',
    'corner_top_right',
    'corner_bottom_left',
    'corner_bottom_right',
)
TOP_CORNER_POSITION_ORDER = (
    'corner_top_left',
    'corner_top_right',
)
BOTTOM_CORNER_POSITION_ORDER = (
    'corner_bottom_left',
    'corner_bottom_right',
)

TASK_DATASETS = {
    'obb': OBB_DATASETS,
    'hbb': HBB_DATASETS,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI; fills in the default split per task and validates --custom-metainfo-file, --label and --models."""
    parser = argparse.ArgumentParser(description='Export a single-model HBB/OBB visualization for one sample')
    parser.add_argument('--task', choices=['obb', 'hbb'], default='obb')
    parser.add_argument(
        '--models',
        default=','.join(spec.key for spec in MODEL_SPECS),
        help=f'Model keys to export, comma-separated; supported: {", ".join(MODEL_SPEC_KEYS)}',
    )
    parser.add_argument('--dataset', default=None, help='Dataset name; defaults to the default dataset of the chosen task')
    parser.add_argument('--question-id', type=int, default=None, help='Sample question_id; defaults to the first sample of the split')
    parser.add_argument('--uid', default=None, help='Optional; when given, the sample is looked up by uid first')
    parser.add_argument('--image-id', default=None, help='Optional; used together with question-id/uid as an extra check')
    parser.add_argument('--source-output-root', type=Path, default=SOURCE_OUTPUT_ROOT, help='Root directory of the existing cache')
    parser.add_argument('--custom-metainfo-file', type=Path, default=None, help='Custom metainfo jsonl holding one or a few samples')
    parser.add_argument('--custom-image-subdir', default=None, help='Image subdirectory for the custom metainfo, e.g. GeoChat')
    parser.add_argument('--custom-dataset-name', default=None, help='Custom dataset name; defaults to the metainfo file name')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help='New single-model export directory; defaults to output/visualizations/single_bbox_model_gallery',
    )
    parser.add_argument('--save-format', choices=['png', 'jpg'], default='png')
    parser.add_argument('--zoom-long-side', type=int, default=1400, help='Target long side of the zoom image; <=0 disables zooming')
    parser.add_argument('--line-width', type=int, default=5)
    parser.add_argument('--vertex-radius', type=int, default=4)
    parser.add_argument('--zoom-font-size', type=int, default=24, help='Label font size; slightly larger by default while keeping labels tight against the box')
    parser.add_argument(
        '--label',
        dest='label_overrides_raw',
        action='append',
        default=[],
        metavar='KEY=NAME',
        help=(
            'Custom label names, may be repeated; '
            f'supported keys: {", ".join(MODEL_SPEC_KEYS)}. '
            'Example: --label \'sft=GeoBox-R1 (SFT)\' --label \'gdpo=GeoBox (SFT+GDPO)\''
        ),
    )
    parser.add_argument('--run-tag', default=None, help='Optional output-directory suffix; generated automatically when omitted so earlier results are not overwritten')
    parser.add_argument('--write-record', action='store_true', help='Also write sample_record.json for traceability')
    args = parser.parse_args()
    if args.custom_metainfo_file is not None:
        if not args.custom_metainfo_file.exists():
            parser.error(f'Custom metainfo does not exist: {args.custom_metainfo_file}')
        if not args.custom_image_subdir:
            parser.error('--custom-image-subdir is required together with --custom-metainfo-file')
        custom_dataset_name = (args.custom_dataset_name or args.custom_metainfo_file.stem).strip()
        if not custom_dataset_name:
            parser.error('The custom dataset name must not be empty')
        if args.dataset is None:
            args.dataset = custom_dataset_name
        elif args.dataset != custom_dataset_name:
            parser.error(
                f'--dataset={args.dataset} disagrees with the custom dataset name {custom_dataset_name}; '
                'make them match or drop --dataset'
            )
    else:
        if args.dataset is None:
            args.dataset = 'geochat_test_filtered' if args.task == 'obb' else 'geochat_test'
        if args.dataset not in TASK_DATASETS[args.task]:
            parser.error(
                f'dataset {args.dataset} does not exist for task={args.task}; available: {", ".join(sorted(TASK_DATASETS[args.task].keys()))}'
            )
    try:
        args.label_overrides = parse_label_overrides(args.label_overrides_raw)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.model_keys = parse_model_keys(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    return args


# Labels and model selection are validated before any cache or image access.
def parse_label_overrides(raw_items: Sequence[str]) -> Dict[str, str]:
    """Parse repeated --label KEY=NAME options into a dict; keys must be model spec keys."""
    overrides: Dict[str, str] = {}
    for raw_item in raw_items:
        text = raw_item.strip()
        if not text:
            raise ValueError('A custom label must not be empty; use the KEY=NAME format')
        if '=' not in text:
            raise ValueError(f'Malformed custom label: {raw_item}; use the KEY=NAME format')
        key, label = text.split('=', 1)
        key = key.strip().lower()
        label = label.strip()
        if key not in MODEL_SPEC_KEYS:
            raise ValueError(f'Unknown label key: {key}; supported: {", ".join(MODEL_SPEC_KEYS)}')
        if not label:
            raise ValueError(f'The name for label {key} must not be empty')
        if key in overrides:
            raise ValueError(f'Label {key} was specified more than once; keep only one')
        overrides[key] = label
    return overrides


def parse_model_keys(raw_models: str) -> List[str]:
    """Comma-separated model keys -> ordered, de-duplicated list; unknown keys raise."""
    items = [item.strip().lower() for item in raw_models.split(',') if item.strip()]
    if not items:
        raise ValueError('At least one model key is required')
    invalid = [item for item in items if item not in MODEL_SPEC_KEYS]
    if invalid:
        raise ValueError(f'Unknown model key: {", ".join(invalid)}; supported: {", ".join(MODEL_SPEC_KEYS)}')
    deduped: List[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def resolve_model_specs(
    label_overrides: Optional[Dict[str, str]] = None,
    selected_keys: Optional[Sequence[str]] = None,
) -> List[ModelVizSpec]:
    """Selected specs in canonical order, with display labels replaced by the --label overrides."""
    overrides = label_overrides or {}
    wanted_keys = list(selected_keys or MODEL_SPEC_KEYS)
    return [
        ModelVizSpec(
            key=spec.key,
            folder_name=spec.folder_name,
            display_label=overrides.get(spec.key, spec.display_label),
            color=spec.color,
            use_prediction=spec.use_prediction,
        )
        for spec in MODEL_SPECS
        if spec.key in wanted_keys
    ]


# ---------------------------------------------------------------------------
# Sample and cache loading
# ---------------------------------------------------------------------------

def resolve_dataset_payload(task: str, dataset_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Metainfo file and image subdirectory of a split, custom or built-in."""
    if args.custom_metainfo_file is not None:
        custom_dataset_name = (args.custom_dataset_name or args.custom_metainfo_file.stem).strip()
        if dataset_name != custom_dataset_name:
            raise KeyError(f'In custom-dataset mode only dataset {custom_dataset_name} is supported, got {dataset_name}')
        return {
            'name': custom_dataset_name,
            'metainfo_file': args.custom_metainfo_file,
            'image_subdir': str(args.custom_image_subdir).strip(),
        }
    spec = TASK_DATASETS[task][dataset_name]
    return {
        'name': spec.name,
        # Same OBB fallback as compare_testsets.py (full split when the filtered file is absent).
        'metainfo_file': resolve_metainfo_file(spec),
        'image_subdir': spec.image_subdir,
    }


def load_samples_for_dataset(task: str, dataset_name: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """All samples of a split with the same uid scheme as compare_testsets.load_samples."""
    spec = resolve_dataset_payload(task, dataset_name, args)
    rows = load_jsonl_multiline(spec['metainfo_file'])
    image_dir = IMAGE_BASE_DIR / spec['image_subdir']
    samples: List[Dict[str, Any]] = []
    for item in rows:
        image_id = item['image_id']
        image_path = resolve_exact_image_path(image_dir, image_id)
        samples.append({
            'dataset': spec['name'],
            'task': task,
            'uid': sample_uid(item.get('question_id'), image_id, item.get('question', '')),
            'question_id': item.get('question_id'),
            'image_id': image_id,
            'image_path': str(image_path) if image_path else '',
            'question': item.get('question', ''),
            'gt_bbox': item.get('bbox'),
            'gt_poly': item.get('poly'),
        })
    return samples


def select_sample(samples: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Pick one sample: by --uid, else by --question-id, else the first sample of the split.

    --image-id narrows the candidates further. An explicit uid or question_id must
    match exactly one sample; the default takes the first candidate.
    """
    matches = list(samples)
    if args.uid:
        matches = [sample for sample in matches if sample['uid'] == args.uid]
    elif args.question_id is not None:
        matches = [sample for sample in matches if sample.get('question_id') == args.question_id]
    if args.image_id:
        matches = [sample for sample in matches if sample.get('image_id') == args.image_id]
    if not matches:
        raise FileNotFoundError(
            f'Sample not found: dataset={args.dataset}, question_id={args.question_id}, uid={args.uid}, image_id={args.image_id}'
        )
    explicit = bool(args.uid) or args.question_id is not None
    if explicit and len(matches) > 1:
        raise RuntimeError(f'The sample matched several results, narrow the query: {[sample["uid"] for sample in matches[:5]]}')
    sample = matches[0]
    if not sample['image_path']:
        raise FileNotFoundError(f'Sample image does not exist: {sample["image_id"]}')
    return sample


# Cached inference is intentionally decoupled from rendering so one run can feed many figures.
def cache_path(source_output_root: Path, task: str, dataset_name: str, model_name: str) -> Path:
    """Location of a compare_testsets.py cache file."""
    return source_output_root / task / 'cache' / model_name / f'{dataset_name}.jsonl'


def load_results_for_sample(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    model_keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Cached rows of the selected models for one sample; a missing cache or uid is an error."""
    results: Dict[str, Dict[str, Any]] = {}
    for model_name in model_keys:
        path = cache_path(args.source_output_root, args.task, args.dataset, model_name)
        if not path.exists():
            raise FileNotFoundError(f'Missing cache file: {path}')
        cache_map = load_cache_map(path)
        result = cache_map.get(sample['uid'])
        if result is None:
            raise FileNotFoundError(f'No result for uid={sample["uid"]} in {path}')
        results[model_name] = result
    return results


def to_polygon(result: Optional[Dict[str, Any]]) -> Optional[List[List[float]]]:
    """A cached prediction as four corners: the OBB when present, else the HBB as a rectangle."""
    if not result:
        return None
    pred_obb = result.get('pred_obb')
    if pred_obb:
        return [[float(x), float(y)] for x, y in pred_obb]
    pred_hbb = result.get('pred_hbb')
    if pred_hbb:
        return hbb_to_poly(pred_hbb)
    return None


def sample_gt_box(sample: Dict[str, Any]) -> Optional[List[float]]:
    """GT as [x1, y1, x2, y2]: the bbox field, else the envelope of the polygon."""
    gt_bbox = sample.get('gt_bbox')
    gt_poly = sample.get('gt_poly')
    if gt_bbox is not None:
        return [float(v) for v in gt_bbox]
    if gt_poly is not None:
        return poly_to_bbox(gt_poly)
    return None


def sample_gt_polygon(sample: Dict[str, Any], task: str) -> Optional[List[List[float]]]:
    """GT as four corners: the bbox for HBB tasks, otherwise the polygon (or the bbox as a rectangle)."""
    gt_poly = sample.get('gt_poly')
    gt_bbox = sample.get('gt_bbox')
    if task == 'hbb' and gt_bbox is not None:
        return hbb_to_poly(gt_bbox)
    if gt_poly is not None:
        return [[float(x), float(y)] for x, y in gt_poly]
    if gt_bbox is not None:
        return hbb_to_poly(gt_bbox)
    return None


def load_font(size: int = 16) -> ImageFont.ImageFont:
    """First available DejaVu font at the given size, else PIL's default bitmap font."""
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------
# Rectangles are (x1, y1, x2, y2) integer pixels. Candidates are scored to stay on the
# canvas, avoid the other label, and cover as little of the polygons as possible.
def rect_intersection_area(
    rect1: Tuple[int, int, int, int],
    rect2: Tuple[int, int, int, int],
) -> int:
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def clamp_rect_to_canvas(
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    margin: int = 4,
) -> Tuple[int, int, int, int]:
    """Shift a rectangle back inside the canvas margin without resizing it."""
    x1, y1, x2, y2 = rect
    canvas_w, canvas_h = canvas_size
    dx = 0
    dy = 0
    if x1 < margin:
        dx = margin - x1
    elif x2 > canvas_w - margin:
        dx = (canvas_w - margin) - x2
    if y1 < margin:
        dy = margin - y1
    elif y2 > canvas_h - margin:
        dy = (canvas_h - margin) - y2
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def rect_outside_area(
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    margin: int = 4,
) -> int:
    """Area of a rectangle that lies outside the canvas margin."""
    x1, y1, x2, y2 = rect
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    canvas_w, canvas_h = canvas_size
    inner_rect = (
        margin,
        margin,
        max(margin, canvas_w - margin),
        max(margin, canvas_h - margin),
    )
    inside_area = rect_intersection_area(rect, inner_rect)
    return max(0, area - inside_area)


def box_to_rect(box: Sequence[float]) -> Tuple[int, int, int, int]:
    return (
        int(round(float(box[0]))),
        int(round(float(box[1]))),
        int(round(float(box[2]))),
        int(round(float(box[3]))),
    )


def nearest_point_to_target(
    points: Sequence[Tuple[float, float]],
    target: Tuple[int, int],
) -> Tuple[int, int]:
    """Polygon corner closest to target, rounded to ints."""
    tx, ty = target
    px, py = min(points, key=lambda point: (point[0] - tx) ** 2 + (point[1] - ty) ** 2)
    return int(round(px)), int(round(py))


def polygon_signed_area(points: Sequence[Tuple[float, float]]) -> float:
    """Shoelace area; the sign encodes the vertex winding."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    return abs(polygon_signed_area(points))


def clip_polygon_with_boundary(
    points: Sequence[Tuple[float, float]],
    inside_fn,
    intersect_fn,
) -> List[Tuple[float, float]]:
    """Clip a polygon against one half-plane while preserving vertex order."""
    if not points:
        return []
    output: List[Tuple[float, float]] = []
    prev = points[-1]
    prev_inside = inside_fn(prev)
    for curr in points:
        curr_inside = inside_fn(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect_fn(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect_fn(prev, curr))
        prev = curr
        prev_inside = curr_inside
    return output


def polygon_rect_intersection_area(
    points: Sequence[Tuple[float, float]],
    rect: Tuple[int, int, int, int],
) -> float:
    """Measure label overlap by clipping the target polygon to the label rectangle."""
    if len(points) < 3:
        return 0.0
    x1, y1, x2, y2 = rect
    clipped: List[Tuple[float, float]] = list(points)
    clipped = clip_polygon_with_boundary(
        clipped,
        inside_fn=lambda point: point[0] >= x1,
        intersect_fn=lambda p1, p2: (
            float(x1),
            float(p1[1] + (p2[1] - p1[1]) * ((x1 - p1[0]) / (p2[0] - p1[0]))),
        ),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        inside_fn=lambda point: point[0] <= x2,
        intersect_fn=lambda p1, p2: (
            float(x2),
            float(p1[1] + (p2[1] - p1[1]) * ((x2 - p1[0]) / (p2[0] - p1[0]))),
        ),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        inside_fn=lambda point: point[1] >= y1,
        intersect_fn=lambda p1, p2: (
            float(p1[0] + (p2[0] - p1[0]) * ((y1 - p1[1]) / (p2[1] - p1[1]))),
            float(y1),
        ),
    )
    clipped = clip_polygon_with_boundary(
        clipped,
        inside_fn=lambda point: point[1] <= y2,
        intersect_fn=lambda p1, p2: (
            float(p1[0] + (p2[0] - p1[0]) * ((y2 - p1[1]) / (p2[1] - p1[1]))),
            float(y2),
        ),
    )
    if len(clipped) < 3:
        return 0.0
    return polygon_area(clipped)


def choose_label_rect(
    candidates: Dict[str, Tuple[int, int, int, int]],
    box_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    occupied_boxes: Sequence[Tuple[int, int, int, int]],
    avoid_polygons: Sequence[Sequence[Tuple[float, float]]],
    position_order: Sequence[str],
) -> Tuple[str, Tuple[int, int, int, int]]:
    """Pick the best candidate in position_order.

    Ranking: least area off-canvas, then least overlap with occupied_boxes, then
    least coverage of avoid_polygons, then the position order itself; the scan
    stops at the first perfect candidate. Falls back to the first candidate.
    """
    box_w, box_h = box_size
    best_key: Optional[str] = None
    best_rect: Optional[Tuple[int, int, int, int]] = None
    best_score: Optional[Tuple[int, int, float, int]] = None
    for idx, key in enumerate(position_order):
        if key not in candidates:
            continue
        raw_rect = candidates[key]
        overflow_area = rect_outside_area(raw_rect, canvas_size)
        rect = clamp_rect_to_canvas(raw_rect, canvas_size)
        overlap_with_labels = sum(rect_intersection_area(rect, other) for other in occupied_boxes)
        overlap_with_polygons = sum(polygon_rect_intersection_area(poly, rect) for poly in avoid_polygons)
        score = (overflow_area, overlap_with_labels, overlap_with_polygons, idx)
        if best_score is None or score < best_score:
            best_key = key
            best_rect = rect
            best_score = score
            if score == (0, 0, 0.0, idx):
                break
    if best_rect is None or best_key is None:
        if candidates:
            first_key = next(iter(candidates.keys()))
            return first_key, clamp_rect_to_canvas(candidates[first_key], canvas_size)
        fallback = (4, 4, 4 + box_w, 4 + box_h)
        return 'fallback', fallback
    return best_key, best_rect


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    text: str,
    color: Tuple[int, int, int],
    font_size: int = 24,
) -> None:
    """Draw a filled label rectangle with white text."""
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    x1, y1, x2, y2 = rect
    pad_x = max(5, int(round(font_size * 0.22)))
    pad_y = max(3, int(round(font_size * 0.14)))
    text_x = x1 + pad_x - bbox[0]
    text_y = y1 + pad_y - bbox[1]
    draw.rectangle([x1, y1, x2, y2], fill=color)
    draw.text((text_x, text_y), text, fill='white', font=font)


def build_touching_corner_candidates(
    points: Sequence[Tuple[float, float]],
    box_w: int,
    box_h: int,
    canvas_size: Tuple[int, int],
) -> Dict[str, Tuple[int, int, int, int]]:
    """Label rectangles that touch a polygon corner from the outside (above the top
    corners, below the bottom ones, extending inward); only those that fit on the canvas."""
    canvas_w, canvas_h = canvas_size
    x1, y1, x2, y2 = box_to_rect(poly_to_bbox(points))
    anchors = {
        'corner_top_left': nearest_point_to_target(points, (x1, y1)),
        'corner_top_right': nearest_point_to_target(points, (x2, y1)),
        'corner_bottom_left': nearest_point_to_target(points, (x1, y2)),
        'corner_bottom_right': nearest_point_to_target(points, (x2, y2)),
    }

    def place_candidate(
        anchor: Tuple[int, int],
        vertical: str,
        horizontal: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        ax, ay = anchor
        if vertical == 'above':
            top = ay - box_h
            bottom = ay
            if top < 0:
                return None
        else:
            top = ay
            bottom = ay + box_h
            if bottom > canvas_h:
                return None

        if horizontal == 'rightward':
            left = min(max(0, ax), max(0, canvas_w - box_w))
            right = left + box_w
        else:
            left = max(0, min(ax - box_w, canvas_w - box_w))
            right = left + box_w

        if not (left <= ax <= right):
            return None
        return (int(left), int(top), int(right), int(bottom))

    candidates: Dict[str, Tuple[int, int, int, int]] = {}
    placements = {
        'corner_top_left': ('above', 'rightward'),
        'corner_top_right': ('above', 'leftward'),
        'corner_bottom_left': ('below', 'rightward'),
        'corner_bottom_right': ('below', 'leftward'),
    }
    for key, (vertical, horizontal) in placements.items():
        rect = place_candidate(anchors[key], vertical, horizontal)
        if rect is not None:
            candidates[key] = rect
    return candidates


def compute_label_rect(
    image_size: Tuple[int, int],
    poly: Sequence[Sequence[float]],
    label: str,
    label_font_size: int = 24,
    occupied_label_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    avoid_polygons: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    position_order: Optional[Sequence[str]] = None,
    preferred_key: Optional[str] = None,
) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
    """Choose a corner-touching label rectangle for a polygon.

    The rectangle is sized from the font, candidates come from
    build_touching_corner_candidates, and choose_label_rect steers clear of
    occupied_label_boxes and avoid_polygons (the list of occupied boxes is updated
    in place). A preferred_key, the placement used in the full-image view, is kept
    when it does not collide, so the full and zoom views label consistently.
    """
    points = [(float(x), float(y)) for x, y in poly]
    if len(points) < 2 or not label:
        return None
    font = load_font(label_font_size)
    text_bbox = font.getbbox(label)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad_x = max(5, int(round(label_font_size * 0.22)))
    pad_y = max(3, int(round(label_font_size * 0.14)))
    label_box_w = text_w + pad_x * 2
    label_box_h = text_h + pad_y * 2
    candidates = build_touching_corner_candidates(points, label_box_w, label_box_h, image_size)
    if not candidates:
        return None
    resolved_occupied = occupied_label_boxes if occupied_label_boxes is not None else []
    resolved_avoid_polygons = [
        [(float(x), float(y)) for x, y in other_poly]
        for other_poly in (avoid_polygons or [points])
        if other_poly
    ]
    if not resolved_avoid_polygons:
        resolved_avoid_polygons = [points]
    order = tuple(position_order or CORNER_POSITION_ORDER)
    if preferred_key and preferred_key in candidates:
        preferred_rect = candidates[preferred_key]
        overlap_with_labels = sum(rect_intersection_area(preferred_rect, other) for other in resolved_occupied)
        if overlap_with_labels == 0:
            resolved_occupied.append(preferred_rect)
            return preferred_key, preferred_rect
        order = (preferred_key,) + tuple(key for key in order if key != preferred_key)
    label_key, label_rect = choose_label_rect(
        candidates=candidates,
        box_size=(label_box_w, label_box_h),
        canvas_size=image_size,
        occupied_boxes=resolved_occupied,
        avoid_polygons=resolved_avoid_polygons,
        position_order=order,
    )
    resolved_occupied.append(label_rect)
    return label_key, label_rect


def scene_layout_score(
    layouts: Dict[str, Optional[Tuple[str, Tuple[int, int, int, int]]]],
    gt_poly: Sequence[Sequence[float]],
    pred_poly: Optional[Sequence[Sequence[float]]],
) -> Tuple[int, int, float]:
    """Rank a GT/prediction label layout: fewer missing labels, then less label overlap, then less polygon coverage."""
    gt_layout = layouts.get('gt')
    pred_layout = layouts.get('pred')
    missing_count = int(gt_layout is None) + int(pred_layout is None)
    label_overlap = 0
    if gt_layout is not None and pred_layout is not None:
        label_overlap = rect_intersection_area(gt_layout[1], pred_layout[1])
    polygons = [gt_poly]
    if pred_poly is not None:
        polygons.append(pred_poly)
    polygon_overlap = 0.0
    for layout in (gt_layout, pred_layout):
        if layout is None:
            continue
        rect = layout[1]
        polygon_overlap += sum(
            polygon_rect_intersection_area([(float(x), float(y)) for x, y in poly], rect)
            for poly in polygons
            if poly is not None
        )
    return missing_count, label_overlap, polygon_overlap


# ---------------------------------------------------------------------------
# Drawing, crops and rendering
# ---------------------------------------------------------------------------

def draw_polygon(
    image: Image.Image,
    poly: Sequence[Sequence[float]],
    color: Tuple[int, int, int],
    line_width: int,
    vertex_radius: int,
    label: Optional[str] = None,
    label_font_size: int = 24,
    occupied_label_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    avoid_polygons: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    position_order: Optional[Sequence[str]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Draw a polygon outline with corner dots and, optionally, its label; returns the label rectangle."""
    draw = ImageDraw.Draw(image)
    points = [(float(x), float(y)) for x, y in poly]
    if len(points) < 2:
        return None
    for idx in range(len(points)):
        draw.line([points[idx], points[(idx + 1) % len(points)]], fill=color, width=line_width)
    if vertex_radius > 0:
        for x, y in points:
            draw.ellipse(
                [x - vertex_radius, y - vertex_radius, x + vertex_radius, y + vertex_radius],
                fill=color,
            )
    if label:
        resolved_occupied = occupied_label_boxes if occupied_label_boxes is not None else []
        layout = compute_label_rect(
            image.size,
            points,
            label,
            label_font_size=label_font_size,
            occupied_label_boxes=resolved_occupied,
            avoid_polygons=avoid_polygons,
            position_order=position_order or CORNER_POSITION_ORDER,
        )
        if layout is not None:
            _, label_rect = layout
            draw_label_box(draw, label_rect, label, color, font_size=label_font_size)
            return label_rect
    return None


def compute_scene_label_layouts(
    image_size: Tuple[int, int],
    gt_poly: Sequence[Sequence[float]],
    pred_poly: Optional[Sequence[Sequence[float]]],
    gt_label: str,
    pred_label: str,
    label_font_size: int,
    preferred_label_keys: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Optional[Tuple[str, Tuple[int, int, int, int]]]]:
    """Place the GT and prediction labels together.

    Three preference orders are tried (GT above / prediction below, the reverse,
    and both free) and the layout with the best scene_layout_score wins, which
    avoids the collisions a greedy one-at-a-time placement produces.
    """
    if pred_poly is None:
        return {
            'gt': compute_label_rect(
                image_size,
                gt_poly,
                gt_label,
                label_font_size=label_font_size,
                occupied_label_boxes=[],
                avoid_polygons=[gt_poly],
                position_order=CORNER_POSITION_ORDER,
                preferred_key=(preferred_label_keys or {}).get('gt'),
            ),
            'pred': None,
        }

    candidate_layouts: List[Dict[str, Optional[Tuple[str, Tuple[int, int, int, int]]]]] = []
    order_specs = (
        {'gt': TOP_CORNER_POSITION_ORDER, 'pred': BOTTOM_CORNER_POSITION_ORDER},
        {'gt': BOTTOM_CORNER_POSITION_ORDER, 'pred': TOP_CORNER_POSITION_ORDER},
        {'gt': CORNER_POSITION_ORDER, 'pred': CORNER_POSITION_ORDER},
    )
    for order_spec in order_specs:
        occupied_label_boxes: List[Tuple[int, int, int, int]] = []
        layouts: Dict[str, Optional[Tuple[str, Tuple[int, int, int, int]]]] = {'gt': None, 'pred': None}
        layouts['gt'] = compute_label_rect(
            image_size,
            gt_poly,
            gt_label,
            label_font_size=label_font_size,
            occupied_label_boxes=occupied_label_boxes,
            avoid_polygons=[poly for poly in [gt_poly, pred_poly] if poly is not None],
            position_order=order_spec['gt'],
            preferred_key=(preferred_label_keys or {}).get('gt'),
        )
        layouts['pred'] = compute_label_rect(
            image_size,
            pred_poly,
            pred_label,
            label_font_size=label_font_size,
            occupied_label_boxes=occupied_label_boxes,
            avoid_polygons=[poly for poly in [pred_poly, gt_poly] if poly is not None],
            position_order=order_spec['pred'],
            preferred_key=(preferred_label_keys or {}).get('pred'),
        )
        candidate_layouts.append(layouts)
    return min(candidate_layouts, key=lambda item: scene_layout_score(item, gt_poly, pred_poly))


def shift_polygon(poly: Sequence[Sequence[float]], dx: float, dy: float) -> List[List[float]]:
    return [[float(x) - dx, float(y) - dy] for x, y in poly]


def scale_polygon(poly: Sequence[Sequence[float]], sx: float, sy: float) -> List[List[float]]:
    return [[float(x) * sx, float(y) * sy] for x, y in poly]


def slugify_prompt(prompt: str, max_len: int = 80) -> str:
    """Filesystem-safe slug of the query, truncated and hashed when too long."""
    text = prompt.strip().lower()
    text = re.sub(r'[^\w]+', '_', text, flags=re.UNICODE)
    text = re.sub(r'_+', '_', text).strip('_')
    if not text:
        return 'no_prompt'
    if len(text) <= max_len:
        return text
    digest = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
    keep_len = max(12, max_len - len(digest) - 2)
    return f'{text[:keep_len].rstrip("_")}__{digest}'


def compute_shared_crop_box(
    sample: Dict[str, Any],
    model_results: Dict[str, Dict[str, Any]],
    task: str,
) -> Tuple[int, int, int, int]:
    """Fit GT and every prediction into one crop shared across model panels."""
    gt_box = sample_gt_box(sample)
    if gt_box is None:
        raise ValueError(f'The {task.upper()} sample has no GT box, cannot continue')
    focus_boxes: List[List[float]] = [gt_box]
    for model_name in MODEL_KEYS:
        pred_poly = to_polygon(model_results.get(model_name))
        if pred_poly is not None:
            focus_boxes.append(poly_to_bbox(pred_poly))
    if not focus_boxes:
        raise RuntimeError('No box available to crop around')
    image = Image.open(sample['image_path']).convert('RGB')
    return compute_crop(image.size, union_bbox(*focus_boxes))


# Every model uses the same crop, making geometric differences directly comparable.
def render_full_image(
    image: Image.Image,
    gt_poly: Sequence[Sequence[float]],
    pred_poly: Optional[Sequence[Sequence[float]]],
    gt_label: str,
    pred_label: str,
    pred_color: Tuple[int, int, int],
    args: argparse.Namespace,
) -> Tuple[Image.Image, Dict[str, Optional[str]]]:
    """Full image with the GT polygon, the prediction and both labels.

    Returns the image and the label placements used, which the zoom view reuses.
    """
    canvas = image.copy().convert('RGB')
    draw_polygon(
        canvas,
        gt_poly,
        GT_COLOR,
        args.line_width,
        args.vertex_radius,
    )
    if pred_poly is not None:
        draw_polygon(
            canvas,
            pred_poly,
            pred_color,
            args.line_width,
            args.vertex_radius,
        )
    draw = ImageDraw.Draw(canvas)
    layouts = compute_scene_label_layouts(
        canvas.size,
        gt_poly,
        pred_poly,
        gt_label,
        pred_label,
        label_font_size=args.zoom_font_size,
    )
    gt_layout = layouts.get('gt')
    if gt_layout is not None:
        _, gt_label_rect = gt_layout
        draw_label_box(draw, gt_label_rect, gt_label, GT_COLOR, font_size=args.zoom_font_size)
    pred_layout = layouts.get('pred')
    if pred_layout is not None:
        _, pred_label_rect = pred_layout
        draw_label_box(draw, pred_label_rect, pred_label, pred_color, font_size=args.zoom_font_size)
    label_keys = {
        'gt': gt_layout[0] if gt_layout is not None else None,
        'pred': pred_layout[0] if pred_layout is not None else None,
    }
    return canvas, label_keys


def resize_with_long_side(image: Image.Image, long_side: int) -> Tuple[Image.Image, float, float]:
    """Resize so the long side equals long_side; returns the image and the x / y scale factors."""
    if long_side <= 0:
        return image, 1.0, 1.0
    src_w, src_h = image.size
    current_long_side = max(src_w, src_h)
    if current_long_side == 0 or current_long_side == long_side:
        return image, 1.0, 1.0
    scale = long_side / float(current_long_side)
    new_size = (
        max(1, int(round(src_w * scale))),
        max(1, int(round(src_h * scale))),
    )
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized, new_size[0] / src_w, new_size[1] / src_h


def render_zoom_image(
    image: Image.Image,
    crop_box: Sequence[int],
    gt_poly: Sequence[Sequence[float]],
    pred_poly: Optional[Sequence[Sequence[float]]],
    gt_label: str,
    pred_label: str,
    pred_color: Tuple[int, int, int],
    args: argparse.Namespace,
    preferred_label_keys: Optional[Dict[str, Optional[str]]] = None,
) -> Image.Image:
    """Crop to crop_box, upscale to --zoom-long-side, and redraw both polygons with
    proportionally thicker lines and larger labels."""
    cl, ct, cr, cb = [int(v) for v in crop_box]
    crop = image.crop((cl, ct, cr, cb)).convert('RGB')
    crop, sx, sy = resize_with_long_side(crop, args.zoom_long_side)
    scale = max(1.0, min(sx, sy))
    zoom_line_width = max(args.line_width, min(18, int(round(args.line_width * scale))))
    zoom_vertex_radius = max(args.vertex_radius, min(12, int(round(args.vertex_radius * scale))))
    zoom_font_size = max(
        args.zoom_font_size + 4,
        min(40, int(round(args.zoom_font_size * min(scale, 1.6)))),
    )
    shifted_gt = scale_polygon(shift_polygon(gt_poly, cl, ct), sx, sy)
    shifted_pred = scale_polygon(shift_polygon(pred_poly, cl, ct), sx, sy) if pred_poly is not None else None
    draw_polygon(
        crop,
        shifted_gt,
        GT_COLOR,
        zoom_line_width,
        zoom_vertex_radius,
    )
    if shifted_pred is not None:
        draw_polygon(
            crop,
            shifted_pred,
            pred_color,
            zoom_line_width,
            zoom_vertex_radius,
        )
    draw = ImageDraw.Draw(crop)
    layouts = compute_scene_label_layouts(
        crop.size,
        shifted_gt,
        shifted_pred,
        gt_label,
        pred_label,
        label_font_size=zoom_font_size,
        preferred_label_keys=preferred_label_keys,
    )
    gt_layout = layouts.get('gt')
    if gt_layout is not None:
        _, gt_label_rect = gt_layout
        draw_label_box(draw, gt_label_rect, gt_label, GT_COLOR, font_size=zoom_font_size)
    pred_layout = layouts.get('pred')
    if pred_layout is not None:
        _, pred_label_rect = pred_layout
        draw_label_box(draw, pred_label_rect, pred_label, pred_color, font_size=zoom_font_size)
    return crop


# ---------------------------------------------------------------------------
# Output paths and records
# ---------------------------------------------------------------------------

def sample_tag(sample: Dict[str, Any]) -> str:
    """Directory-safe sample identifier: dataset, image stem, uid and query slug."""
    image_stem = Path(sample['image_id']).stem
    prompt_slug = slugify_prompt(sample.get('question', ''))
    return f'{sample["dataset"]}__{image_stem}__{sample["uid"]}__{prompt_slug}'


def resolve_run_tag(run_tag: Optional[str]) -> str:
    """Sanitised --run-tag, or a timestamp so repeated exports never overwrite each other."""
    raw = run_tag or datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    text = re.sub(r'[^\w.-]+', '_', raw, flags=re.UNICODE).strip('._')
    return text or datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')


def build_run_root(args: argparse.Namespace, sample: Dict[str, Any]) -> Path:
    """Output directory of one export: <output-root>/<TASK>/<sample tag>__<run tag>."""
    tag = sample_tag(sample)
    run_tag = resolve_run_tag(args.run_tag)
    return args.output_root / args.task.upper() / f'{tag}__{run_tag}'


def save_image(path: Path, image: Image.Image, save_format: str) -> None:
    ensure_parent(path)
    if save_format == 'jpg':
        image.save(path, quality=95)
    else:
        image.save(path)


def build_output_paths(args: argparse.Namespace, sample: Dict[str, Any], model_spec: ModelVizSpec) -> Dict[str, Path]:
    """full_image/ and zoom_region/ paths for one model; folder names switch _OBB -> _HBB for HBB tasks."""
    tag = sample_tag(sample)
    folder_name = model_spec.folder_name if args.task == 'obb' else model_spec.folder_name.replace('_OBB', '_HBB')
    base_dir = build_run_root(args, sample) / folder_name
    file_name = f'{tag}.{args.save_format}'
    return {
        'full': base_dir / 'full_image' / file_name,
        'zoom': base_dir / 'zoom_region' / file_name,
    }


def write_record_file(
    args: argparse.Namespace,
    sample: Dict[str, Any],
    model_results: Dict[str, Dict[str, Any]],
    crop_box: Sequence[int],
    model_keys: Sequence[str],
) -> Path:
    """Write sample_record.json with the crop box and every model's cached prediction."""
    record_path = build_run_root(args, sample) / 'sample_record.json'
    payload: Dict[str, Any] = {
        'task': args.task,
        'dataset': sample['dataset'],
        'question_id': sample['question_id'],
        'uid': sample['uid'],
        'image_id': sample['image_id'],
        'image_path': sample['image_path'],
        'question': sample['question'],
        'crop_box': list(crop_box),
        'models': {},
    }
    for model_name in model_keys:
        result = model_results[model_name]
        payload['models'][model_name] = {
            'status': result.get('status'),
            'pred_kind': result.get('pred_kind'),
            'pred_hbb': result.get('pred_hbb'),
            'pred_obb': result.get('pred_obb'),
        }
    ensure_parent(record_path)
    record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return record_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Select the sample, load its cached results, and render full and zoom views per model."""
    args = parse_args()
    args.run_tag = resolve_run_tag(args.run_tag)
    model_specs = resolve_model_specs(getattr(args, 'label_overrides', None), getattr(args, 'model_keys', None))
    gt_spec = next(spec for spec in model_specs if spec.key == GT_KEY)
    pred_model_keys = [spec.key for spec in model_specs if spec.use_prediction]
    samples = load_samples_for_dataset(args.task, args.dataset, args)
    sample = select_sample(samples, args)
    model_results = load_results_for_sample(sample, args, pred_model_keys)
    crop_box = compute_shared_crop_box(sample, model_results, args.task)

    image = Image.open(sample['image_path']).convert('RGB')
    gt_poly = sample_gt_polygon(sample, args.task)
    if not gt_poly:
        raise ValueError(f'This sample has no drawable GT box: task={args.task}, dataset={args.dataset}, uid={sample["uid"]}')

    written_paths: List[Path] = []
    for model_spec in model_specs:
        pred_poly = None if not model_spec.use_prediction else to_polygon(model_results[model_spec.key])
        output_paths = build_output_paths(args, sample, model_spec)
        full_image, label_keys = render_full_image(
            image,
            gt_poly,
            pred_poly,
            gt_spec.display_label,
            model_spec.display_label,
            model_spec.color,
            args,
        )
        zoom_image = render_zoom_image(
            image,
            crop_box,
            gt_poly,
            pred_poly,
            gt_spec.display_label,
            model_spec.display_label,
            model_spec.color,
            args,
            preferred_label_keys=label_keys,
        )
        save_image(output_paths['full'], full_image, args.save_format)
        save_image(output_paths['zoom'], zoom_image, args.save_format)
        written_paths.extend([output_paths['full'], output_paths['zoom']])

    record_path = None
    if args.write_record:
        record_path = write_record_file(args, sample, model_results, crop_box, pred_model_keys)

    print(f'Sample: task={args.task}, dataset={sample["dataset"]}, question_id={sample["question_id"]}, uid={sample["uid"]}')
    print(f'Image: {sample["image_path"]}')
    print(f'Run tag: {args.run_tag}')
    print(f'Shared crop box: {list(crop_box)}')
    print('Files written:')
    for path in written_paths:
        print(f'- {path}')
    if record_path is not None:
        print(f'- {record_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

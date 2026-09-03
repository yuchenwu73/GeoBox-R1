#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch inference and five-panel comparison figures across the evaluation splits.

Purpose
    Run the compared models on the refGeo evaluation splits, cache every
    prediction, and render one figure per sample with the panels
    GT | QWEN3-VL-4B | GEOGROUND | SFT | SFT+GDPO (see MODEL_ORDER). The model
    keys are qwen (base Qwen3-VL-4B-Instruct), geoground, sft (GeoBox-R1-SFT)
    and gdpo (GeoBox-R1). These are the qualitative comparison figures.

Inputs
    data/refGeo/metainfo/<split>.jsonl and data/refGeo/images/<subset>/ for the
    seven HBB splits; data/refGeo/metainfo/OBB_Selected/*_filtered.jsonl for the
    three OBB splits, falling back to the full *_test.jsonl files when the filtered
    ones are absent. Checkpoints default to models/pretrained/Qwen3-VL-4B-Instruct,
    models/checkpoints/GeoBox-R1-SFT, models/checkpoints/GeoBox-R1 and the
    GeoGround LLaVA-1.5 checkpoint under models/pretrained/. Workers run inside
    conda environments: --conda-env (default: the active one) for the Qwen family and
    --geoground-conda-env (default geoground when CONDA_ENVS_DIR contains it).

Outputs (under --output-root, default output/visualizations/visual_compare_all_testsets)
    <task>/cache/<model>/<dataset>.jsonl   one row per sample, keyed by uid
    <task>/<dataset>/<figure>.png          rendered five-panel figures
    <task>/manifest.jsonl                  per-sample status and metric per model
    <task>/failures.jsonl                  samples whose inference or rendering failed
    logs/<task>__<model>__<dataset>__gpu<N>.log, runtime_info.json

How it runs
    The parent process builds one job per (task, model, dataset), polls nvidia-smi
    for the GPUs in --gpu-pool and spawns this same file as a worker subprocess
    (`--worker infer`, see worker_main) per job, one per free GPU. Workers pick a
    batch size from the free memory (choose_batch_size), halve it on OOM and
    append to the cache after every batch, so an interrupted run resumes with
    --resume. GeoGround is probed first and skipped when its runtime is missing.
    Rendering (render_task) only reads caches and never runs inference.

Usage, from the repository root
    python visualization/compare_testsets.py --task obb --models all
    python visualization/compare_testsets.py --task hbb --datasets rsvg_test --render-only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Constants and default paths
# ---------------------------------------------------------------------------

# Qwen-VL preprocessing settings, identical to the training and evaluation launchers.
# They must be in the environment before ms-swift is imported inside a worker.
os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '1024')
os.environ.setdefault('QWENVL_BBOX_FORMAT', 'new')

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_DIR / 'output' / 'visualizations' / 'visual_compare_all_testsets'
IMAGE_BASE_DIR = REPO_DIR / 'data' / 'refGeo' / 'images'
METAINFO_DIR = REPO_DIR / 'data' / 'refGeo' / 'metainfo'
OBB_SELECTED_DIR = METAINFO_DIR / 'OBB_Selected'

# Conda environments used to spawn workers: the active one for the Qwen-family models
# (the README's geobox-r1), and GeoGround's own LLaVA-1.5 environment. When the latter
# does not exist GeoGround falls back to the Qwen environment, where the runtime check
# disables it.
DEFAULT_QWEN_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'geobox-r1')
CONDA_ENVS_DIR = Path(os.environ.get('CONDA_ENVS_DIR', Path.home() / 'anaconda3' / 'envs'))
DEFAULT_GEOGROUND_ENV = 'geoground' if (CONDA_ENVS_DIR / 'geoground').exists() else DEFAULT_QWEN_ENV
DEFAULT_CONDA_BIN = os.environ.get('CONDA_EXE', 'conda')
DEFAULT_BASE_MODEL = REPO_DIR / 'models' / 'pretrained' / 'Qwen3-VL-4B-Instruct'
DEFAULT_SFT_MODEL = REPO_DIR / 'models' / 'checkpoints' / 'GeoBox-R1-SFT'
DEFAULT_GDPO_ROOT = REPO_DIR / 'models' / 'checkpoints' / 'GeoBox-R1'
DEFAULT_GEOGROUND_MODEL = (
    REPO_DIR / 'models' / 'pretrained' / 'llava-v1.5-7b-task-geoground'
    if (REPO_DIR / 'models' / 'pretrained' / 'llava-v1.5-7b-task-geoground').exists()
    else REPO_DIR / 'models' / 'pretrained' / 'llava-v1.5-7b-task-lora-geoground'
)

# Panel colours (RGB) and fonts.
GT_COLOR = (30, 200, 80)
QWEN_COLOR = (235, 80, 80)
GEOGROUND_COLOR = (180, 80, 255)
SFT_COLOR = (255, 170, 40)
GDPO_COLOR = (50, 150, 255)
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

# Model keys in panel order (GT comes first). Cache directories and --models use them.
MODEL_ORDER = ['qwen', 'geoground', 'sft', 'gdpo']
MODEL_LABELS = {
    'qwen': 'QWEN3-VL-4B',
    'geoground': 'GEOGROUND',
    'sft': 'SFT',
    'gdpo': 'SFT+GDPO',
}
MODEL_COLORS = {
    'qwen': QWEN_COLOR,
    'geoground': GEOGROUND_COLOR,
    'sft': SFT_COLOR,
    'gdpo': GDPO_COLOR,
}


# ---------------------------------------------------------------------------
# Logging and record types
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str) -> None:
    print(f'[{_ts()}] {msg}', flush=True)


@dataclass(frozen=True)
class DatasetSpec:
    """One evaluation split: metainfo JSONL, image subdirectory and default batch size."""

    name: str
    task: str
    metainfo_file: Path
    image_subdir: str
    pretty_name: str
    default_batch_size: int


@dataclass(frozen=True)
class GPUState:
    """Snapshot of one GPU from nvidia-smi (memory in MB, utilization in percent)."""

    index: int
    name: str
    mem_total_mb: int
    mem_used_mb: int
    mem_free_mb: int
    util: int

    @property
    def free_gb(self) -> float:
        return self.mem_free_mb / 1024.0


@dataclass(frozen=True)
class Job:
    """One inference unit (task, model, dataset); each job becomes one worker subprocess."""

    task: str
    model: str
    dataset: str

    def key(self) -> str:
        return f'{self.task}:{self.model}:{self.dataset}'


# ---------------------------------------------------------------------------
# Evaluation splits
# ---------------------------------------------------------------------------

# HBB uses the seven refGeo test/val splits. OBB prefers the three filtered splits under
# metainfo/OBB_Selected/ and falls back to the full *_test.jsonl (resolve_metainfo_file).
# The last field is the batch size on a free GPU.
HBB_DATASETS: Dict[str, DatasetSpec] = {
    'dior_rsvg_test': DatasetSpec('dior_rsvg_test', 'hbb', METAINFO_DIR / 'dior_rsvg_test.jsonl', 'DIOR-RSVG', 'DIOR-RSVG-Test', 80),
    'dior_rsvg_val': DatasetSpec('dior_rsvg_val', 'hbb', METAINFO_DIR / 'dior_rsvg_val.jsonl', 'DIOR-RSVG', 'DIOR-RSVG-Val', 80),
    'rsvg_test': DatasetSpec('rsvg_test', 'hbb', METAINFO_DIR / 'rsvg_test.jsonl', 'RSVG', 'RSVG-Test', 60),
    'rsvg_val': DatasetSpec('rsvg_val', 'hbb', METAINFO_DIR / 'rsvg_val.jsonl', 'RSVG', 'RSVG-Val', 60),
    'geochat_test': DatasetSpec('geochat_test', 'hbb', METAINFO_DIR / 'geochat_test.jsonl', 'GeoChat', 'GeoChat-Test', 60),
    'vrsbench_test': DatasetSpec('vrsbench_test', 'hbb', METAINFO_DIR / 'vrsbench_test.jsonl', 'VRSBench', 'VRSBench-Test', 96),
    'avvg_test': DatasetSpec('avvg_test', 'hbb', METAINFO_DIR / 'avvg_test.jsonl', 'AVVG', 'AVVG-Test', 12),
}

OBB_DATASETS: Dict[str, DatasetSpec] = {
    'geochat_test_filtered': DatasetSpec('geochat_test_filtered', 'obb', OBB_SELECTED_DIR / 'geochat_test_filtered.jsonl', 'GeoChat', 'GeoChat-Test-Filtered', 60),
    'vrsbench_test_filtered': DatasetSpec('vrsbench_test_filtered', 'obb', OBB_SELECTED_DIR / 'vrsbench_test_filtered.jsonl', 'VRSBench', 'VRSBench-Test-Filtered', 96),
    'avvg_test_filtered': DatasetSpec('avvg_test_filtered', 'obb', OBB_SELECTED_DIR / 'avvg_test_filtered.jsonl', 'AVVG', 'AVVG-Test-Filtered', 12),
}

TASK_DATASETS = {'hbb': HBB_DATASETS, 'obb': OBB_DATASETS}


# ---------------------------------------------------------------------------
# Filesystem and JSONL helpers
# ---------------------------------------------------------------------------

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def strip_code_fence(text: str) -> str:
    return text.replace('```json', '').replace('```', '').strip()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open('a', encoding='utf-8') as f:
        for row in rows:
            f.write(json_dumps(row) + '\n')


def load_jsonl_multiline(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL, tolerating records that were pretty-printed across several lines."""
    data: List[Dict[str, Any]] = []
    buffer = ''
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            buffer = f'{buffer} {line}'.strip() if buffer else line
            try:
                data.append(json.loads(buffer))
                buffer = ''
            except json.JSONDecodeError:
                continue
    if buffer:
        data.append(json.loads(buffer))
    return data


def shorten_error(text: Optional[str], limit: int = 80) -> str:
    if not text:
        return ''
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + '...'


def sample_uid(question_id: Any, image_id: str, question: str) -> str:
    """Stable per-sample key: the question_id when present, else a hash of image and question.

    Cache rows, manifests and export_sample.py all look samples up by this key.
    """
    if question_id is not None:
        return f'qid{question_id}'
    digest = hashlib.md5(f'{image_id}::{question}'.encode('utf-8')).hexdigest()[:10]
    return f'h{digest}'


def resolve_repo_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_DIR))
    except Exception:
        return str(path)


def resolve_exact_image_path(image_dir: Path, image_id: str) -> Optional[Path]:
    """Locate an image, trying the other common extensions when the metainfo suffix is wrong."""
    direct = image_dir / image_id
    if direct.exists():
        return direct
    stem = Path(image_id).stem
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        candidate = image_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None


def build_python_cmd(conda_env: str) -> List[str]:
    """Command prefix that runs python inside conda_env (the current interpreter when already active)."""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env == conda_env:
        return [sys.executable, '-u']
    conda_bin = DEFAULT_CONDA_BIN if Path(DEFAULT_CONDA_BIN).exists() else 'conda'
    return [conda_bin, 'run', '--no-capture-output', '-n', conda_env, 'python', '-u']


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding='utf-8')


# ---------------------------------------------------------------------------
# Dataset resolution and sample loading
# ---------------------------------------------------------------------------
def dataset_specs_for_task(task: str) -> Dict[str, DatasetSpec]:
    if task not in TASK_DATASETS:
        raise KeyError(f'Unknown task: {task}')
    return TASK_DATASETS[task]


def resolve_task_list(task: str) -> List[str]:
    if task == 'all':
        return ['obb', 'hbb']
    return [task]


def resolve_datasets(task: str, dataset_args: List[str]) -> List[DatasetSpec]:
    specs = dataset_specs_for_task(task)
    if not dataset_args or dataset_args == ['all']:
        return list(specs.values())
    resolved = []
    for name in dataset_args:
        if name not in specs:
            raise KeyError(f'Unknown dataset for task {task}: {name}')
        resolved.append(specs[name])
    return resolved


def resolve_custom_dataset(task: str, args: argparse.Namespace) -> Optional[DatasetSpec]:
    """Ad-hoc split given with --custom-metainfo-file; it replaces the built-in split list."""
    custom_metainfo_file = getattr(args, 'custom_metainfo_file', None)
    if not custom_metainfo_file:
        return None
    dataset_name = (args.custom_dataset_name or custom_metainfo_file.stem).strip()
    pretty_name = (args.custom_pretty_name or dataset_name).strip()
    image_subdir = (args.custom_image_subdir or '').strip()
    if not dataset_name:
        raise ValueError('Custom dataset name cannot be empty')
    if not image_subdir:
        raise ValueError('--custom-image-subdir is required with --custom-metainfo-file')
    return DatasetSpec(
        name=dataset_name,
        task=task,
        metainfo_file=custom_metainfo_file,
        image_subdir=image_subdir,
        pretty_name=pretty_name,
        default_batch_size=max(1, int(args.custom_default_batch_size)),
    )


def resolve_datasets_with_args(task: str, dataset_args: List[str], args: argparse.Namespace) -> List[DatasetSpec]:
    custom_spec = resolve_custom_dataset(task, args)
    if custom_spec is not None:
        return [custom_spec]
    return resolve_datasets(task, dataset_args)


def resolve_dataset_spec_for_name(task: str, dataset_name: str, args: argparse.Namespace) -> DatasetSpec:
    custom_spec = resolve_custom_dataset(task, args)
    if custom_spec is not None:
        if dataset_name != custom_spec.name:
            raise KeyError(f'Custom dataset is {custom_spec.name}, got {dataset_name}')
        return custom_spec
    specs = dataset_specs_for_task(task)
    if dataset_name not in specs:
        raise KeyError(f'Unknown dataset for task {task}: {dataset_name}')
    return specs[dataset_name]


def resolve_metainfo_file(spec: DatasetSpec) -> Path:
    """Metainfo file to read for a split, with the OBB fallback.

    The filtered OBB splits under metainfo/OBB_Selected/ are preferred. When such a
    file is missing (the documented data layout ships only the full splits), the full
    split metainfo/<name without _filtered>.jsonl is used and a notice is logged.
    """
    if spec.metainfo_file.exists() or spec.task != 'obb' or not spec.name.endswith('_filtered'):
        return spec.metainfo_file
    fallback = METAINFO_DIR / f"{spec.name[: -len('_filtered')]}.jsonl"
    if fallback.exists():
        log(f'{spec.name}: {resolve_repo_relpath(spec.metainfo_file)} not found, '
            f'using the full split {resolve_repo_relpath(fallback)}')
        return fallback
    return spec.metainfo_file


def load_samples(spec: DatasetSpec, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read one split into sample dicts (uid, image path, question, GT bbox and polygon)."""
    rows = load_jsonl_multiline(resolve_metainfo_file(spec))
    image_dir = IMAGE_BASE_DIR / spec.image_subdir
    samples: List[Dict[str, Any]] = []
    for item in rows:
        image_id = item['image_id']
        image_path = resolve_exact_image_path(image_dir, image_id)
        uid = sample_uid(item.get('question_id'), image_id, item.get('question', ''))
        gt_bbox = item.get('bbox')
        gt_poly = item.get('poly')
        samples.append({
            'dataset': spec.name,
            'task': spec.task,
            'uid': uid,
            'question_id': item.get('question_id'),
            'image_id': image_id,
            'image_path': str(image_path) if image_path else '',
            'image_relpath': resolve_repo_relpath(image_path) if image_path else '',
            'question': item.get('question', ''),
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
        })
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


# ---------------------------------------------------------------------------
# Output layout and result cache
# ---------------------------------------------------------------------------
# <output_root>/<task>/cache/<model>/<dataset>.jsonl   cached predictions
# <output_root>/<task>/<dataset>/                      rendered figures
# <output_root>/<task>/manifest.jsonl, failures.jsonl  per-sample bookkeeping
def task_output_dir(output_root: Path, task: str) -> Path:
    return output_root / task


def cache_file(output_root: Path, task: str, model: str, dataset: str) -> Path:
    return task_output_dir(output_root, task) / 'cache' / model / f'{dataset}.jsonl'


def render_dir(output_root: Path, task: str, dataset: str) -> Path:
    return task_output_dir(output_root, task) / dataset


def logs_dir(output_root: Path) -> Path:
    return output_root / 'logs'


def runtime_info_file(output_root: Path) -> Path:
    return output_root / 'runtime_info.json'


def manifest_file(output_root: Path, task: str) -> Path:
    return task_output_dir(output_root, task) / 'manifest.jsonl'


def failures_file(output_root: Path, task: str) -> Path:
    return task_output_dir(output_root, task) / 'failures.jsonl'


def load_cache_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a cache file as uid -> row (the last row wins for duplicate uids)."""
    if not path.exists():
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            mapping[item['uid']] = item
    return mapping


# ---------------------------------------------------------------------------
# Geometry (all boxes are in original-image pixels after decoding)
# ---------------------------------------------------------------------------
def maybe_import_shapely_polygon():
    from shapely.geometry import Polygon  # type: ignore

    return Polygon


def calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Axis-aligned IoU of two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def normalize_obb_points(poly: Sequence[Sequence[float]]) -> Optional[List[List[float]]]:
    """Validate a four-corner polygon and return it as floats, or None when malformed."""
    if not isinstance(poly, (list, tuple)) or len(poly) != 4:
        return None

    points: List[List[float]] = []
    for point in poly:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append([x, y])
    return points


def _alignment_cost(
    pred_points: Sequence[Sequence[float]],
    gt_points: Sequence[Sequence[float]],
    perm: Sequence[int],
) -> float:
    cost = 0.0
    for pred_idx, gt_idx in enumerate(perm):
        dx = float(pred_points[pred_idx][0]) - float(gt_points[gt_idx][0])
        dy = float(pred_points[pred_idx][1]) - float(gt_points[gt_idx][1])
        cost += dx * dx + dy * dy
    return cost


def follows_annotation_order(
    pred_points: Sequence[Sequence[float]],
    gt_points: Sequence[Sequence[float]],
) -> bool:
    """True when predicted corner i is the nearest match of GT corner i for every i.

    All 24 vertex permutations are scored by summed squared distance and only the
    identity permutation passes. calculate_rotated_iou uses this as a hard gate, so
    a geometrically correct polygon whose corners are listed in another order gets
    0 here. evaluation/evaluate_obb.py instead reorders corners around the centroid
    and has no such gate, so its RIoU can be higher for the same prediction.
    """
    pred = normalize_obb_points(pred_points)
    gt = normalize_obb_points(gt_points)
    if pred is None or gt is None:
        return False

    best_perm = None
    best_cost = None
    for perm in permutations(range(4)):
        perm_tuple = tuple(perm)
        cost = _alignment_cost(pred, gt, perm_tuple)
        if best_cost is None or cost < best_cost:
            best_perm = perm_tuple
            best_cost = cost
    return best_perm == (0, 1, 2, 3)


def calculate_rotated_iou(poly1: Sequence[Sequence[float]], poly2: Sequence[Sequence[float]]) -> float:
    """Figure-only rotated IoU: polygon IoU via shapely, gated by follows_annotation_order.

    A prediction whose corner order differs from the GT scores 0 here. The metric
    scripts under evaluation/ do not apply this gate, so the RIoU printed on a figure
    can be lower than the reported metric for the same prediction. 0.0 on any failure.
    """
    try:
        Polygon = maybe_import_shapely_polygon()
        if not follows_annotation_order(poly1, poly2):
            return 0.0
        p1 = Polygon([(float(x), float(y)) for x, y in poly1])
        p2 = Polygon([(float(x), float(y)) for x, y in poly2])
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        union = p1.union(p2).area
        if union <= 0:
            return 0.0
        return p1.intersection(p2).area / union
    except Exception:
        return 0.0


def poly_to_bbox(poly: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(pt[0]) for pt in poly]
    ys = [float(pt[1]) for pt in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def hbb_to_poly(bbox: Sequence[float]) -> List[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def union_bbox(*boxes: Sequence[float]) -> List[float]:
    xs1 = [float(b[0]) for b in boxes]
    ys1 = [float(b[1]) for b in boxes]
    xs2 = [float(b[2]) for b in boxes]
    ys2 = [float(b[3]) for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def compute_crop(image_size: Tuple[int, int], focus_box: Sequence[float]) -> Tuple[int, int, int, int]:
    """Crop window (pixels) around focus_box, padded by multiples of the box size.

    The padding is asymmetric (more context on the left and top) and never leaves
    the image. Shared by export_sample.py so both tools zoom the same way.
    """
    img_w, img_h = image_size
    x1, y1, x2, y2 = [float(v) for v in focus_box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cl = max(0, int(round(cx - max(bw * 2.2, 140))))
    ct = max(0, int(round(cy - max(bh * 2.0, 120))))
    cr = min(img_w, int(round(cx + max(bw * 1.2, 120))))
    cb = min(img_h, int(round(cy + max(bh * 1.5, 90))))
    return cl, ct, cr, cb


def shift_bbox(bbox: Sequence[float], dx: float, dy: float) -> List[float]:
    return [float(bbox[0]) - dx, float(bbox[1]) - dy, float(bbox[2]) - dx, float(bbox[3]) - dy]


def shift_poly(poly: Sequence[Sequence[float]], dx: float, dy: float) -> List[List[float]]:
    return [[float(pt[0]) - dx, float(pt[1]) - dy] for pt in poly]


def obb5_to_corners_le90(cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[List[float]]:
    """Corners of a (cx, cy, w, h, angle) box in the long-edge-90 convention.

    When h > w the sides are swapped and the angle rotated by 90 degrees so that
    the angle always refers to the long side; the angle is then taken mod 180.
    """
    if h > w:
        w, h = h, w
        angle_deg += 90.0
    a = math.radians(angle_deg % 180.0)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = w / 2.0, h / 2.0
    corners: List[List[float]] = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        dx = sx * hw * cos_a - sy * hh * sin_a
        dy = sx * hw * sin_a + sy * hh * cos_a
        corners.append([cx + dx, cy + dy])
    return corners


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# The Qwen-family prompts are the training/evaluation prompts verbatim; the GeoGround
# prompts are its refer instruction plus an explicit output-format request.
def build_obb_prompt(question: str, include_image_token: bool = False) -> str:
    prefix = "<image>" if include_image_token else ""
    return (
        f"{prefix}Locate the instance that matches the description: [{question}]. "
        "Report oriented bbox coordinates in following JSON format:\n"
        "```json\n"
        "[\n"
        '\t{"oriented_bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}\n'
        "]\n"
        "```"
    )


def get_hbb_prompt(question: str) -> str:
    return f"""Locate the instance that matches the description: [{question}]. Report horizontal bbox coordinates in following JSON format:
```json
[
\t{{\"horizontal_bbox\": [x1, y1, x2, y2]}}
]
```"""


def get_obb_prompt(question: str) -> str:
    return build_obb_prompt(question)


def get_geoground_hbb_prompt(question: str) -> str:
    """GeoGround HBB refer prompt with a format suffix.

    Figure-only convention: the metric scripts under baselines/evaluate use the bare
    '[refer] ... in the image.' prompt, so the reported GeoGround numbers were produced
    without this suffix. Here it additionally asks for exactly one <box>[[x1,y1,x2,y2]]</box>.
    """
    return (
        '[refer] output the bounding box of the <ref>' + question + '</ref> in the image. '
        'Output only one horizontal bounding box in exactly this format: '
        '<box>[[x1,y1,x2,y2]]</box>. '
        'Use integer coordinates normalized to 0-1000. '
        'Do not output any extra words, explanation, or additional boxes.'
    )


def get_geoground_obb_prompt(question: str) -> str:
    """GeoGround OBB refer prompt with a format suffix asking for one Text-OBB tuple.

    Figure-only convention: the metric scripts under baselines/evaluate use the bare
    refer prompt; the <obb>[[cx,cy,w,h,angle]]</obb> instruction here is not part of the
    reported GeoGround numbers.
    """
    return (
        '[refer] output the oriented bounding box of the <ref>' + question + '</ref> in the image. '
        'Output only one oriented bounding box in exactly this format: '
        '<obb>[[cx,cy,w,h,angle]]</obb>. '
        'Use integer values. Use Text-OBB resolution 100 for cx, cy, w, h. '
        'Use long-side 90-degree representation and keep angle in [0,90]. '
        'Do not output polygon points, extra numbers, explanation, or additional boxes.'
    )


# ---------------------------------------------------------------------------
# Decoding Qwen-style JSON answers
# ---------------------------------------------------------------------------

def maybe_denormalize_bbox(bbox: Sequence[float], image_size: Tuple[int, int], norm_bbox: Optional[str]) -> List[float]:
    """norm1000 -> original-image pixels, unless the template reports norm_bbox == 'none' (absolute)."""
    if not bbox or len(bbox) < 4:
        return list(bbox)
    width, height = image_size
    bbox = [float(v) for v in bbox[:4]]
    if norm_bbox == 'none':
        return bbox
    return [bbox[0] / 1000.0 * width, bbox[1] / 1000.0 * height, bbox[2] / 1000.0 * width, bbox[3] / 1000.0 * height]


def maybe_denormalize_obb(poly: Sequence[Sequence[float]], image_size: Tuple[int, int], norm_bbox: Optional[str]) -> List[List[float]]:
    """Polygon counterpart of maybe_denormalize_bbox."""
    width, height = image_size
    if norm_bbox == 'none':
        return [[float(x), float(y)] for x, y in poly]
    return [[float(x) / 1000.0 * width, float(y) / 1000.0 * height] for x, y in poly]


def parse_horizontal_bbox_json(text: str) -> Optional[List[float]]:
    """HBB from the JSON answer (regex first, json.loads second); a flat four-number oriented_bbox is accepted too."""
    cleaned = strip_code_fence(text)
    patterns = [
        r'"horizontal_bbox"\s*:\s*\[([^\]]+)\]',
        r'"oriented_bbox"\s*:\s*\[([0-9\.\-,\s]+)\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        try:
            values = [float(x.strip()) for x in match.group(1).split(',') if x.strip()]
        except ValueError:
            continue
        if len(values) == 4:
            return values
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            item = data[0]
            for key in ('horizontal_bbox', 'oriented_bbox'):
                bbox = item.get(key)
                if isinstance(bbox, list) and len(bbox) == 4 and not isinstance(bbox[0], list):
                    return [float(v) for v in bbox]
    except Exception:
        pass
    return None


def parse_oriented_polygon_json(text: str) -> Optional[List[List[float]]]:
    """Four-corner polygon from the JSON answer, or None."""
    cleaned = strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            item = data[0]
            poly = item.get('oriented_bbox')
            if isinstance(poly, list) and len(poly) == 4 and isinstance(poly[0], list):
                return [[float(p[0]), float(p[1])] for p in poly]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Decoding GeoGround text answers (<box>/<obb> tags with heuristic repair)
# ---------------------------------------------------------------------------
# GeoGround often emits extra prose, repeated boxes or over-long digit strings.
# Every 4- or 5-number window is scored for plausibility and the best one is kept.
def _extract_numeric_values(text: str) -> List[float]:
    return [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', text)]


def _tag_payload(text: str, tag: str) -> str:
    """Text inside <tag>...</tag>, or the whole fence-stripped text when the tag is absent."""
    match = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', strip_code_fence(text), re.S)
    return match.group(1) if match else strip_code_fence(text)


def _digits_only(value: float) -> str:
    try:
        return re.sub(r'\D', '', str(abs(int(round(float(value))))))
    except Exception:
        return ''


def _clip_to_resolution(value: float, resolution: int) -> float:
    """Repair an out-of-range integer by keeping its leading digits (12345 on a 0-1000 grid -> 123)."""
    value = float(value)
    sign = -1.0 if value < 0 else 1.0
    abs_value = abs(value)
    if abs_value <= resolution:
        return value

    digits = _digits_only(value)
    if not digits:
        return value

    if resolution == 1000:
        keep = 4 if digits.startswith('1000') else 3
    elif resolution == 100:
        keep = 3 if digits.startswith('100') else 2
    else:
        keep = len(str(resolution))

    clipped = float(digits[:keep])
    if clipped > resolution:
        clipped = float(resolution)
    return sign * clipped


def _normalize_hbb_values(values: Sequence[float]) -> List[float]:
    return [_clip_to_resolution(v, 1000) for v in values[:4]]


def _normalize_obb_spatial(values: Sequence[float]) -> List[float]:
    return [max(0.0, _clip_to_resolution(v, 100)) for v in values[:4]]


def _angle_variants(token: float) -> List[float]:
    """Candidate angles for one token: itself when within [0, 90], its value mod 90, and 90 minus that."""
    abs_token = abs(float(token))
    mod = abs_token % 90.0
    candidates = []
    for candidate in (abs_token if abs_token <= 90.0 else None, mod, 90.0 - mod if mod != 0.0 else 0.0):
        if candidate is None:
            continue
        candidate = float(candidate)
        if 0.0 <= candidate <= 90.0 and all(abs(candidate - prev) > 1e-6 for prev in candidates):
            candidates.append(candidate)
    return candidates or [0.0]


def _score_hbb_candidate(values: Sequence[float], window_index: int, transformed: bool) -> float:
    """Plausibility of a 4-number window: in-range values, positive area, not near
    full-image size; earlier windows and unrepaired values are preferred."""
    x1, y1, x2, y2 = [float(v) for v in values[:4]]
    score = 0.0
    in_range = sum(0.0 <= v <= 1000.0 for v in (x1, y1, x2, y2))
    score += in_range * 20.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width > 0.0 and height > 0.0:
        score += 18.0
    if 1.0 <= width <= 1000.0:
        score += 6.0
    if 1.0 <= height <= 1000.0:
        score += 6.0
    if width < 1.0 or height < 1.0:
        score -= 18.0
    if width > 980.0 or height > 980.0:
        score -= 8.0
    if transformed:
        score -= 1.5
    score -= window_index * 0.35
    return score


def _best_hbb_candidate(values: Sequence[float]) -> Optional[List[float]]:
    """Scan every 4-number window of the tokens, raw and digit-repaired, and return the best-scoring one."""
    if len(values) < 4:
        return None

    candidates: List[Tuple[float, List[float]]] = []
    for i in range(len(values) - 3):
        raw = [float(v) for v in values[i : i + 4]]
        norm = _normalize_hbb_values(raw)
        candidates.append((_score_hbb_candidate(raw, i, transformed=False), raw))
        candidates.append((_score_hbb_candidate(norm, i, transformed=True), norm))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _score_obb5_candidate(values: Sequence[float], priority: int, clipped: bool, angle_from_tail: bool) -> float:
    """Plausibility of a (cx, cy, w, h, angle) tuple on the 0-100 Text-OBB grid."""
    cx, cy, bw, bh, angle = [float(v) for v in values[:5]]
    score = 0.0
    if 0.0 <= cx <= 100.0:
        score += 25.0
    if 0.0 <= cy <= 100.0:
        score += 25.0
    if 0.0 < bw <= 100.0:
        score += 18.0
    if 0.0 < bh <= 100.0:
        score += 18.0
    if 0.0 <= angle <= 90.0:
        score += 8.0
    if bw < 1.0 or bh < 1.0:
        score -= 16.0
    if max(bw, bh) > 95.0:
        score -= 6.0
    if clipped:
        score -= 1.0
    if angle_from_tail:
        score += 2.0
    score -= priority * 0.4
    return score


def _best_obb5_candidate(values: Sequence[float]) -> Optional[List[float]]:
    """Recover the most plausible (cx, cy, w, h, angle) tuple from noisy tokens.

    The first four numbers are tried raw and clipped to the 0-100 grid; the angle is
    tried from several tail tokens (negative numbers first, then the last and the
    first tail token), each expanded through _angle_variants, and every combination
    is scored with _score_obb5_candidate.
    """
    if len(values) < 5:
        return None

    first4_raw = [float(v) for v in values[:4]]
    first4_norm = _normalize_obb_spatial(first4_raw)
    angle_tokens: List[Tuple[int, float, bool]] = []
    tail = [float(v) for v in values[4:]]
    negatives = [v for v in tail if v < 0]
    if negatives:
        angle_tokens.append((0, negatives[-1], True))
        angle_tokens.append((1, negatives[0], True))
    if tail:
        angle_tokens.append((2, tail[-1], True))
        angle_tokens.append((3, tail[0], False))
    else:
        angle_tokens.append((4, values[4], False))

    candidates: List[Tuple[float, List[float]]] = []
    for base_priority, base_values, clipped in ((0, first4_raw, False), (1, first4_norm, True)):
        for angle_priority, token, from_tail in angle_tokens:
            for angle in _angle_variants(token):
                candidate = list(base_values) + [float(angle)]
                score = _score_obb5_candidate(candidate, base_priority + angle_priority, clipped=clipped, angle_from_tail=from_tail)
                candidates.append((score, candidate))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def _clip_bbox_to_image(bbox: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return [x1, y1, x2, y2]


def _clip_poly_to_image(poly: Sequence[Sequence[float]], image_size: Tuple[int, int]) -> List[List[float]]:
    width, height = image_size
    return [[min(max(float(x), 0.0), float(width)), min(max(float(y), 0.0), float(height))] for x, y in poly]


def parse_geoground_hbb(text: str, image_size: Tuple[int, int]) -> Optional[List[float]]:
    """Decode <box>...</box> into pixel [x1, y1, x2, y2]; values within [0, 1] are read as
    fractions, anything else as norm1000; the box is clipped to the image."""
    values = _extract_numeric_values(_tag_payload(text, 'box'))
    candidate = _best_hbb_candidate(values)
    if candidate is None:
        return None

    width, height = image_size
    max_abs = max(abs(v) for v in candidate)
    if max_abs <= 1.0:
        scaled = [candidate[0] * width, candidate[1] * height, candidate[2] * width, candidate[3] * height]
    else:
        scaled = [candidate[0] / 1000.0 * width, candidate[1] / 1000.0 * height, candidate[2] / 1000.0 * width, candidate[3] / 1000.0 * height]
    x1, y1, x2, y2 = scaled
    return _clip_bbox_to_image([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], image_size)


def parse_geoground_obb(text: str, image_size: Tuple[int, int]) -> Optional[List[List[float]]]:
    """Decode <obb>...</obb> into four pixel corners.

    Five numbers are a Text-OBB tuple (cx, cy, w, h on the 0-100 grid, angle in
    degrees, long-edge-90 convention); eight numbers are taken as corners on a 0-1,
    0-100 or 0-1000 grid depending on their magnitude.
    """
    values = _extract_numeric_values(_tag_payload(text, 'obb'))
    candidate5 = _best_obb5_candidate(values)
    if candidate5 is not None:
        width, height = image_size
        cx, cy, bw, bh, angle = candidate5
        max_abs = max(abs(cx), abs(cy), abs(bw), abs(bh))
        if max_abs <= 1.0:
            cx, cy = cx * width, cy * height
            bw, bh = bw * width, bh * height
        else:
            cx, cy = cx / 100.0 * width, cy / 100.0 * height
            bw, bh = bw / 100.0 * width, bh / 100.0 * height
        return _clip_poly_to_image(obb5_to_corners_le90(cx, cy, bw, bh, angle), image_size)
    if len(values) >= 8:
        coords = values[:8]
        pts = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        max_abs = max(abs(v) for v in coords)
        width, height = image_size
        if max_abs <= 1.0:
            pts = [[x * width, y * height] for x, y in pts]
        elif max_abs <= 100.0:
            pts = [[x / 100.0 * width, y / 100.0 * height] for x, y in pts]
        elif max_abs <= 1000.0:
            pts = [[x / 1000.0 * width, y / 1000.0 * height] for x, y in pts]
        return _clip_poly_to_image(pts, image_size)
    return None


# ---------------------------------------------------------------------------
# Per-model result records (the rows written to the cache)
# ---------------------------------------------------------------------------

def parse_qwen_result(task: str, raw_output: str, image_size: Tuple[int, int], norm_bbox: Optional[str], gt_bbox: Optional[Sequence[float]], gt_poly: Optional[Sequence[Sequence[float]]]) -> Dict[str, Any]:
    """Turn one Qwen-family answer into a cache row.

    Sets status and pred_kind (hbb / obb / none), the pixel boxes and the metric
    against the GT. In OBB mode an answer that only contains an HBB is kept as
    pred_kind='hbb' with an IoU against gt_bbox, so the figure shows the format
    fallback instead of an empty panel; in HBB mode an OBB answer is reduced to
    its bounding box.
    """
    result: Dict[str, Any] = {
        'status': 'parse_failed',
        'pred_kind': 'none',
        'pred_hbb': None,
        'pred_obb': None,
        'metric_name': None,
        'metric': None,
        'error': None,
        'raw_output': raw_output,
    }
    if task == 'obb':
        pred_obb_norm = parse_oriented_polygon_json(raw_output)
        if pred_obb_norm is not None:
            pred_obb = maybe_denormalize_obb(pred_obb_norm, image_size, norm_bbox)
            result.update({'status': 'ok', 'pred_kind': 'obb', 'pred_obb': pred_obb})
            if gt_poly is not None:
                result['metric_name'] = 'riou'
                result['metric'] = calculate_rotated_iou(pred_obb, gt_poly)
            return result
        pred_hbb_norm = parse_horizontal_bbox_json(raw_output)
        if pred_hbb_norm is not None:
            pred_hbb = maybe_denormalize_bbox(pred_hbb_norm, image_size, norm_bbox)
            result.update({'status': 'ok', 'pred_kind': 'hbb', 'pred_hbb': pred_hbb})
            if gt_bbox is not None:
                result['metric_name'] = 'iou'
                result['metric'] = calculate_iou(pred_hbb, gt_bbox)
            return result
        result['error'] = 'No valid box found'
        return result

    pred_hbb_norm = parse_horizontal_bbox_json(raw_output)
    if pred_hbb_norm is not None:
        pred_hbb = maybe_denormalize_bbox(pred_hbb_norm, image_size, norm_bbox)
        result.update({'status': 'ok', 'pred_kind': 'hbb', 'pred_hbb': pred_hbb})
        if gt_bbox is not None:
            result['metric_name'] = 'iou'
            result['metric'] = calculate_iou(pred_hbb, gt_bbox)
        return result

    pred_obb_norm = parse_oriented_polygon_json(raw_output)
    if pred_obb_norm is not None:
        pred_obb = maybe_denormalize_obb(pred_obb_norm, image_size, norm_bbox)
        result.update({'status': 'ok', 'pred_kind': 'obb', 'pred_obb': pred_obb, 'pred_hbb': poly_to_bbox(pred_obb)})
        if gt_bbox is not None:
            result['metric_name'] = 'iou'
            result['metric'] = calculate_iou(result['pred_hbb'], gt_bbox)
        return result

    result['error'] = 'No valid box found'
    return result


def parse_geoground_result(task: str, raw_output: str, image_size: Tuple[int, int], gt_bbox: Optional[Sequence[float]], gt_poly: Optional[Sequence[Sequence[float]]]) -> Dict[str, Any]:
    """GeoGround counterpart of parse_qwen_result, using the <box>/<obb> decoders."""
    result: Dict[str, Any] = {
        'status': 'parse_failed',
        'pred_kind': 'none',
        'pred_hbb': None,
        'pred_obb': None,
        'metric_name': None,
        'metric': None,
        'error': None,
        'raw_output': raw_output,
    }
    if task == 'obb':
        pred_obb = parse_geoground_obb(raw_output, image_size)
        if pred_obb is not None:
            result.update({'status': 'ok', 'pred_kind': 'obb', 'pred_obb': pred_obb})
            if gt_poly is not None:
                result['metric_name'] = 'riou'
                result['metric'] = calculate_rotated_iou(pred_obb, gt_poly)
            return result
        pred_hbb = parse_geoground_hbb(raw_output, image_size)
        if pred_hbb is not None:
            result.update({'status': 'ok', 'pred_kind': 'hbb', 'pred_hbb': pred_hbb})
            if gt_bbox is not None:
                result['metric_name'] = 'iou'
                result['metric'] = calculate_iou(pred_hbb, gt_bbox)
            return result
        result['error'] = 'No valid box found after retrying GeoGround Text-OBB format'
        return result

    pred_hbb = parse_geoground_hbb(raw_output, image_size)
    if pred_hbb is not None:
        result.update({'status': 'ok', 'pred_kind': 'hbb', 'pred_hbb': pred_hbb})
        if gt_bbox is not None:
            result['metric_name'] = 'iou'
            result['metric'] = calculate_iou(pred_hbb, gt_bbox)
        return result
    pred_obb = parse_geoground_obb(raw_output, image_size)
    if pred_obb is not None:
        pred_hbb = poly_to_bbox(pred_obb)
        result.update({'status': 'ok', 'pred_kind': 'obb', 'pred_obb': pred_obb, 'pred_hbb': pred_hbb})
        if gt_bbox is not None:
            result['metric_name'] = 'iou'
            result['metric'] = calculate_iou(pred_hbb, gt_bbox)
        return result
    result['error'] = 'No valid box found after retrying GeoGround text formats'
    return result


# ---------------------------------------------------------------------------
# GPU discovery and batch-size policy
# ---------------------------------------------------------------------------
def parse_gpu_pool(pool_text: str) -> List[int]:
    if pool_text == 'auto':
        states = query_gpus(None)
        return [s.index for s in states]
    return [int(x.strip()) for x in pool_text.split(',') if x.strip()]


def query_gpus(pool: Optional[List[int]]) -> List[GPUState]:
    """Snapshot the requested GPUs with nvidia-smi, most free memory first."""
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    states: List[GPUState] = []
    for raw in result.stdout.strip().splitlines():
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) != 6:
            continue
        state = GPUState(
            index=int(parts[0]),
            name=parts[1],
            mem_total_mb=int(parts[2]),
            mem_used_mb=int(parts[3]),
            mem_free_mb=int(parts[4]),
            util=int(parts[5]),
        )
        if pool is None or state.index in pool:
            states.append(state)
    states.sort(key=lambda s: (-s.mem_free_mb, s.util, s.index))
    return states


def choose_batch_size(job: Job, spec: DatasetSpec, gpu: GPUState, args: argparse.Namespace) -> int:
    """Batch size for a job on a GPU; 0 means the GPU is not free enough yet.

    The 'fixed' policy returns the configured sizes; 'auto' scales the split's
    default batch size by free memory. GeoGround has its own thresholds and caps.
    """
    if args.batch_policy == 'fixed':
        if job.model == 'geoground' and args.geoground_fixed_batch_size is not None:
            return args.geoground_fixed_batch_size
        if args.fixed_batch_size is not None:
            return args.fixed_batch_size
    free_gb = gpu.free_gb
    if job.model == 'geoground':
        if free_gb < args.geoground_min_free_gb:
            return 0
        if free_gb >= 32:
            return min(spec.default_batch_size, 8)
        if free_gb >= 28:
            return min(spec.default_batch_size, 4)
        return 1
    if free_gb < args.min_free_gb:
        return 0
    if free_gb >= 30:
        return spec.default_batch_size
    if free_gb >= 24:
        return max(1, spec.default_batch_size // 2)
    if free_gb >= 18:
        return max(1, spec.default_batch_size // 4)
    return 0


# ---------------------------------------------------------------------------
# Drawing and figure rendering
# ---------------------------------------------------------------------------
def load_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def draw_text_box(draw, xy: Tuple[float, float], text: str, color: Tuple[int, int, int], anchor: str = 'lt') -> None:
    """Filled label box with white text at xy; anchor 'rt' / 'lb' shifts it left / up."""
    font = load_font(16)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy
    if anchor == 'rt':
        x -= tw + 10
    if anchor == 'lb':
        y -= th + 10
    pad = 4
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=color)
    draw.text((x + pad, y + pad), text, fill='white', font=font)


def draw_hbb(img, bbox: Sequence[float], color: Tuple[int, int, int], label: Optional[str], width: int = 4) -> None:
    """Outline an HBB and put its label above the top-left corner."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        draw_text_box(draw, (x1, max(4, y1 - 28)), label, color)


def draw_obb(img, poly: Sequence[Sequence[float]], color: Tuple[int, int, int], label: Optional[str], width: int = 4) -> None:
    """Outline an OBB with corner dots and put its label above the topmost corner."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    pts = [(int(round(x)), int(round(y))) for x, y in poly]
    for i in range(4):
        draw.line([pts[i], pts[(i + 1) % 4]], fill=color, width=width)
    for x, y in pts:
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    if label:
        top_pt = min(pts, key=lambda p: p[1])
        draw_text_box(draw, (top_pt[0], max(4, top_pt[1] - 28)), label, color)


def fit_size(src_size: Tuple[int, int], max_w: int, max_h: int) -> Tuple[int, int]:
    w, h = src_size
    scale = min(max_w / w, max_h / h)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def draw_gt(img, task: str, gt_bbox: Optional[Sequence[float]], gt_poly: Optional[Sequence[Sequence[float]]]) -> None:
    if task == 'obb' and gt_poly is not None:
        draw_obb(img, gt_poly, GT_COLOR, 'GT', width=4)
    elif gt_bbox is not None:
        draw_hbb(img, gt_bbox, GT_COLOR, 'GT', width=4)


def draw_pred(img, pred_kind: str, pred_hbb: Optional[Sequence[float]], pred_obb: Optional[Sequence[Sequence[float]]], color: Tuple[int, int, int], label: str) -> None:
    if pred_kind == 'obb' and pred_obb is not None:
        draw_obb(img, pred_obb, color, label, width=4)
    elif pred_kind == 'hbb' and pred_hbb is not None:
        draw_hbb(img, pred_hbb, color, label, width=4)


def make_panel_image(
    task: str,
    full_img,
    crop_box: Sequence[int],
    gt_bbox: Optional[Sequence[float]],
    gt_poly: Optional[Sequence[Sequence[float]]],
    pred_kind: str = 'none',
    pred_hbb: Optional[Sequence[float]] = None,
    pred_obb: Optional[Sequence[Sequence[float]]] = None,
    pred_color: Tuple[int, int, int] = QWEN_COLOR,
    pred_label: Optional[str] = None,
):
    """One panel: the full image with GT (and the prediction), the crop rectangle,
    and a zoom of that crop pasted as an inset in the top-left corner."""
    from PIL import ImageDraw, Image

    canvas = full_img.copy().convert('RGB')
    draw = ImageDraw.Draw(canvas)
    draw_gt(canvas, task, gt_bbox, gt_poly)
    if pred_kind != 'none' and pred_label:
        draw_pred(canvas, pred_kind, pred_hbb, pred_obb, pred_color, pred_label)

    cl, ct, cr, cb = [int(v) for v in crop_box]
    draw.rectangle([cl, ct, cr, cb], outline=(255, 255, 255), width=3)

    zoom = full_img.crop((cl, ct, cr, cb)).convert('RGB')
    if task == 'obb' and gt_poly is not None:
        draw_obb(zoom, shift_poly(gt_poly, cl, ct), GT_COLOR, 'GT', width=4)
    elif gt_bbox is not None:
        draw_hbb(zoom, shift_bbox(gt_bbox, cl, ct), GT_COLOR, 'GT', width=4)

    if pred_kind == 'obb' and pred_obb is not None and pred_label:
        draw_obb(zoom, shift_poly(pred_obb, cl, ct), pred_color, pred_label, width=4)
    elif pred_kind == 'hbb' and pred_hbb is not None and pred_label:
        draw_hbb(zoom, shift_bbox(pred_hbb, cl, ct), pred_color, pred_label, width=4)

    inset_size = fit_size(zoom.size, max_w=270, max_h=220)
    zoom = zoom.resize(inset_size, Image.Resampling.LANCZOS)
    zx, zy = 18, 18
    border = 5
    canvas.paste(zoom, (zx + border, zy + border))
    draw.rectangle([zx, zy, zx + inset_size[0] + border * 2, zy + inset_size[1] + border * 2], outline=(255, 255, 255), width=3)
    return canvas


def panel_note(task: str, result: Optional[Dict[str, Any]]) -> str:
    """Caption for a panel: metric, fallback kind, shortened error text or 'Not run'."""
    if not result:
        return 'Not run'
    status = result.get('status')
    if status == 'ok':
        pred_kind = result.get('pred_kind', 'none')
        metric_name = result.get('metric_name')
        metric = result.get('metric')
        if pred_kind == 'none':
            return 'No prediction parsed'
        if metric is None or metric_name is None:
            return f'{pred_kind.upper()} output'
        if metric_name == 'riou':
            return f'OBB · RIoU={metric:.3f}'
        if task == 'obb' and pred_kind == 'hbb':
            return f'HBB fallback · IoU={metric:.3f}'
        if task == 'hbb' and pred_kind == 'obb':
            return f'OBB output · HBB-IoU={metric:.3f}'
        return f'IoU={metric:.3f}'
    error = shorten_error(result.get('error'))
    if error:
        return error
    return status or 'Failed'


def render_single_figure(
    task: str,
    sample: Dict[str, Any],
    model_results: Dict[str, Optional[Dict[str, Any]]],
    output_path: Path,
    max_render_side: int = 1600,
) -> Dict[str, Any]:
    """Render the GT panel plus one panel per model for a sample and save the figure.

    Images larger than max_render_side are downscaled with every box rescaled
    alongside. The zoom crop is shared by all panels and covers the union of the
    GT and all parsed predictions. Returns the manifest record.
    """
    from PIL import Image
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    cjk_font = FontProperties(fname=CJK_FONT_PATH) if Path(CJK_FONT_PATH).exists() else FontProperties()

    if not sample['image_path']:
        raise FileNotFoundError(f"Image not found: {sample['image_id']}")

    image = Image.open(sample['image_path']).convert('RGB')
    if max(image.size) > max_render_side:
        scale = max_render_side / max(image.size)
        new_size = (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        def _scale_bbox(bbox):
            return [float(v) * scale for v in bbox] if bbox is not None else None
        def _scale_poly(poly):
            return [[float(x) * scale, float(y) * scale] for x, y in poly] if poly is not None else None
        sample = dict(sample)
        sample['gt_bbox'] = _scale_bbox(sample.get('gt_bbox'))
        sample['gt_poly'] = _scale_poly(sample.get('gt_poly'))
        scaled_results = {}
        for model_name, result in model_results.items():
            if result is None:
                scaled_results[model_name] = None
                continue
            r = dict(result)
            if r.get('pred_hbb') is not None:
                r['pred_hbb'] = _scale_bbox(r['pred_hbb'])
            if r.get('pred_obb') is not None:
                r['pred_obb'] = _scale_poly(r['pred_obb'])
            scaled_results[model_name] = r
        model_results = scaled_results
    gt_bbox = sample['gt_bbox']
    gt_poly = sample['gt_poly']
    if task == 'obb' and gt_poly is None and gt_bbox is not None:
        gt_poly = hbb_to_poly(gt_bbox)
    if task == 'hbb' and gt_bbox is None and gt_poly is not None:
        gt_bbox = poly_to_bbox(gt_poly)

    focus_boxes: List[List[float]] = []
    if gt_bbox is not None:
        focus_boxes.append([float(v) for v in gt_bbox])
    elif gt_poly is not None:
        focus_boxes.append(poly_to_bbox(gt_poly))
    for result in model_results.values():
        if not result or result.get('status') != 'ok':
            continue
        if result.get('pred_kind') == 'obb' and result.get('pred_obb') is not None:
            focus_boxes.append(poly_to_bbox(result['pred_obb']))
        elif result.get('pred_hbb') is not None:
            focus_boxes.append([float(v) for v in result['pred_hbb']])
    if not focus_boxes:
        raise RuntimeError('No box is available for cropping')
    crop_box = compute_crop(image.size, union_bbox(*focus_boxes))

    panels = [
        {
            'title': 'GT',
            'note': 'Ground Truth',
            'image': make_panel_image(task, image, crop_box, gt_bbox, gt_poly),
        }
    ]
    for model_name in MODEL_ORDER:
        result = model_results.get(model_name)
        panels.append({
            'title': MODEL_LABELS[model_name],
            'note': panel_note(task, result),
            'image': make_panel_image(
                task,
                image,
                crop_box,
                gt_bbox,
                gt_poly,
                pred_kind=result.get('pred_kind', 'none') if result else 'none',
                pred_hbb=result.get('pred_hbb') if result else None,
                pred_obb=result.get('pred_obb') if result else None,
                pred_color=MODEL_COLORS[model_name],
                pred_label=MODEL_LABELS[model_name] if result and result.get('pred_kind') != 'none' else None,
            ),
        })

    fig, axes = plt.subplots(1, len(panels), figsize=(4.25 * len(panels), 5.2))
    for ax, panel in zip(axes, panels):
        ax.imshow(panel['image'])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDBDBD')
            spine.set_linewidth(0.9)
        ax.set_title(panel['title'], fontsize=11, fontweight='bold', pad=8, fontproperties=cjk_font)
        ax.text(
            0.02,
            0.02,
            panel['note'],
            transform=ax.transAxes,
            fontsize=9,
            color='#222222',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.90, edgecolor='#BDBDBD'),
            ha='left',
            va='bottom',
            fontproperties=cjk_font,
        )

    fig.suptitle(f'Prompt: "{sample["question"]}"', fontsize=14, fontweight='bold', color='#1B5E20', y=0.975, fontproperties=cjk_font)
    fig.text(0.5, 0.925, f'Split: {sample["dataset"]} | Question ID: {sample["question_id"]}', ha='center', va='center', fontsize=10, color='#555555', fontproperties=cjk_font)
    path_text = textwrap.fill(f'Image: {sample["image_relpath"]}', width=160)
    fig.text(0.5, 0.03, path_text, ha='center', va='center', fontsize=9, color='#555555', fontproperties=cjk_font)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.10, wspace=0.04)
    ensure_parent(output_path)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

    return build_manifest_record(task, sample, model_results, output_path)


def build_manifest_record(task: str, sample: Dict[str, Any], model_results: Dict[str, Optional[Dict[str, Any]]], output_path: Path) -> Dict[str, Any]:
    """Compact per-sample summary (status, metric and pred_kind per model) for manifest.jsonl."""
    return {
        'uid': sample['uid'],
        'dataset': sample['dataset'],
        'question_id': sample['question_id'],
        'image_id': sample['image_id'],
        'image_path': sample['image_path'],
        'image_relpath': sample['image_relpath'],
        'figure_path': str(output_path),
        'question': sample['question'],
        'task': task,
        'results': {
            model_name: {
                'status': (model_results.get(model_name) or {}).get('status', 'not_run'),
                'metric_name': (model_results.get(model_name) or {}).get('metric_name'),
                'metric': (model_results.get(model_name) or {}).get('metric'),
                'pred_kind': (model_results.get(model_name) or {}).get('pred_kind', 'none'),
            }
            for model_name in MODEL_ORDER
        },
    }


# ---------------------------------------------------------------------------
# Inference workers (executed as `--worker infer` subprocesses)
# ---------------------------------------------------------------------------

def model_bundle(model_name: str) -> Dict[str, Optional[str]]:
    """Backend and checkpoint path for a model key."""
    if model_name == 'qwen':
        return {'backend': 'qwen', 'model_path': str(DEFAULT_BASE_MODEL), 'adapter_path': None}
    if model_name == 'sft':
        return {'backend': 'qwen', 'model_path': str(DEFAULT_SFT_MODEL), 'adapter_path': None}
    if model_name == 'gdpo':
        return {'backend': 'qwen', 'model_path': str(DEFAULT_GDPO_ROOT), 'adapter_path': None}
    if model_name == 'geoground':
        return {'backend': 'geoground', 'model_path': str(DEFAULT_GEOGROUND_MODEL), 'adapter_path': None}
    raise KeyError(f'Unknown model: {model_name}')


def require_matching_batch_size(requests: Sequence[Any], responses: Sequence[Any], backend: str) -> None:
    """Reject partial batches so cache rows cannot drift away from their samples."""
    if len(requests) != len(responses):
        raise RuntimeError(
            f'{backend} returned {len(responses)} responses for {len(requests)} requests'
        )


def worker_qwen_like(args: argparse.Namespace) -> int:
    """Worker for qwen / sft / gdpo: cache every pending sample of one split.

    Loads the checkpoint with ms-swift's TransformersEngine, decodes greedily,
    halves the batch on CUDA OOM, and appends parsed rows to the cache file after
    each batch so a killed run resumes with --resume.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from swift.infer_engine import TransformersEngine
    from swift.infer_engine.protocol import RequestConfig

    spec = resolve_dataset_spec_for_name(args.task, args.dataset, args)
    bundle = model_bundle(args.model)
    samples = load_samples(spec, max_samples=args.max_samples)
    cache_path = cache_file(Path(args.output_root), args.task, args.model, args.dataset)
    if cache_path.exists() and not args.resume:
        cache_path.unlink()
    existing = load_cache_map(cache_path) if (args.resume or args.skip_existing) else {}

    pending = [s for s in samples if s['uid'] not in existing]
    done_before = len(samples) - len(pending)
    log(f'worker[{args.model}/{args.dataset}] total={len(samples)} pending={len(pending)} batch={args.batch_size} gpu={args.gpu}')
    if not pending:
        return 0

    engine = TransformersEngine(
        model=bundle['model_path'],
        adapters=[bundle['adapter_path']] if bundle['adapter_path'] else None,
        model_type='qwen3_vl',
        torch_dtype=torch.bfloat16,
        max_batch_size=max(1, args.batch_size),
        attn_impl='flash_attn',
    )
    # The template reports whether the model emits norm1000 (default) or absolute coordinates.
    norm_bbox = getattr(getattr(engine, 'default_template', None), 'norm_bbox', None)
    request_config = RequestConfig(max_tokens=256, temperature=0, seed=42)

    idx = 0
    current_batch = max(1, args.batch_size)
    while idx < len(pending):
        batch = pending[idx : idx + current_batch]
        requests = []
        for sample in batch:
            prompt = get_obb_prompt(sample['question']) if args.task == 'obb' else get_hbb_prompt(sample['question'])
            requests.append({
                'messages': [{'role': 'user', 'content': f'<image>{prompt}'}],
                'images': [sample['image_path']],
            })
        try:
            responses = engine.infer(requests, request_config=request_config, use_tqdm=False)
        except Exception as exc:
            message = repr(exc)
            if 'out of memory' in message.lower() and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                log(f'worker[{args.model}/{args.dataset}] OOM; retrying with batch={current_batch}')
                torch.cuda.empty_cache()
                continue
            rows = []
            for sample in batch:
                rows.append({
                    'uid': sample['uid'],
                    'dataset': sample['dataset'],
                    'task': sample['task'],
                    'question_id': sample['question_id'],
                    'image_id': sample['image_id'],
                    'image_path': sample['image_path'],
                    'image_relpath': sample['image_relpath'],
                    'question': sample['question'],
                    'status': 'inference_error',
                    'pred_kind': 'none',
                    'pred_hbb': None,
                    'pred_obb': None,
                    'metric_name': None,
                    'metric': None,
                    'error': message,
                    'raw_output': '',
                    'model': args.model,
                })
            append_jsonl(cache_path, rows)
            idx += len(batch)
            log(f'worker[{args.model}/{args.dataset}] progress={done_before + idx}/{len(samples)} pending_done={idx}/{len(pending)} cache={resolve_repo_relpath(cache_path)}')
            torch.cuda.empty_cache()
            continue

        response_batch = responses if isinstance(responses, list) else [responses]
        require_matching_batch_size(batch, response_batch, 'Qwen')
        rows = []
        for sample, response in zip(batch, response_batch):
            text = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
            from PIL import Image
            image_size = Image.open(sample['image_path']).size
            parsed = parse_qwen_result(args.task, text, image_size, norm_bbox, sample['gt_bbox'], sample['gt_poly'])
            parsed.update({
                'uid': sample['uid'],
                'dataset': sample['dataset'],
                'task': sample['task'],
                'question_id': sample['question_id'],
                'image_id': sample['image_id'],
                'image_path': sample['image_path'],
                'image_relpath': sample['image_relpath'],
                'question': sample['question'],
                'model': args.model,
            })
            rows.append(parsed)
        append_jsonl(cache_path, rows)
        idx += len(batch)
        log(f'worker[{args.model}/{args.dataset}] progress={done_before + idx}/{len(samples)} pending_done={idx}/{len(pending)} cache={resolve_repo_relpath(cache_path)}')

    # Free model memory before the next queued worker claims this GPU.
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# GeoGround runs in its own worker because it needs the LLaVA-1.5 code base and environment.
def check_geoground_runtime(args: argparse.Namespace) -> int:
    """Probe that the llava package imports in this environment; prints a one-line JSON verdict."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    try:
        import llava  # noqa: F401
    except Exception as exc:
        payload = {'ok': False, 'reason': repr(exc)}
        print(json_dumps(payload), flush=True)
        return 1
    payload = {'ok': True, 'reason': None}
    print(json_dumps(payload), flush=True)
    return 0


def worker_geoground(args: argparse.Namespace) -> int:
    """Worker for GeoGround (LLaVA-1.5 code base) with the same cache protocol.

    Patches CLIPVisionTower to load local files only, wraps forward/generate for
    newer transformers versions (cache_position, inputs_embeds path), left-pads the
    prompts into a batch and decodes greedily on 336x336 inputs.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    from transformers import CLIPImageProcessor, CLIPVisionModel
    import types

    spec = resolve_dataset_spec_for_name(args.task, args.dataset, args)
    samples = load_samples(spec, max_samples=args.max_samples)
    cache_path = cache_file(Path(args.output_root), args.task, args.model, args.dataset)
    if cache_path.exists() and not args.resume:
        cache_path.unlink()
    existing = load_cache_map(cache_path) if (args.resume or args.skip_existing) else {}
    pending = [s for s in samples if s['uid'] not in existing]
    done_before = len(samples) - len(pending)
    log(f'worker[geoground/{args.dataset}] total={len(samples)} pending={len(pending)} batch={args.batch_size} gpu={args.gpu}')
    if not pending:
        return 0

    disable_torch_init()

    def _clip_load_model_local(self, device_map=None):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, local_files_only=True, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    CLIPVisionTower.load_model = _clip_load_model_local

    model_path = str(DEFAULT_GEOGROUND_MODEL)
    model_name = get_model_name_from_path(model_path)
    if 'llava' not in model_name.lower():
        model_name = f'llava-{model_name}'
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)

    # Older LLaVA versions do not accept the cache_position argument.
    original_forward = model.forward

    def _forward_compat(*f_args, **f_kwargs):
        f_kwargs.pop('cache_position', None)
        return original_forward(*f_args, **f_kwargs)

    model.forward = _forward_compat

    # Newer transformers route generate() through inputs_embeds; build the multimodal
    # embeddings here and call the plain HF generate with them.
    def _generate_compat(self, inputs=None, images=None, image_sizes=None, **kwargs):
        position_ids = kwargs.pop('position_ids', None)
        attention_mask = kwargs.pop('attention_mask', None)
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        return super(LlavaLlamaForCausalLM, self).generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    model.generate = types.MethodType(_generate_compat, model)

    idx = 0
    current_batch = max(1, args.batch_size)
    while idx < len(pending):
        batch = pending[idx : idx + current_batch]
        input_batch = []
        image_batch = []
        stop_str = None
        conv = conv_templates['llava_v1'].copy()

        for sample in batch:
            qs = get_geoground_obb_prompt(sample['question']) if args.task == 'obb' else get_geoground_hbb_prompt(sample['question'])
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_batch.append(input_ids)
            image = Image.open(sample['image_path']).convert('RGB').resize((336, 336))
            image_batch.append(image)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        max_length = max(tensor.size(1) for tensor in input_batch)
        final_input_list = [torch.cat((torch.zeros((1, max_length - tensor.size(1)), dtype=tensor.dtype, device=tensor.device), tensor), dim=1) for tensor in input_batch]
        final_input_tensors = torch.cat(final_input_list, dim=0)
        image_tensor_batch = image_processor.preprocess(image_batch, crop_size={'height': 336, 'width': 336}, size={'shortest_edge': 336}, return_tensors='pt')['pixel_values']

        try:
            with torch.inference_mode():
                output_ids = model.generate(final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False, temperature=0.0, top_p=None, num_beams=1, max_new_tokens=64, use_cache=True)
        except Exception as exc:
            message = repr(exc)
            if 'out of memory' in message.lower() and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                log(f'worker[geoground/{args.dataset}] OOM; retrying with batch={current_batch}')
                torch.cuda.empty_cache()
                continue
            rows = []
            for sample in batch:
                rows.append({
                    'uid': sample['uid'],
                    'dataset': sample['dataset'],
                    'task': sample['task'],
                    'question_id': sample['question_id'],
                    'image_id': sample['image_id'],
                    'image_path': sample['image_path'],
                    'image_relpath': sample['image_relpath'],
                    'question': sample['question'],
                    'status': 'inference_error',
                    'pred_kind': 'none',
                    'pred_hbb': None,
                    'pred_obb': None,
                    'metric_name': None,
                    'metric': None,
                    'error': message,
                    'raw_output': '',
                    'model': args.model,
                })
            append_jsonl(cache_path, rows)
            idx += len(batch)
            log(f'worker[geoground/{args.dataset}] progress={done_before + idx}/{len(samples)} pending_done={idx}/{len(pending)} cache={resolve_repo_relpath(cache_path)}')
            torch.cuda.empty_cache()
            continue

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        require_matching_batch_size(batch, outputs, 'GeoGround')
        rows = []
        for sample, output in zip(batch, outputs):
            text = output.strip()
            if stop_str and text.endswith(stop_str):
                text = text[:-len(stop_str)].strip()
            image_size = Image.open(sample['image_path']).size
            parsed = parse_geoground_result(args.task, text, image_size, sample['gt_bbox'], sample['gt_poly'])
            parsed.update({
                'uid': sample['uid'],
                'dataset': sample['dataset'],
                'task': sample['task'],
                'question_id': sample['question_id'],
                'image_id': sample['image_id'],
                'image_path': sample['image_path'],
                'image_relpath': sample['image_relpath'],
                'question': sample['question'],
                'model': args.model,
            })
            rows.append(parsed)
        append_jsonl(cache_path, rows)
        idx += len(batch)
        log(f'worker[geoground/{args.dataset}] progress={done_before + idx}/{len(samples)} pending_done={idx}/{len(pending)} cache={resolve_repo_relpath(cache_path)}')

    # Free model memory before the next queued worker claims this GPU.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------------------
# GeoGround runtime probes and cache repair (parent side)
# ---------------------------------------------------------------------------
# The probes keep an unavailable GeoGround runtime from blocking the other jobs.
def runtime_check_geoground(output_root: Path, conda_env: str, gpu: int) -> Dict[str, Any]:
    """Run the import probe inside the GeoGround environment and record the verdict in runtime_info.json."""
    cmd = build_python_cmd(conda_env) + [str(Path(__file__).resolve()), '--worker', 'check-geoground-runtime', '--gpu', str(gpu)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR)
    payload: Dict[str, Any] = {'ok': False, 'reason': 'GeoGround runtime check produced no output'}
    for line in reversed(result.stdout.strip().splitlines() if result.stdout else []):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if result.returncode != 0 and not payload.get('reason'):
        payload['reason'] = shorten_error(result.stderr or result.stdout)
    runtime_path = runtime_info_file(output_root)
    info = {}
    if runtime_path.exists():
        try:
            info = json.loads(runtime_path.read_text(encoding='utf-8'))
        except Exception:
            info = {}
    info['geoground_runtime'] = payload
    write_text(runtime_path, json_dumps(info))
    return payload


def maybe_probe_geoground_obb(output_root: Path, args: argparse.Namespace, gpu: int) -> bool:
    """Run a few OBB samples through GeoGround first; it is disabled for OBB when none parses."""
    if args.geoground_probe_samples <= 0:
        return True
    probe_dataset = next(iter(OBB_DATASETS.keys()))
    log(f'Starting GeoGround OBB probe: dataset={probe_dataset}, samples={args.geoground_probe_samples}, gpu={gpu}')
    cmd = build_python_cmd(args.geoground_conda_env) + [
        str(Path(__file__).resolve()),
        '--worker', 'infer',
        '--task', 'obb',
        '--dataset', probe_dataset,
        '--model', 'geoground',
        '--gpu', str(gpu),
        '--batch-size', '1',
        '--output-root', str(output_root),
        '--max-samples', str(args.geoground_probe_samples),
        '--resume',
        '--skip-existing',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR)
    if result.returncode != 0:
        log(f'GeoGround OBB probe failed: {shorten_error(result.stderr or result.stdout)}')
        return False
    cache_path = cache_file(output_root, 'obb', 'geoground', probe_dataset)
    cache_map = load_cache_map(cache_path)
    ok_count = sum(1 for item in cache_map.values() if item.get('status') == 'ok')
    log(f'GeoGround OBB probe finished: successful_samples={ok_count}')
    return ok_count > 0


def _copy_jsonl_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _repair_geoground_cache_for_dataset(source_root: Path, dest_root: Path, task: str, dataset: str, max_samples: Optional[int] = None) -> Path:
    """Re-parse the cached raw_output of one split with the current GeoGround decoder (no inference)."""
    spec = dataset_specs_for_task(task)[dataset]
    samples = load_samples(spec, max_samples=max_samples)
    sample_map = {sample['uid']: sample for sample in samples}

    src_cache = cache_file(source_root, task, 'geoground', dataset)
    dst_cache = cache_file(dest_root, task, 'geoground', dataset)
    if not src_cache.exists():
        raise FileNotFoundError(f'GeoGround cache not found: {src_cache}')

    ensure_parent(dst_cache)
    repaired_rows: List[Dict[str, Any]] = []
    for row in load_jsonl_multiline(src_cache):
        sample = sample_map.get(row['uid'])
        if sample is None:
            repaired_rows.append(row)
            continue
        image_size = Image.open(sample['image_path']).size
        parsed = parse_geoground_result(task, row.get('raw_output', ''), image_size, sample.get('gt_bbox'), sample.get('gt_poly'))
        parsed.update({
            'uid': row.get('uid'),
            'dataset': row.get('dataset', dataset),
            'task': row.get('task', task),
            'question_id': row.get('question_id'),
            'image_id': row.get('image_id'),
            'image_path': row.get('image_path', sample.get('image_path', '')),
            'image_relpath': row.get('image_relpath', sample.get('image_relpath', '')),
            'question': row.get('question', sample.get('question', '')),
            'model': row.get('model', 'geoground'),
        })
        repaired_rows.append(parsed)

    with dst_cache.open('w', encoding='utf-8') as f:
        for row in repaired_rows:
            f.write(json_dumps(row) + '\n')
    return dst_cache


def repair_geoground_cache(args: argparse.Namespace, source_root: Path) -> None:
    """--repair-geoground-cache: copy the Qwen-family caches and rebuild the GeoGround cache into output_root."""
    dest_root = Path(args.output_root)
    source_root = Path(source_root)

    ensure_parent(dest_root / 'placeholder.tmp')
    for task in resolve_task_list(args.task):
        datasets = resolve_datasets_with_args(task, args.datasets, args)
        task_dir = source_root / task / 'cache'
        for model in ('qwen', 'sft', 'gdpo'):
            _copy_jsonl_tree_if_exists(task_dir / model, dest_root / task / 'cache' / model)
        for spec in datasets:
            repaired_path = _repair_geoground_cache_for_dataset(source_root, dest_root, task, spec.name, max_samples=args.max_samples)
            log(f'Repaired GeoGround cache: {resolve_repo_relpath(repaired_path)}')

    runtime_info_src = runtime_info_file(source_root)
    runtime_info_dst = runtime_info_file(dest_root)
    if runtime_info_src.exists():
        ensure_parent(runtime_info_dst)
        shutil.copy2(runtime_info_src, runtime_info_dst)


# ---------------------------------------------------------------------------
# Parent-side scheduler
# ---------------------------------------------------------------------------

def build_jobs(task: str, datasets: List[DatasetSpec], models: List[str], geoground_enabled: bool) -> List[Job]:
    jobs: List[Job] = []
    for model in models:
        if model == 'geoground' and not geoground_enabled:
            continue
        for spec in datasets:
            jobs.append(Job(task=task, model=model, dataset=spec.name))
    return jobs


def conda_env_for_model(model_name: str, args: argparse.Namespace) -> str:
    return args.geoground_conda_env if model_name == 'geoground' else args.conda_env


def build_worker_command(job: Job, batch_size: int, args: argparse.Namespace) -> List[str]:
    """Command line of a worker subprocess; the '{GPU}' placeholder is filled in by spawn_job."""
    cmd = build_python_cmd(conda_env_for_model(job.model, args)) + [
        str(Path(__file__).resolve()),
        '--worker', 'infer',
        '--task', job.task,
        '--dataset', job.dataset,
        '--model', job.model,
        '--gpu', '{GPU}',
        '--batch-size', str(batch_size),
        '--output-root', str(args.output_root),
        '--resume' if args.resume else '--no-resume',
        '--skip-existing' if args.skip_existing else '--no-skip-existing',
    ] + ([] if args.max_samples is None else ['--max-samples', str(args.max_samples)])
    if args.custom_metainfo_file is not None:
        cmd += [
            '--custom-metainfo-file', str(args.custom_metainfo_file),
            '--custom-image-subdir', str(args.custom_image_subdir),
            '--custom-dataset-name', str(args.custom_dataset_name or args.custom_metainfo_file.stem),
            '--custom-default-batch-size', str(args.custom_default_batch_size),
        ]
        if args.custom_pretty_name:
            cmd += ['--custom-pretty-name', str(args.custom_pretty_name)]
    return cmd


def spawn_job(job: Job, gpu: GPUState, args: argparse.Namespace):
    """Start one worker on a GPU (log under logs/); None when the GPU cannot take the job right now."""
    spec = resolve_dataset_spec_for_name(job.task, job.dataset, args)
    batch_size = choose_batch_size(job, spec, gpu, args)
    if batch_size <= 0:
        return None
    cmd = build_worker_command(job, batch_size, args)
    cmd = [str(gpu.index) if x == '{GPU}' else x for x in cmd]
    log_path = logs_dir(Path(args.output_root)) / f'{job.task}__{job.model}__{job.dataset}__gpu{gpu.index}.log'
    ensure_parent(log_path)
    log_file = log_path.open('w', encoding='utf-8')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu.index)
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    log(f'Started job={job.key()} gpu={gpu.index} batch={batch_size} log={resolve_repo_relpath(log_path)} pid={proc.pid}')
    return {'proc': proc, 'log_file': log_file, 'job': job, 'gpu': gpu, 'batch_size': batch_size, 'log_path': log_path}


def run_infer_jobs(task: str, datasets: List[DatasetSpec], models: List[str], args: argparse.Namespace) -> None:
    """Run every (model, dataset) job of a task on the GPU pool, one worker per GPU.

    Validates the pool with nvidia-smi and probes GeoGround once, then loops:
    reap finished workers, re-query free memory, start the first queued job whose
    batch size fits on each idle GPU, and sleep --scheduler-poll-sec. Raises after
    the queue drains if any worker exited non-zero.
    """
    try:
        pool = parse_gpu_pool(args.geoground_gpu_pool if models == ['geoground'] else args.gpu_pool)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Unable to query GPU availability') from exc
    if not pool:
        raise RuntimeError('GPU pool is empty')
    try:
        visible_gpus = query_gpus(pool)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Unable to query GPU availability') from exc
    if not visible_gpus:
        raise RuntimeError(f'No requested GPU exists or is visible: {pool}')
    visible_ids = {gpu.index for gpu in visible_gpus}
    missing_ids = [gpu_id for gpu_id in pool if gpu_id not in visible_ids]
    if missing_ids:
        raise RuntimeError(f'Requested GPU IDs are not visible: {missing_ids}')

    geoground_enabled = True
    if 'geoground' in models:
        probe_gpu = visible_gpus[0].index
        runtime_payload = runtime_check_geoground(Path(args.output_root), args.geoground_conda_env, probe_gpu)
        geoground_enabled = bool(runtime_payload.get('ok'))
        if geoground_enabled and task == 'obb':
            geoground_enabled = maybe_probe_geoground_obb(Path(args.output_root), args, probe_gpu)
        if not geoground_enabled:
            log(f'GeoGround disabled: {runtime_payload.get("reason") or "probe failed"}')

    jobs = build_jobs(task, datasets, models, geoground_enabled)
    if not jobs:
        log(f'No inference jobs available for {task}')
        return

    queue: List[Job] = jobs.copy()
    running: Dict[int, Dict[str, Any]] = {}
    failed_jobs: List[str] = []
    max_workers = len(pool) if args.max_workers <= 0 else min(args.max_workers, len(pool))

    while queue or running:
        # Reap finished workers and release their GPUs.
        finished_gpus: List[int] = []
        for gpu_idx, info in list(running.items()):
            proc = info['proc']
            ret = proc.poll()
            if ret is None:
                continue
            info['log_file'].close()
            job = info['job']
            if ret == 0:
                log(f'Completed job={job.key()} gpu={gpu_idx}')
            else:
                log(f'Failed job={job.key()} gpu={gpu_idx} code={ret} log={resolve_repo_relpath(info["log_path"])}')
                failed_jobs.append(f'{job.key()} (exit {ret}, log {resolve_repo_relpath(info["log_path"])})')
            finished_gpus.append(gpu_idx)
        for gpu_idx in finished_gpus:
            running.pop(gpu_idx, None)

        if not queue and not running:
            break

        if len(running) >= max_workers:
            time.sleep(args.scheduler_poll_sec)
            continue

        # Launch on idle GPUs, taking the first queued job that fits the free memory.
        states = query_gpus(pool)
        launched = False
        for state in states:
            if state.index in running:
                continue
            if len(running) >= max_workers:
                break
            chosen: Optional[Job] = None
            for job in queue:
                spec = resolve_dataset_spec_for_name(job.task, job.dataset, args)
                if choose_batch_size(job, spec, state, args) > 0:
                    chosen = job
                    break
            if chosen is None:
                continue
            job_info = spawn_job(chosen, state, args)
            if job_info is None:
                continue
            running[state.index] = job_info
            queue.remove(chosen)
            launched = True

        if not launched:
            wait_msg = 'Waiting for an available GPU...' if queue else 'Waiting for remaining jobs...'
            log(wait_msg)
            time.sleep(args.scheduler_poll_sec)

    if failed_jobs:
        raise RuntimeError(f'{len(failed_jobs)} worker job(s) failed: ' + '; '.join(failed_jobs))


# ---------------------------------------------------------------------------
# Rendering pass over cached results
# ---------------------------------------------------------------------------
# Only caches are read here; inference workers never touch figure outputs.
def result_for_uid(cache_map: Dict[str, Dict[str, Any]], uid: str) -> Optional[Dict[str, Any]]:
    return cache_map.get(uid)


def render_task(task: str, datasets: List[DatasetSpec], args: argparse.Namespace) -> None:
    """Render every sample of the given splits from the caches.

    Appends a manifest row per sample and failure rows for models whose cached
    status is not 'ok' or that have no cached result; --skip-existing reuses
    figures already on disk and --render-ready-only waits for complete rows.
    """
    manifest_path = manifest_file(Path(args.output_root), task)
    failures_path = failures_file(Path(args.output_root), task)
    if manifest_path.exists() and not args.append_manifest:
        manifest_path.unlink()
    if failures_path.exists() and not args.append_manifest:
        failures_path.unlink()

    for spec in datasets:
        samples = load_samples(spec, max_samples=args.max_samples)
        caches = {model: load_cache_map(cache_file(Path(args.output_root), task, model, spec.name)) for model in MODEL_ORDER}
        log(f'Rendering {task}/{spec.name}: samples={len(samples)}')
        created = 0
        reused = 0
        waiting = 0
        failed = 0
        total = len(samples)
        for sample_idx, sample in enumerate(samples, 1):
            fig_name = f'{spec.name}__{Path(sample["image_id"]).stem}__{sample["uid"]}.{args.save_format}'
            out_path = render_dir(Path(args.output_root), task, spec.name) / fig_name
            model_results = {model: result_for_uid(caches[model], sample['uid']) for model in MODEL_ORDER}
            if args.render_ready_only and any(model_results.get(model) is None for model in MODEL_ORDER):
                waiting += 1
                if sample_idx % args.render_log_every == 0 or sample_idx == total:
                    log(f'render[{task}/{spec.name}] progress={sample_idx}/{total} created={created} reused={reused} waiting={waiting} failed={failed}')
                continue
            try:
                if args.skip_existing and out_path.exists():
                    manifest = build_manifest_record(task, sample, model_results, out_path)
                    reused += 1
                else:
                    manifest = render_single_figure(task, sample, model_results, out_path, max_render_side=args.max_render_side)
                    created += 1
                append_jsonl(manifest_path, [manifest])
                failure_rows = []
                for model_name, result in model_results.items():
                    if result and result.get('status') not in {'ok'}:
                        failure_rows.append({
                            'task': task,
                            'dataset': spec.name,
                            'uid': sample['uid'],
                            'question_id': sample['question_id'],
                            'image_id': sample['image_id'],
                            'model': model_name,
                            'status': result.get('status'),
                            'error': result.get('error'),
                        })
                    if result is None:
                        failure_rows.append({
                            'task': task,
                            'dataset': spec.name,
                            'uid': sample['uid'],
                            'question_id': sample['question_id'],
                            'image_id': sample['image_id'],
                            'model': model_name,
                            'status': 'not_run',
                            'error': 'Cached result not found',
                        })
                if failure_rows:
                    append_jsonl(failures_path, failure_rows)
            except Exception as exc:
                failed += 1
                append_jsonl(failures_path, [{
                    'task': task,
                    'dataset': spec.name,
                    'uid': sample['uid'],
                    'question_id': sample['question_id'],
                    'image_id': sample['image_id'],
                    'model': 'render',
                    'status': 'render_error',
                    'error': traceback.format_exc() if args.debug else repr(exc),
                }])
            if sample_idx % args.render_log_every == 0 or sample_idx == total:
                log(f'render[{task}/{spec.name}] progress={sample_idx}/{total} created={created} reused={reused} waiting={waiting} failed={failed}')
        log(f'Finished rendering {task}/{spec.name}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_model_list(model_text: str) -> List[str]:
    if model_text == 'all':
        return MODEL_ORDER.copy()
    models = [x.strip().lower() for x in model_text.split(',') if x.strip()]
    for model in models:
        if model not in MODEL_ORDER:
            raise KeyError(f'Unknown model: {model}')
    return models


def positive_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    ivalue = int(value)
    return ivalue if ivalue > 0 else None


def parse_args() -> argparse.Namespace:
    """CLI; the trailing --worker/--dataset/--model/--gpu/--batch-size group is only used by subprocesses."""
    parser = argparse.ArgumentParser(description='Batch inference and five-panel visualization across evaluation sets')
    parser.add_argument('--task', choices=['obb', 'hbb', 'all'], default='all')
    parser.add_argument('--datasets', nargs='*', default=['all'], help='Dataset splits; all selects every split for the task')
    parser.add_argument('--models', default='all', help='all or a comma-separated subset of qwen,geoground,sft,gdpo')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--save-format', choices=['png', 'jpg'], default='png')
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    parser.set_defaults(skip_existing=True)
    parser.add_argument('--append-manifest', action='store_true', help='Append instead of replacing manifest and failure files')
    parser.add_argument('--render-ready-only', action='store_true', help='Render only samples with all model results available')
    parser.add_argument('--max-render-side', type=int, default=1600, help='Resize large images to this maximum side before rendering')

    parser.add_argument('--gpu-pool', default='auto', help='auto or a comma-separated GPU list such as 0,1,3')
    parser.add_argument('--geoground-gpu-pool', default='auto', help='GPU pool for the GeoGround runtime check and probe')
    parser.add_argument('--min-free-gb', type=float, default=18.0)
    parser.add_argument('--geoground-min-free-gb', type=float, default=24.0)
    parser.add_argument('--batch-policy', choices=['auto', 'fixed'], default='auto')
    parser.add_argument('--fixed-batch-size', type=int, default=None)
    parser.add_argument('--geoground-fixed-batch-size', type=int, default=None)
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum concurrent GPUs; <=0 uses the full pool')
    parser.add_argument('--scheduler-poll-sec', type=int, default=20)
    parser.add_argument('--conda-env', default=DEFAULT_QWEN_ENV, help='Conda env for the Qwen-family workers (default: the active one)')
    parser.add_argument('--geoground-conda-env', default=DEFAULT_GEOGROUND_ENV, help='Conda env with the GeoGround/LLaVA-1.5 runtime')
    parser.add_argument('--geoground-probe-samples', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render-log-every', type=int, default=100, help='Render progress interval in samples')
    parser.add_argument('--repair-geoground-cache', action='store_true', help='Reparse an existing GeoGround cache before rendering')
    parser.add_argument('--source-output-root', type=Path, default=OUTPUT_ROOT, help='Existing cache root used in repair mode')
    parser.add_argument('--custom-metainfo-file', type=Path, default=None, help='Custom metainfo JSONL for ad-hoc inference or rendering')
    parser.add_argument('--custom-image-subdir', default=None, help='Image subdirectory for custom metainfo')
    parser.add_argument('--custom-dataset-name', default=None, help='Custom dataset name; defaults to the metainfo filename')
    parser.add_argument('--custom-pretty-name', default=None, help='Display name for a custom dataset')
    parser.add_argument('--custom-default-batch-size', type=int, default=1, help='Default batch size for a custom dataset')

    parser.add_argument('--worker', choices=['infer', 'check-geoground-runtime'], default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=1)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.infer_only and args.render_only:
        raise ValueError('--infer-only and --render-only are mutually exclusive')
    if args.custom_metainfo_file is not None:
        if args.task == 'all':
            raise ValueError('--task must be obb or hbb with --custom-metainfo-file')
        if not args.custom_metainfo_file.exists():
            raise FileNotFoundError(f'Custom metainfo file not found: {args.custom_metainfo_file}')
        if not args.custom_image_subdir:
            raise ValueError('--custom-image-subdir is required with --custom-metainfo-file')
    if args.worker == 'infer':
        if not args.dataset or not args.model:
            raise ValueError('--dataset and --model are required for worker=infer')


def worker_main(args: argparse.Namespace) -> int:
    """Dispatch a --worker invocation to the matching worker function."""
    if args.worker == 'check-geoground-runtime':
        return check_geoground_runtime(args)
    if args.worker == 'infer':
        bundle = model_bundle(args.model)
        if bundle['backend'] == 'qwen':
            return worker_qwen_like(args)
        return worker_geoground(args)
    raise ValueError(f'Unknown worker: {args.worker}')


def main() -> int:
    """Entry point: worker dispatch, cache repair, or inference followed by rendering per task."""
    args = parse_args()
    validate_args(args)
    if args.worker:
        return worker_main(args)

    if args.repair_geoground_cache:
        log(f'Repair mode: source={resolve_repo_relpath(Path(args.source_output_root))} -> dest={resolve_repo_relpath(Path(args.output_root))}')
        repair_geoground_cache(args, Path(args.source_output_root))
        if not args.infer_only:
            for task in resolve_task_list(args.task):
                render_task(task, resolve_datasets_with_args(task, args.datasets, args), args)
        return 0

    models = parse_model_list(args.models)
    tasks = resolve_task_list(args.task)
    runtime_info = runtime_info_file(Path(args.output_root))
    if runtime_info.exists() and not args.append_manifest:
        runtime_info.unlink()

    for task in tasks:
        datasets = resolve_datasets_with_args(task, args.datasets, args)
        log(f'==== Starting {task.upper()}: datasets={[d.name for d in datasets]}, models={models} ====')
        if not args.render_only:
            run_infer_jobs(task, datasets, models, args)
        if not args.infer_only:
            render_task(task, datasets, args)
        log(f'==== Finished {task.upper()} ====')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

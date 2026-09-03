"""Shared pieces of the baseline evaluation scripts.

Each baseline script plugs three model-specific callables into `run_evaluation`:
`infer_batch(image_paths, prompts) -> texts`, `make_prompt(question) -> str` and
`parse(text, image_size) -> box | None`. The driver owns everything else, so every
baseline is scored under the same protocol as `evaluation/`:

* HBB: 7 evaluation sets, Acc@0.5 / Acc@0.7 / mIoU with axis-aligned IoU.
* OBB: 3 evaluation sets, Acc@0.5 / Acc@0.7 / mRIoU with polygon (rotated) IoU.
* Ground truth lives in original-image pixels; `parse` must return pixels as well.
* A sample that has no image, fails inference or cannot be parsed scores 0 and stays
  in the denominator. Averages are macro averages over datasets.

Only the standard library, Pillow and tqdm are needed here; shapely is imported
lazily by `riou`, so the pure functions stay importable in every baseline's
environment.
"""
from __future__ import annotations

import glob
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm

# Evaluation splits: name -> (metainfo file, image sub-directory).
DATASETS = {
    "dior_rsvg_test": ("dior_rsvg_test.jsonl", "DIOR-RSVG"),
    "dior_rsvg_val": ("dior_rsvg_val.jsonl", "DIOR-RSVG"),
    "rsvg_test": ("rsvg_test.jsonl", "RSVG"),
    "rsvg_val": ("rsvg_val.jsonl", "RSVG"),
    "geochat_test": ("geochat_test.jsonl", "GeoChat"),
    "vrsbench_test": ("vrsbench_test.jsonl", "VRSBench"),
    "avvg_test": ("avvg_test.jsonl", "AVVG"),
}
# Only these splits carry real rotated annotations; the other four store the
# horizontal box as a degenerate polygon, so OBB is not scored on them.
OBB_DATASETS = ("geochat_test", "vrsbench_test", "avvg_test")

DEFAULT_METAINFO_DIR = "data/refGeo/metainfo"
DEFAULT_IMAGE_DIR = "data/refGeo/images"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".tif", ".bmp")


def load_jsonl(path: str) -> List[dict]:
    """Read JSONL, joining physical lines until a record parses (some files wrap records)."""
    rows, buf = [], ""
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            buf = f"{buf} {line}" if buf else line
            try:
                rows.append(json.loads(buf))
                buf = ""
            except json.JSONDecodeError:
                continue
    return rows


def find_image(root: str, subdir: str, image_id: str) -> Optional[str]:
    """Locate an image under `root/subdir` (root may be a glob), trying other extensions."""
    for base in glob.glob(os.path.join(root, subdir)) or [os.path.join(root, subdir)]:
        path = os.path.join(base, image_id)
        if os.path.exists(path):
            return path
        stem = os.path.splitext(image_id)[0]
        for ext in IMAGE_EXTENSIONS:
            path = os.path.join(base, stem + ext)
            if os.path.exists(path):
                return path
    return None


def clip_hbb(box: Sequence[float], size: Sequence[int]) -> List[float]:
    """Order the corners of [x1, y1, x2, y2] and clip them to the image."""
    w, h = size
    x1, x2 = sorted((float(box[0]), float(box[2])))
    y1, y2 = sorted((float(box[1]), float(box[3])))
    return [max(0.0, min(w - 1.0, x1)), max(0.0, min(h - 1.0, y1)),
            max(0.0, min(w - 1.0, x2)), max(0.0, min(h - 1.0, y2))]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Axis-aligned IoU of two [x1, y1, x2, y2] boxes."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (max(0, a[2] - a[0]) * max(0, a[3] - a[1])
             + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter)
    return inter / union if union else 0.0


def riou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Polygon IoU of two 4-point boxes.

    Corners are first sorted by polar angle around the centroid: a model may emit
    the four points in any order, and an unsorted order can form a self-intersecting
    polygon that shapely rejects as invalid (which would score 0 unfairly).
    """
    try:
        from shapely.geometry import Polygon

        def order(pts):
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        pa = Polygon(order([(float(p[0]), float(p[1])) for p in a]))
        pb = Polygon(order([(float(p[0]), float(p[1])) for p in b]))
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        union = pa.union(pb).area
        return pa.intersection(pb).area / union if union else 0.0
    except Exception:
        return 0.0


def metric_keys(task: str) -> tuple:
    return ("acc@0.5", "acc@0.7", "mIoU" if task == "hbb" else "mRIoU")


def summarize(scores: List[float], task: str) -> Dict[str, float]:
    n = len(scores)
    acc05, acc07, miou = metric_keys(task)
    return {
        acc05: sum(s >= 0.5 for s in scores) / n if n else 0.0,
        acc07: sum(s >= 0.7 for s in scores) / n if n else 0.0,
        miou: sum(scores) / n if n else 0.0,
    }


def add_common_args(parser, batch_size: int = 8) -> None:
    """CLI flags shared by every baseline script."""
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all",
                        help="Evaluation split, or all (7 splits for HBB, 3 for OBB)")
    parser.add_argument("--metainfo_dir", default=DEFAULT_METAINFO_DIR, help="Directory of refGeo test JSONL files")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="refGeo image root (glob patterns allowed)")
    parser.add_argument("--output_dir", required=True, help="Results go to a timestamped sub-directory of this path")
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate only the first N samples per split (smoke test)")


def select_datasets(task: str, dataset: str) -> List[str]:
    if dataset != "all":
        return [dataset]
    return list(OBB_DATASETS if task == "obb" else DATASETS)


def make_run_dir(output_dir: str) -> Path:
    run = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_markdown(summary: Dict[str, dict], task: str, model_name: str, run_dir: Path) -> None:
    """Write a two-level table (dataset x metric, plus AVG) ready to paste into a paper.

    `evaluation_metrics.md` inside the run directory is overwritten; the same table
    is also appended with a timestamp to `table_<task>_*.md` one level up, so
    repeated runs of one baseline accumulate in a single file.
    """
    metrics = metric_keys(task)
    titles = {"acc@0.5": "Acc@0.5", "acc@0.7": "Acc@0.7", "mIoU": "mIoU", "mRIoU": "mRIoU"}
    groups = list(summary) + ["AVG"]
    top = ['  <tr><th rowspan="2">Model</th>'] + [f'<th colspan="3">{g}</th>' for g in groups] + ["</tr>"]
    second = ["  <tr>"] + [f"<th>{titles[m]}</th>" for _ in groups for m in metrics] + ["</tr>"]
    row = [f"  <tr><td>{model_name}</td>"]
    avg = {m: [] for m in metrics}
    for ds in summary:
        for m in metrics:
            value = summary[ds][m]
            avg[m].append(value)
            row.append(f'<td align="right">{value * 100:.2f}</td>')
    for m in metrics:
        row.append(f'<td align="right">{sum(avg[m]) / len(avg[m]) * 100:.2f}</td>' if avg[m] else "<td>-</td>")
    row.append("</tr>")
    lines = [f"# {model_name} {task.upper()} evaluation: Acc@0.5 / Acc@0.7 / {metrics[2]}", "",
             "<table>", "<thead>", "".join(top), "".join(second), "</thead>",
             "<tbody>", "".join(row), "</tbody>", "</table>"]
    with open(run_dir / "evaluation_metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    table_file = run_dir.parent / f"table_{task}_acc05_acc07_{metrics[2].lower()}.md"
    with open(table_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n" + "\n".join(lines) + "\n")
    print(f"Markdown table written to {run_dir / 'evaluation_metrics.md'} and appended to {table_file}")


def run_evaluation(
    task: str,
    datasets: Sequence[str],
    infer_batch: Callable[[List[str], List[str]], List[str]],
    make_prompt: Callable[[str], str],
    parse: Callable[[str, Sequence[int]], Optional[list]],
    args,
    model_name: str,
    summary_extra: Optional[dict] = None,
) -> Dict[str, dict]:
    """Evaluate one model on the requested splits and write all result files.

    Per split: load metainfo, keep rows with a ground truth (`bbox` for HBB, `poly`
    for OBB), resolve images, run batched inference, parse and score. When a whole
    batch raises (typically OOM) every sample of that batch is retried alone so one
    failure does not silently zero the batch. Files written to the run directory:
    `<split>_<task>_predictions.jsonl`, `summary_<task>.json`, `evaluation_metrics.md`.
    """
    score = iou if task == "hbb" else riou
    score_key = "iou" if task == "hbb" else "rotated_iou"
    gt_key = "bbox" if task == "hbb" else "poly"
    run_dir = make_run_dir(args.output_dir)
    summary: Dict[str, dict] = {}

    for ds in datasets:
        meta, subdir = DATASETS[ds]
        rows = load_jsonl(os.path.join(args.metainfo_dir, meta))
        if args.max_samples is not None:
            rows = rows[:args.max_samples]
        items = [(row, row[gt_key], find_image(args.image_dir, subdir, row["image_id"]))
                 for row in rows if row.get(gt_key) is not None]

        records, scores = [], []
        for start in tqdm(range(0, len(items), args.batch_size), desc=f"{ds}-{task}"):
            batch = items[start:start + args.batch_size]
            valid = [(row, path) for row, _, path in batch if path]
            texts: List[str] = []
            if valid:
                prompts = [make_prompt(row["question"]) for row, _ in valid]
                try:
                    texts = infer_batch([path for _, path in valid], prompts)
                except Exception as exc:
                    print(f"[warning] {ds}: batch inference failed ({type(exc).__name__}: {str(exc)[:120]}); "
                          "retrying one sample at a time", flush=True)
                    texts = []
                    for (row, path), prompt in zip(valid, prompts):
                        try:
                            texts.append(infer_batch([path], [prompt])[0])
                        except Exception as exc2:
                            texts.append(f"ERROR: {exc2}")
            k = 0
            for row, gt, path in batch:
                text, error = "", None
                if path:
                    text = texts[k]
                    k += 1
                    if text.startswith("ERROR:"):  # never parse an error message: it may contain numbers
                        error, text = text, ""
                else:
                    error = "image not found"
                pred = None
                if path and text:
                    try:
                        pred = parse(text, Image.open(path).size)
                    except Exception:
                        pred = None
                value = score(pred, gt) if pred else 0.0
                scores.append(value)
                records.append({
                    "question_id": row.get("question_id", row.get("id")),
                    "image_id": row["image_id"],
                    "output": text,
                    "prediction": pred,
                    "ground_truth": gt,
                    score_key: value,
                    "error": error,
                })

        with open(run_dir / f"{ds}_{task}_predictions.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary[ds] = {
            "count": len(scores),
            "infer_errors": sum(1 for r in records if r["error"]),
            "parse_fail": sum(1 for r in records if r["prediction"] is None),
            **summarize(scores, task),
        }
        print(f"[{ds}] {json.dumps(summary[ds])}", flush=True)

    with open(run_dir / f"summary_{task}.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, **(summary_extra or {}), "datasets": summary}, f, ensure_ascii=False, indent=2)
    write_markdown(summary, task, model_name, run_dir)
    keys = metric_keys(task)
    macro = {m: sum(summary[ds][m] for ds in summary) / len(summary) for m in keys} if summary else {}
    print(f"Macro average over {len(summary)} split(s): "
          + ", ".join(f"{m}={macro[m] * 100:.2f}" for m in keys if m in macro))
    print(f"Results saved to {run_dir}")
    return summary

#!/usr/bin/env python3
"""HuggingFace-transformers baselines (LLaVA-OneVision-1.5, InternVL3-HF): zero-shot or LoRA fine-tuned.

Reuses the prompts, parsers and scoring of `evaluation/evaluate_hbb.py` and
`evaluate_obb.py`, so these rows are scored exactly like GeoBox-R1, while inference
runs through plain transformers (AutoProcessor + chat template, trust_remote_code)
instead of ms-swift. Rows reproduced:
    LLaVA-OV-1.5 (zero-shot)  --prompt_mode zeroshot --zeroshot_style qwenvl
    LLaVA-OV-1.5 (SFT)        --adapter_dir ... --prompt_mode trained --coord_space norm1 --max_pixels 802816
    InternVL3 (SFT)           --adapter_dir ... --prompt_mode trained --crop_to_patches
(the adapters come from `baselines/finetune/run_llava_ov15.sh` / `run_internvl3.sh`).
GeoChat and GeoGround have their own conversation pipelines and their own scripts.

Coordinates: `trained` mode parses the training format and scales it by
`--coord_space` (norm1000 by default, i.e. the GeoBox-R1 convention); `zeroshot` mode
parses leniently and infers 0-1 coordinates from the value range.

Environment: transformers >= 4.55, peft, flash-attn (or pass --attn_implementation sdpa).
Usage, from the repository root:
    python baselines/evaluate/eval_hf.py --model_path models/pretrained/LLaVA-OneVision-1.5-4B-Instruct \
        --task hbb --prompt_mode zeroshot --zeroshot_style qwenvl --run_name ov15_4b_zeroshot
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm

# Prompts, parsers and metrics are shared with the main evaluation scripts.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
import evaluate_hbb as hbb_base  # noqa: E402
import evaluate_obb as obb_base  # noqa: E402


# --------------------------------------------------------------------------- arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF transformers baseline evaluation (zero-shot / SFT)")
    parser.add_argument("--model_path", required=True, help="Base model directory or HF repo id")
    parser.add_argument("--adapter_dir", default=None, help="PEFT LoRA adapter to merge (SFT evaluation)")
    parser.add_argument("--task", choices=("hbb", "obb"), default="hbb")
    parser.add_argument("--dataset", default="all", help="Evaluation split name, or all")
    parser.add_argument("--prompt_mode", choices=("trained", "zeroshot"), default="trained",
                        help="trained: the SFT prompt and training-format parser; zeroshot: model-native prompt and lenient parser")
    parser.add_argument("--zeroshot_style", choices=("qwenvl", "generic"), default="qwenvl",
                        help="qwenvl: JSON bbox_2d in absolute pixels (used for LLaVA-OV-1.5); generic: the training prompt")
    parser.add_argument("--coord_space", choices=("auto", "norm1", "norm100", "norm1000", "abs"), default="auto",
                        help="Coordinate space of the model output; auto = norm1000 in trained mode, inferred by style in zeroshot mode")
    parser.add_argument("--metainfo_dir", default="data/refGeo/metainfo")
    parser.add_argument("--image_dir", default="data/refGeo/images")
    parser.add_argument("--output_dir", default="eval_results/baselines")
    parser.add_argument("--run_name", default=None, help="Result sub-directory; default: <model>_<task>_<mode>_<timestamp>")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate only the first N samples per split (smoke test)")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Override max_pixels of Qwen2-VL-style image processors (controls the image token budget)")
    parser.add_argument("--crop_to_patches", action="store_true",
                        help="InternVL dynamic high-resolution tiling; must match the training setting")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


# --------------------------------------------------------------------------- model loading
def load_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    load_kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
                       attn_implementation=args.attn_implementation, device_map="cuda")
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, **load_kwargs)
    except (ValueError, KeyError, OSError):
        # Custom architectures such as LLaVA-OneVision-1.5 only register AutoModelForCausalLM.
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model = model.merge_and_unload()  # merged weights infer faster
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"  # batched generation needs left padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.max_pixels is not None and hasattr(processor, "image_processor"):
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = args.max_pixels
            if isinstance(getattr(processor.image_processor, "size", None), dict):
                processor.image_processor.size["longest_edge"] = args.max_pixels
    return model, processor


# --------------------------------------------------------------------------- prompts
def build_prompt(question: str, task: str, prompt_mode: str, zeroshot_style: str) -> str:
    if prompt_mode == "trained":
        return hbb_base.get_prompt(question) if task == "hbb" else obb_base.get_obb_prompt(question)
    if task == "hbb":
        if zeroshot_style == "qwenvl":
            return (
                f'Locate the object this description refers to: "{question}". '
                'Output its bounding box in JSON format: {"bbox_2d": [x1, y1, x2, y2]} '
                "using absolute pixel coordinates."
            )
        return hbb_base.get_prompt(question)
    # Zero-shot OBB keeps the training prompt: general models rarely answer it, which is reported as is.
    if zeroshot_style == "qwenvl":
        return obb_base.get_obb_prompt(question) + "\nUse absolute pixel coordinates."
    return obb_base.get_obb_prompt(question)


# --------------------------------------------------------------------------- lenient parsing (zero-shot)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _find_json_bbox(output: str) -> Optional[list[float]]:
    """Any JSON object in the output carrying a 4-number box field."""
    cleaned = output.replace("```json", "").replace("```", "")
    for match in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            obj = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        for key in ("horizontal_bbox", "bbox_2d", "bbox", "box"):
            value = obj.get(key)
            if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
                return [float(v) for v in value]
    return None


def parse_hbb_tolerant(output: str) -> Optional[list[float]]:
    """Training format -> any JSON box -> `<box>(x1,y1),(x2,y2)</box>` -> first four numbers."""
    bbox = hbb_base.parse_bbox_from_output(output)
    if bbox:
        return [float(v) for v in bbox]
    bbox = _find_json_bbox(output)
    if bbox:
        return bbox
    match = re.search(r"<box>\s*\(?([\d.]+),\s*([\d.]+)\)?\s*,\s*\(?([\d.]+),\s*([\d.]+)\)?\s*</box>", output)
    if match:
        return [float(g) for g in match.groups()]
    values = _FLOAT_RE.findall(output)
    if len(values) >= 4:
        return [float(v) for v in values[:4]]
    return None


def parse_obb_tolerant(output: str) -> Optional[list[list[float]]]:
    """Training format -> first eight numbers as four corners."""
    poly = obb_base.parse_obb_from_output(output)
    if poly:
        return poly
    values = _FLOAT_RE.findall(output)
    if len(values) >= 8:
        vals = [float(v) for v in values[:8]]
        return [[vals[i], vals[i + 1]] for i in range(0, 8, 2)]
    return None


def resolve_coord_space(coord_space: str, zeroshot_style: str) -> str:
    if coord_space != "auto":
        return coord_space
    return {"qwenvl": "abs"}.get(zeroshot_style, "norm1000")


def auto_space(values: list[float], resolved: str, requested: str) -> str:
    """In auto mode, values that are all within [-2, 2] are taken as 0-1 coordinates
    (LLaVA-OneVision-1.5 answers in 0-1 even when asked for pixels)."""
    if requested == "auto" and values and max(abs(v) for v in values) <= 2.0:
        return "norm1"
    return resolved


def scale_coords(values: list[float], space: str, width: int, height: int) -> list[float]:
    """Scale an (x, y, x, y, ...) sequence to absolute pixels."""
    if space == "abs":
        return values
    div = {"norm1": 1.0, "norm100": 100.0, "norm1000": 1000.0}[space]
    return [float(v) / div * (width if i % 2 == 0 else height) for i, v in enumerate(values)]


# --------------------------------------------------------------------------- inference
IMAGE_KWARGS: dict = {}  # extra processor kwargs (e.g. crop_to_patches), filled in by main()


def generate_batch(model, processor, prompts: list[str], images: list[Image.Image], max_new_tokens: int) -> list[str]:
    import torch

    texts = []
    for prompt in prompts:
        message = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        texts.append(processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt", **IMAGE_KWARGS).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=(processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None))
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def to_pixels(pred, task: str, args: argparse.Namespace, width: int, height: int, coord_space: str):
    """Convert a parsed box from the model's coordinate space to original-image pixels."""
    if task == "hbb":
        if args.prompt_mode == "trained":
            if args.coord_space in ("norm1", "norm100", "norm1000"):
                return scale_coords(pred, args.coord_space, width, height)
            if args.coord_space == "abs":
                return pred
            return hbb_base._maybe_denormalize_bbox(pred, (width, height), "norm1000")
        return scale_coords(pred, auto_space(pred, coord_space, args.coord_space), width, height)
    flat = [c for point in pred for c in point]
    if args.prompt_mode == "trained":
        if args.coord_space in ("norm1", "norm100", "norm1000"):
            flat = scale_coords(flat, args.coord_space, width, height)
        elif args.coord_space == "abs":
            return pred
        else:
            return obb_base._maybe_denormalize_obb(pred, (width, height), "norm1000")
    else:
        flat = scale_coords(flat, auto_space(flat, coord_space, args.coord_space), width, height)
    return [[flat[i], flat[i + 1]] for i in range(0, 8, 2)]


def evaluate_one(model, processor, args, base, dataset_name: str, run_dir: Path) -> dict[str, Any]:
    """Evaluate one split; writes predictions_<split>.json and returns its metrics."""
    config = base.DATASET_CONFIGS[dataset_name]
    test_file = Path(args.metainfo_dir) / config["test_file"]
    image_dir = Path(args.image_dir) / config["image_subdir"]
    data = base.load_test_data(str(test_file))
    if args.max_samples:
        data = data[: args.max_samples]

    gt_key = "bbox" if args.task == "hbb" else "poly"
    coord_space = resolve_coord_space(args.coord_space, args.zeroshot_style)
    parse = (hbb_base.parse_bbox_from_output if args.prompt_mode == "trained" else parse_hbb_tolerant) \
        if args.task == "hbb" else (obb_base.parse_obb_from_output if args.prompt_mode == "trained" else parse_obb_tolerant)
    score = base.calculate_iou if args.task == "hbb" else base.calculate_rotated_iou

    records, ious = [], []
    batch: list[dict] = []

    def flush():
        nonlocal batch
        if not batch:
            return
        images = []
        for it in batch:
            with Image.open(it["image_path"]) as im:
                images.append(im.convert("RGB").copy())
        outputs = generate_batch(model, processor, [it["prompt"] for it in batch], images, args.max_new_tokens)
        for it, im, out in zip(batch, images, outputs):
            width, height = im.size
            pred, iou = None, 0.0
            try:
                pred = parse(out)
                if pred:
                    pred = to_pixels(pred, args.task, args, width, height, coord_space)
                    iou = score(pred, it["gt"])
            except Exception:
                # Malformed output (e.g. a truncated JSON corner) counts as a parse failure, not a crash.
                pred, iou = None, 0.0
            ious.append(iou)
            records.append({"question_id": it["qid"], "question": it["question"], "output": out,
                            "pred": pred, "gt": it["gt"], "iou": round(float(iou), 4)})
        batch = []

    missing = 0
    for item in tqdm(data, desc=dataset_name, ncols=100):
        gt = item.get(gt_key)
        if gt is None or len(gt) != 4:
            continue
        image_path = image_dir / item["image_id"]
        if not image_path.exists():
            stem = image_path.with_suffix("")
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                candidate = Path(str(stem) + ext)
                if candidate.exists():
                    image_path = candidate
                    break
        if not image_path.exists():
            missing += 1
            ious.append(0.0)
            records.append({"question_id": item.get("question_id", item.get("id")), "question": item["question"],
                            "output": "", "pred": None, "gt": gt, "iou": 0.0, "error": "image not found"})
            continue
        batch.append({
            "qid": item.get("question_id", item.get("id")),
            "question": item["question"],
            "prompt": build_prompt(item["question"], args.task, args.prompt_mode, args.zeroshot_style),
            "image_path": str(image_path),
            "gt": gt,
        })
        if len(batch) >= args.batch_size:
            flush()
    flush()

    total = len(ious)
    metrics = {
        "dataset": dataset_name,
        "total": total,
        "missing_images": missing,
        "acc@0.5": round(100.0 * sum(i >= 0.5 for i in ious) / total, 2) if total else None,
        "acc@0.7": round(100.0 * sum(i >= 0.7 for i in ious) / total, 2) if total else None,
        "miou": round(100.0 * sum(ious) / total, 2) if total else None,
        "parse_fail": sum(1 for r in records if r["pred"] is None),
    }
    with open(run_dir / f"predictions_{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": records}, f, ensure_ascii=False, indent=1)
    print(f"[{dataset_name}] {metrics}")
    return metrics


# --------------------------------------------------------------------------- main
def main() -> None:
    import torch

    args = parse_args()
    if args.crop_to_patches:
        IMAGE_KWARGS["crop_to_patches"] = True
    base = hbb_base if args.task == "hbb" else obb_base
    datasets = list(base.DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{Path(args.model_path).name}_{args.task}_{args.prompt_mode}_{stamp}"
    run_dir = Path(args.output_dir) / (args.run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args)

    all_metrics = []
    for name in datasets:
        all_metrics.append(evaluate_one(model, processor, args, base, name, run_dir))
        gc.collect()
        torch.cuda.empty_cache()

    # Merge with results already in this run directory, so splits can be run in several invocations.
    merged: dict[str, dict] = {}
    for prediction_file in run_dir.glob("predictions_*.json"):
        try:
            metrics = json.load(open(prediction_file, encoding="utf-8")).get("metrics")
            if metrics and metrics.get("dataset"):
                merged[metrics["dataset"]] = metrics
        except Exception:
            pass
    for metrics in all_metrics:
        merged[metrics["dataset"]] = metrics
    order = list(base.DATASET_CONFIGS)
    datasets_list = [merged[k] for k in order if k in merged] + [v for k, v in merged.items() if k not in order]

    valid = [m for m in datasets_list if m["total"]]
    summary = {
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "task": args.task,
        "prompt_mode": args.prompt_mode,
        "zeroshot_style": args.zeroshot_style if args.prompt_mode == "zeroshot" else None,
        "timestamp": stamp,
        "datasets": datasets_list,
        "macro_avg": {key: round(sum(m[key] for m in valid) / len(valid), 2) if valid else None
                      for key in ("acc@0.5", "acc@0.7", "miou")},
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    iou_label = "mIoU" if args.task == "hbb" else "mRIoU"
    lines = [f"# Evaluation summary: {run_dir.name}", "",
             f"- model: `{args.model_path}`",
             f"- adapter: `{args.adapter_dir or 'none (zero-shot)'}`",
             f"- task: {args.task.upper()} | prompt mode: {args.prompt_mode}", "",
             f"| Dataset | Acc@0.5 | Acc@0.7 | {iou_label} | Samples | Parse failures |",
             "|---|---|---|---|---|---|"]
    for m in datasets_list:
        lines.append(f"| {m['dataset']} | {m['acc@0.5']} | {m['acc@0.7']} | {m['miou']} | {m['total']} | {m['parse_fail']} |")
    macro = summary["macro_avg"]
    lines.append(f"| **Average** | **{macro['acc@0.5']}** | **{macro['acc@0.7']}** | **{macro['miou']}** | | |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults written to {run_dir}/summary.json and summary.md")


if __name__ == "__main__":
    main()

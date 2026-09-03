#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the five bundled demo examples from refGeo metadata."""

import json
import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REFGEO_ROOT = os.environ.get("REFGEO_ROOT", os.path.join(REPO_ROOT, "data", "refGeo"))
META = os.path.join(REFGEO_ROOT, "metainfo")
IMG_BASE = os.path.join(REFGEO_ROOT, "images")
EXAMPLES_DIR = os.path.join(HERE, "examples")
MANIFEST = os.path.join(HERE, "examples_manifest.json")

SUBDIR = {
    "dior_rsvg_test": "DIOR-RSVG",
    "rsvg_test": "RSVG",
    "vrsbench_test": "VRSBench",
    "avvg_test": "AVVG",
    "geochat_test": "GeoChat",
}
DATASET_LABEL = {
    "dior_rsvg_test": "DIOR-RSVG",
    "rsvg_test": "RSVG",
    "vrsbench_test": "VRSBench",
    "avvg_test": "AVVG",
    "geochat_test": "GeoChat",
}

# Dataset, image ID, question substring, task, and display note.
SELECTED = [
    ("dior_rsvg_test", "09017.jpg", "tennis court is on the upper left of the red vehicle",
     "hbb", "Tennis court located by its relation to the red vehicle."),
    ("vrsbench_test", "10255_0000.png", "airplane positioned on the far right",
     "hbb", "Rightmost airplane in a scene with several aircraft."),
    ("geochat_test", "train_10750_0000.jpg", "large liquid-cargo-ship",
     "obb", "Diagonal cargo ship suited to an oriented box."),
    ("avvg_test", "436392aa-DJI_0146.JPG", "black volkswagen mid-size car in the bottom-middle",
     "obb", "Slanted vehicle whose orientation is visually clear."),
    ("vrsbench_test", "P1476_0005.png", "large vehicle towards the middle of the row",
     "obb", "Diagonal vehicle where an oriented box avoids excess background."),
]

def load_jsonl(path):
    """Read JSONL while tolerating records split across physical lines."""
    data, buf = [], ""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            buf = f"{buf} {line}" if buf else line
            try:
                data.append(json.loads(buf))
                buf = ""
            except json.JSONDecodeError:
                continue
    return data


def resolve_image(img_dir, image_id):
    """Resolve metadata image IDs despite extension mismatches."""
    p = os.path.join(img_dir, image_id)
    if os.path.exists(p):
        return p
    base = os.path.splitext(image_id)[0]
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        q = os.path.join(img_dir, base + ext)
        if os.path.exists(q):
            return q
    return None


def main():
    cache = {}
    prepared = []
    manifest = []
    errors = []
    print("Preparing GeoBox-R1 demo examples")

    for idx, (ds, image_id, qsub, task, note) in enumerate(SELECTED, 1):
        jsonl = os.path.join(META, f"{ds}.jsonl")
        if ds not in cache:
            cache[ds] = load_jsonl(jsonl)
        records = cache[ds]

        base_id = os.path.splitext(image_id)[0]
        hit = None
        for r in records:
            if os.path.splitext(r.get("image_id", ""))[0] != base_id:
                continue
            if qsub.lower() in r.get("question", "").lower():
                hit = r
                break
        if hit is None:
            errors.append(f"[{idx:2d}] No matching record: {ds} {image_id} ~ '{qsub}'")
            continue

        img_dir = os.path.join(IMG_BASE, SUBDIR[ds])
        src = resolve_image(img_dir, hit["image_id"])
        if not src:
            errors.append(f"[{idx:2d}] Missing image: {ds} {hit['image_id']}")
            continue

        ext = os.path.splitext(src)[1]
        dst_name = f"{idx:02d}_{ds.replace('_test','')}_{base_id}{ext}"

        from PIL import Image
        with Image.open(src) as im:
            W, H = im.size

        entry = {
            "id": idx,
            "dataset": ds,
            "dataset_label": DATASET_LABEL[ds],
            "image_file": dst_name,
            "question": hit["question"],
            "recommended_task": task,
            "gt_bbox": [float(x) for x in hit["bbox"]],
            "gt_poly": [[float(p[0]), float(p[1])] for p in hit["poly"]] if hit.get("poly") else None,
            "image_width": W,
            "image_height": H,
            "note": note,
        }
        manifest.append(entry)
        prepared.append((src, entry))

    # Keep the published manifest intact unless every selected source resolves.
    if errors:
        raise RuntimeError("Unable to prepare all demo examples:\n" + "\n".join(errors))

    # Stage every asset before touching the published examples or manifest.
    with tempfile.TemporaryDirectory(dir=HERE, prefix=".prepare_examples.") as staging_root:
        staged_examples = os.path.join(staging_root, "examples")
        staged_manifest = os.path.join(staging_root, "examples_manifest.json")
        os.makedirs(staged_examples)

        from PIL import Image
        for src, entry in prepared:
            staged_image = os.path.join(staged_examples, entry["image_file"])
            shutil.copy2(src, staged_image)
            with Image.open(staged_image) as im:
                if im.size != (entry["image_width"], entry["image_height"]):
                    raise RuntimeError(f"Staged image has unexpected dimensions: {staged_image}")

        with open(staged_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        backup_examples = os.path.join(staging_root, "previous_examples")
        displaced_examples = os.path.join(staging_root, "failed_examples")
        had_examples = os.path.isdir(EXAMPLES_DIR)
        old_examples_moved = False
        new_examples_published = False
        try:
            if had_examples:
                os.replace(EXAMPLES_DIR, backup_examples)
                old_examples_moved = True
            os.replace(staged_examples, EXAMPLES_DIR)
            new_examples_published = True
            os.replace(staged_manifest, MANIFEST)
        except Exception:
            # Restore the previous directory if the final publication step fails.
            if new_examples_published and os.path.isdir(EXAMPLES_DIR):
                os.replace(EXAMPLES_DIR, displaced_examples)
            if old_examples_moved:
                os.replace(backup_examples, EXAMPLES_DIR)
            raise

    for _, entry in prepared:
        print(
            f"[{entry['id']:2d}] ✓ {entry['dataset_label']:10s} "
            f"{entry['image_file']:42s} {entry['image_width']}x{entry['image_height']} "
            f"[{entry['recommended_task']}]"
        )
        print(f"      Q: {entry['question'][:70]}")

    print(f"Copied {len(manifest)} examples to {EXAMPLES_DIR}")
    print(f"Wrote manifest to {MANIFEST}")


if __name__ == "__main__":
    main()

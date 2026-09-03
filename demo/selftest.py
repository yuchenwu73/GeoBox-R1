#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the bundled examples headlessly and save annotated outputs."""

import json
import os

import inference
import visualize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "selftest_out")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(HERE, "examples_manifest.json"), encoding="utf-8"))
    model = inference.get_model()

    print("=" * 86)
    print(f"{'#':>2}  {'Dataset':<11}{'Task':<5}{'Size':<11}{'IoU/RIoU':<10}{'Hit@0.5':<8} Query")
    print("-" * 86)

    ious, hits = [], 0
    for m in manifest:
        img = os.path.join(HERE, "examples", m["image_file"])
        task = m["recommended_task"]
        res = model.infer(img, m["question"], task)

        iou = None
        if res["parsed_ok"]:
            if task == "hbb" and res["bbox_px"] and m.get("gt_bbox"):
                iou = inference.iou_hbb(res["bbox_px"], m["gt_bbox"])
            elif task == "obb" and res["poly_px"] and m.get("gt_poly"):
                iou = inference.iou_obb(res["poly_px"], m["gt_poly"])

        viz = visualize.draw_result(img, res, gt_bbox=m.get("gt_bbox"),
                                    gt_poly=m.get("gt_poly"), iou=iou)
        viz.save(os.path.join(OUT, f"{m['id']:02d}_{task}_{m['dataset']}.png"))

        iou_str = f"{iou:.3f}" if iou is not None else "—"
        hit = (iou is not None and iou >= 0.5)
        if iou is not None:
            ious.append(iou)
            hits += int(hit)
        size = f"{m['image_width']}x{m['image_height']}"
        print(f"{m['id']:>2}  {m['dataset_label']:<11}{task:<5}{size:<11}{iou_str:<10}"
              f"{'✔' if hit else '✘':<8} {m['question'][:34]}")

    print("-" * 86)
    if ious:
        print(f"Mean IoU/RIoU: {sum(ious)/len(ious):.3f}   Hit rate@0.5: {hits}/{len(ious)} "
              f"({100*hits/len(ious):.1f}%)")
    print(f"Visualizations saved to: {OUT}")
    print("=" * 86)


if __name__ == "__main__":
    main()

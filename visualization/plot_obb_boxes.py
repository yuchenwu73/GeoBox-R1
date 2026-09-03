#!/usr/bin/env python3
"""Draw a ground-truth OBB and a predicted OBB on a crop of one image.

Both boxes are given on the command line as JSON pixel coordinates
([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]); the crop is computed around their
union and the figure is saved as <output-dir>/<prefix>_gt_pred.png.

Usage, from the repository root
    python visualization/plot_obb_boxes.py --image-path img.png --gt-obb '[[10,10],[60,10],[60,40],[10,40]]' --pred-obb '[[12,8],[62,12],[58,42],[8,38]]' --prefix ship
"""
import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'output' / 'visualizations' / 'gt_pred_boxes'
GT_COLOR = '#1E90FF'
PRED_COLOR = '#FF8F00'


def parse_json_arg(value: str):
    """argparse type: parse a JSON literal."""
    return json.loads(value)


def poly_to_bbox(poly: Sequence[Sequence[float]]):
    """Axis-aligned envelope of a polygon."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_bbox(*bboxes):
    """Envelope of several [x1, y1, x2, y2] boxes."""
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def compute_crop(image_size, focus_bbox):
    """Crop around both boxes with extra room for labels near the top-left."""
    img_w, img_h = image_size
    x1, y1, x2, y2 = focus_bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    margin_left = max(bw * 1.8, 70)
    margin_right = max(bw * 1.3, 70)
    margin_top = max(bh * 1.6, 60)
    margin_bottom = max(bh * 1.6, 60)
    cl = max(0, int(round(cx - margin_left)))
    ct = max(0, int(round(cy - margin_top)))
    cr = min(img_w, int(round(cx + margin_right)))
    cb = min(img_h, int(round(cy + margin_bottom)))
    return cl, ct, cr, cb


def shift_poly(poly, dx, dy):
    """Translate a polygon into crop coordinates."""
    return [[p[0] - dx, p[1] - dy] for p in poly]


def draw_poly(ax, poly, color, label, linestyle='-', linewidth=2.8, label_pos='top'):
    """Draw one polygon with a contrast stroke and an anchored label."""
    patch = MplPolygon(poly, closed=True, fill=False, edgecolor=color,
                       linewidth=linewidth, linestyle=linestyle, zorder=10)
    patch.set_path_effects([
        pe.withStroke(linewidth=linewidth + 2.0, foreground='black', alpha=0.22),
        pe.Normal(),
    ])
    ax.add_patch(patch)
    if label_pos == 'left_top':
        anchor = min(poly, key=lambda p: p[0] + p[1])
        x, y, ha = anchor[0] - 4, anchor[1] - 8, 'right'
    elif label_pos == 'top_right':
        anchor = max(poly, key=lambda p: p[0] - p[1])
        x, y, ha = anchor[0], anchor[1] - 6, 'left'
    elif label_pos == 'bottom_right':
        anchor = max(poly, key=lambda p: p[0] + p[1])
        x, y, ha = anchor[0] + 4, anchor[1] + 4, 'left'
    else:
        anchor = min(poly, key=lambda p: p[1])
        x, y, ha = anchor[0], anchor[1] - 6, 'left'
    txt = ax.text(x, y, label,
                  fontsize=11, fontweight='bold', color='white',
                  bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='none', alpha=0.95),
                  ha=ha, va='bottom', zorder=15)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground='black', alpha=0.15)])


def main():
    """CLI: crop around both boxes, draw GT (solid) and prediction (dashed), save the figure."""
    parser = argparse.ArgumentParser(description='Pure box OBB GT vs Pred visualization')
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--gt-obb', type=parse_json_arg, required=True)
    parser.add_argument('--pred-obb', type=parse_json_arg, required=True)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--prefix', default='obb')
    args = parser.parse_args()

    img = Image.open(args.image_path).convert('RGB')
    gt_obb = [[float(x), float(y)] for x, y in args.gt_obb]
    pred_obb = [[float(x), float(y)] for x, y in args.pred_obb]

    crop_box = compute_crop(img.size, union_bbox(poly_to_bbox(gt_obb), poly_to_bbox(pred_obb)))
    cl, ct, cr, cb = crop_box
    crop_np = np.asarray(img.crop(crop_box))

    gt_crop = shift_poly(gt_obb, cl, ct)
    pred_crop = shift_poly(pred_obb, cl, ct)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.prefix}_gt_pred.png')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(crop_np)
    draw_poly(ax, gt_crop, GT_COLOR, 'GT OBB', linestyle='-', label_pos='left_top')
    draw_poly(ax, pred_crop, PRED_COLOR, 'Pred OBB', linestyle='--', label_pos='top_right')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

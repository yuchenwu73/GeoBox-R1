#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw HBB or OBB predictions and optional ground truth on an image."""

import math
import os
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

CYAN = (0, 184, 217)
AMBER = (240, 138, 8)
GT_GREEN = (22, 178, 90)
WHITE = (255, 255, 255)
HALO = (12, 18, 26)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _clamp_point(x, y, w, h):
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def _dashed_line(draw, p1, p2, color, width, dash=16, gap=10):
    """Draw one dashed line segment."""
    x1, y1 = p1
    x2, y2 = p2
    total = math.hypot(x2 - x1, y2 - y1)
    if total < 1:
        return
    dx, dy = (x2 - x1) / total, (y2 - y1) / total
    dist = 0.0
    while dist < total:
        seg = min(dash, total - dist)
        a = (x1 + dx * dist, y1 + dy * dist)
        b = (x1 + dx * (dist + seg), y1 + dy * (dist + seg))
        draw.line([a, b], fill=color, width=width)
        dist += dash + gap


def _dashed_polygon(draw, pts, color, width, dash=16, gap=10, halo=True):
    n = len(pts)
    # Satellite imagery often matches annotation colors, so add a dark contrast halo.
    if halo:
        for i in range(n):
            _dashed_line(draw, pts[i], pts[(i + 1) % n], HALO, width + 2, dash, gap)
    for i in range(n):
        _dashed_line(draw, pts[i], pts[(i + 1) % n], color, width, dash, gap)


def _corner_brackets(draw, box, color, width, arm):
    """Draw L-shaped markers at the four HBB corners."""
    x1, y1, x2, y2 = box
    segs = [
        [(x1, y1 + arm), (x1, y1), (x1 + arm, y1)],
        [(x2 - arm, y1), (x2, y1), (x2, y1 + arm)],
        [(x2, y2 - arm), (x2, y2), (x2 - arm, y2)],
        [(x1 + arm, y2), (x1, y2), (x1, y2 - arm)],
    ]
    for seg in segs:
        draw.line(seg, fill=color, width=width, joint="curve")


def _solid_outline(draw, box, color, width):
    """Draw a solid rectangle with a contrasting halo."""
    x1, y1, x2, y2 = box
    draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=HALO, width=width)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _solid_polygon(draw, pts, color, width):
    draw.line(list(pts) + [pts[0]], fill=HALO, width=width + 2, joint="curve")
    draw.line(list(pts) + [pts[0]], fill=color, width=width, joint="curve")


def _label(
    draw,
    x: float,
    y: float,
    text: str,
    bg: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
    W: int,
    H: int,
    place: str = "above",
    pad: int = 6,
):
    """Draw a label above or below an anchor and keep it within the image."""
    try:
        b = draw.textbbox((0, 0), text, font=font)
        tw, th = b[2] - b[0], b[3] - b[1]
        ox, oy = b[0], b[1]
    except Exception:
        tw, th, ox, oy = len(text) * 9, 16, 0, 0

    bw, bh = tw + pad * 2, th + pad * 2
    lx = x
    # Place ground truth above and predictions below to avoid overlapping labels.
    ly = (y - bh) if place == "above" else y
    lx = int(max(2, min(W - bw - 2, lx)))
    ly = int(max(2, min(H - bh - 2, ly)))
    rad = max(3, bh // 5)

    draw.rounded_rectangle([lx + 2, ly + 3, lx + bw + 2, ly + bh + 3],
                           radius=rad, fill=(0, 0, 0))
    draw.rounded_rectangle([lx, ly, lx + bw, ly + bh],
                           radius=rad, fill=bg, outline=WHITE, width=1)
    draw.text((lx + pad - ox, ly + pad - oy), text, fill=WHITE, font=font)
    return (lx, ly, lx + bw, ly + bh)


def draw_result(
    image_path: str,
    result: dict,
    gt_bbox: Optional[List[float]] = None,
    gt_poly: Optional[List[List[float]]] = None,
    iou: Optional[float] = None,
) -> Image.Image:
    """Return an image annotated with predictions and optional ground truth."""
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    lw = max(2, round(min(W, H) / 300))
    arm = max(12, round(min(W, H) / 22))
    font_size = min(42, max(14, round(min(W, H) / 52)))
    font = _load_font(font_size)
    pad = max(4, font.size // 4) if hasattr(font, "size") else 5

    task = result.get("task", "hbb")
    accent = CYAN if task == "hbb" else AMBER

    # Draw ground truth first so predictions remain prominent.
    if task == "hbb" and gt_bbox:
        x1, y1, x2, y2 = gt_bbox
        draw = ImageDraw.Draw(im)
        _dashed_polygon(draw, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                        GT_GREEN, max(2, lw - 1))
    elif task == "obb" and gt_poly:
        draw = ImageDraw.Draw(im)
        _dashed_polygon(draw, [(p[0], p[1]) for p in gt_poly],
                        GT_GREEN, max(2, lw - 1))

    if task == "hbb" and result.get("bbox_px"):
        x1, y1, x2, y2 = result["bbox_px"]
        x1, y1 = _clamp_point(x1, y1, W, H)
        x2, y2 = _clamp_point(x2, y2, W, H)
        od.rectangle([x1, y1, x2, y2], fill=(accent[0], accent[1], accent[2], 34))
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(im)
        _solid_outline(draw, (x1, y1, x2, y2), accent, lw)
        _corner_brackets(draw, (x1, y1, x2, y2), accent, lw + 2, arm)

        if gt_bbox:
            _label(draw, gt_bbox[0], gt_bbox[1], "GT", GT_GREEN, font, W, H, place="above", pad=pad)
        ptag = "Pred" if iou is None else f"Pred · IoU {iou:.2f}"
        _label(draw, x1, y2, ptag, accent, font, W, H, place="below", pad=pad)

    elif task == "obb" and result.get("poly_px"):
        pts = [(_clamp_point(p[0], p[1], W, H)) for p in result["poly_px"]]
        od.polygon(pts, fill=(accent[0], accent[1], accent[2], 34))
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(im)
        _solid_polygon(draw, pts, accent, lw)
        r = lw + 2
        for (px, py) in pts:
            draw.rectangle([px - r, py - r, px + r, py + r], outline=accent, width=max(2, lw - 1))

        if gt_poly:
            gx, gy = min(((p[0], p[1]) for p in gt_poly), key=lambda p: p[1])
            _label(draw, gx, gy, "GT", GT_GREEN, font, W, H, place="above", pad=pad)
        bx, by = max(pts, key=lambda p: p[1])
        ptag = "Pred" if iou is None else f"Pred · RIoU {iou:.2f}"
        _label(draw, bx, by, ptag, accent, font, W, H, place="below", pad=pad)

    return im


def make_placeholder(text: str = "AWAITING INPUT", size=(960, 600)) -> Image.Image:
    """Create the empty-state image used by the web interface."""
    W, H = size
    im = Image.new("RGB", (W, H), (248, 251, 255))
    draw = ImageDraw.Draw(im)
    grid = (229, 237, 246)
    for x in range(0, W, 48):
        draw.line([(x, 0), (x, H)], fill=grid, width=1)
    for y in range(0, H, 48):
        draw.line([(0, y), (W, y)], fill=grid, width=1)
    border = (209, 222, 235)
    draw.rectangle([1, 1, W - 2, H - 2], outline=border, width=2)

    teal = (14, 116, 144)
    cx, cy = W // 2, H // 2
    draw.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], outline=teal, width=2)
    draw.line([(cx - 25, cy), (cx + 25, cy)], fill=teal, width=2)
    draw.line([(cx, cy - 25), (cx, cy + 25)], fill=teal, width=2)

    font = _load_font(22)
    try:
        b = draw.textbbox((0, 0), text, font=font)
        tw = b[2] - b[0]
    except Exception:
        tw = len(text) * 11
    draw.text((cx - tw // 2, cy + 48), text, fill=(100, 116, 139), font=font)
    return im

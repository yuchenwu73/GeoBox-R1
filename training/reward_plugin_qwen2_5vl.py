"""Geometric rewards for Qwen2.5-VL OBB grounding, loaded by gdpo_qwen2_5vl.sh.

Same rewards as reward_plugin_qwen3vl.py (rotated IoU and the adaptive Wasserstein reward
tau / (tau + W2) with tau = C_TAU * sqrt(Tr(Sigma_g)), mixed 0.5 / 0.5), different
coordinate space. Qwen2.5-VL is trained with --norm_bbox none, so it emits pixel
coordinates of the smart-resized image it actually sees, not norm1000. This is the
``resized_absolute`` mode of evaluation/evaluate_obb.py: the resize is recomputed here
(``qwen_resized_size``, factor 28, pixel budget from IMAGE_MAX_TOKEN_NUM) and every
prediction is rescaled to original-image pixels (``map_resized_to_orig``) before it is
compared with the pixel-space GT. Scoring the two spaces against each other directly would
zero out most rotated IoUs.

Reward names carry a ``_qwen2_5`` suffix so both plugins can be loaded together.
"""
import os
import math
import re
import json
import numpy as np
from swift.rewards import ORM, orms
from typing import List, Optional

try:
    from shapely.geometry import Polygon
except ImportError:
    raise ImportError("shapely is required: pip install shapely")

C_TAU = 8  # dimensionless scale of the adaptive tau: tau = C_TAU * sqrt(Tr(Sigma_g))

# Qwen2.5-VL preprocessing alignment: patch_size (14) * spatial_merge_size (2).
# Qwen3-VL uses patch_size 16 and therefore factor 32 — this file is Qwen2.5-VL only.
QWEN_IMAGE_FACTOR = 28

# OBB format throughout this file: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


def qwen_resized_size(width: int, height: int) -> tuple:
    """Qwen-VL fetch_image smart_resize size under the current IMAGE_MAX_TOKEN_NUM.

    Returns (resized_width, resized_height), both multiples of QWEN_IMAGE_FACTOR. The
    pixel budget comes from IMAGE_MAX_TOKEN_NUM / IMAGE_MIN_TOKEN_NUM, so the training
    process must export the same values as rollout and evaluation (1024 / 4 here).
    """
    factor = QWEN_IMAGE_FACTOR
    min_pixels = int(os.environ.get("IMAGE_MIN_TOKEN_NUM", "4")) * factor * factor
    max_pixels = int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1024")) * factor * factor
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = int(math.floor(height / beta / factor)) * factor
        w_bar = int(math.floor(width / beta / factor)) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil(height * beta / factor)) * factor
        w_bar = int(math.ceil(width * beta / factor)) * factor
    return w_bar, h_bar


def map_resized_to_orig(obb: List[List[float]], image_width, image_height) -> List[List[float]]:
    """Map an OBB from resized-image pixels back to original-image pixels, where it can
    be compared against the GT directly."""
    rw, rh = qwen_resized_size(int(image_width), int(image_height))
    if rw <= 0 or rh <= 0:
        return obb
    return [[p[0] * image_width / rw, p[1] * image_height / rh] for p in obb]


def _order_points_clockwise(points: List[List[float]]) -> List[List[float]]:
    """Sort polygon corners clockwise around their centroid."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: (math.atan2(p[1] - cy, p[0] - cx)))


def _extract_json_candidates(completion: str) -> List[object]:
    """Collect every JSON array in the completion.

    Fenced ```json blocks take priority; bracket matching is the fallback. Both paths
    return a list so that CoT outputs carrying two JSON blocks stay parseable.
    """
    text = completion.strip()
    candidates = []

    fenced_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text, flags=re.IGNORECASE)
    for block in fenced_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            candidates.append(json.loads(block))
        except Exception:
            continue

    if candidates:
        return candidates

    n = len(text)
    for start in range(n):
        if text[start] != '[':
            continue
        depth = 0
        for end in range(start, n):
            ch = text[end]
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    snippet = text[start:end + 1]
                    try:
                        candidates.append(json.loads(snippet))
                    except Exception:
                        pass
                    break

    return candidates


def parse_obb_from_completion(completion: str) -> Optional[List[List[float]]]:
    """Read the OBB out of `[{"oriented_bbox": [[x1,y1], ...]}]`, or None on failure.

    The raw numbers are returned unchanged, i.e. resized-image pixels for Qwen2.5-VL;
    call map_resized_to_orig before comparing against the GT.
    """
    try:
        candidates = _extract_json_candidates(completion)
        for data in reversed(candidates):
            if not (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict)):
                continue

            item = data[0]
            if "oriented_bbox" not in item:
                continue

            poly = item["oriented_bbox"]
            if len(poly) == 4 and isinstance(poly[0], (list, tuple)):
                return [[float(p[0]), float(p[1])] for p in poly]

        return None
    except Exception:
        return None


def calculate_rotated_iou(poly1: List[List[float]], poly2: List[List[float]]) -> float:
    """Rotated IoU of two quadrilaterals living in the same coordinate space."""
    try:
        if len(poly1) != 4 or len(poly2) != 4:
            return 0.0

        corners1 = [(p[0], p[1]) for p in poly1]
        corners2 = [(p[0], p[1]) for p in poly2]

        # Reorder corners so a shuffled point order cannot produce a self-intersecting polygon
        corners1 = _order_points_clockwise([[p[0], p[1]] for p in corners1])
        corners2 = _order_points_clockwise([[p[0], p[1]] for p in corners2])

        polygon1 = Polygon([(p[0], p[1]) for p in corners1])
        polygon2 = Polygon([(p[0], p[1]) for p in corners2])

        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0

        inter_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    except Exception:
        return 0.0


def poly_to_gaussian(poly: List[List[float]]) -> tuple:
    """MLE fit of a 2D Gaussian to the four corners:

        mu    = (1/N) * sum(x_i)
        Sigma = (1/N) * sum((x_i - mu)(x_i - mu)^T)

    Returns (mu (2,), Sigma (2,2)), or (None, None) on failure.
    """
    try:
        if len(poly) != 4:
            return None, None

        points = np.array(poly, dtype=np.float64)

        mu = np.mean(points, axis=0)
        centered = points - mu
        Sigma = np.dot(centered.T, centered) / len(points)

        # Ridge term keeps Sigma invertible for degenerate (collinear) boxes
        Sigma = Sigma + np.eye(2) * 1e-6

        return mu, Sigma
    except Exception:
        return None, None


def calculate_wd(mu_g: np.ndarray, Sigma_g: np.ndarray,
                 mu_p: np.ndarray, Sigma_p: np.ndarray) -> float:
    """Squared 2-Wasserstein distance between two 2D Gaussians:

        D_W = ||mu_p - mu_g||^2 + Tr(Sigma_p) + Tr(Sigma_g)
              - 2 * Tr((Sigma_p^{1/2} Sigma_g Sigma_p^{1/2})^{1/2})

    Returns a non-negative W2^2, or inf on failure.
    """
    try:
        mu_diff = mu_p - mu_g
        center_term = float(np.dot(mu_diff, mu_diff))

        trace_sum = np.trace(Sigma_p) + np.trace(Sigma_g)

        # The Cholesky factor stands in for Sigma_p^{1/2}: M below is not the symmetric
        # product but shares its eigenvalues, so Tr(M^{1/2}) is unchanged.
        L_p = np.linalg.cholesky(Sigma_p)
        M = L_p.T @ Sigma_g @ L_p

        eigenvalues = np.linalg.eigvalsh(M)
        eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp round-off noise below zero
        cross_term = 2.0 * np.sum(np.sqrt(eigenvalues))

        dw = center_term + trace_sum - cross_term

        return max(0.0, float(dw))

    except Exception:
        return float('inf')


def adaptive_tau(gt_obb: List[List[float]]) -> float:
    """Size-adaptive tau = C_TAU * sqrt(Tr(Sigma_g)), a distance like W2 itself.

    Tr(Sigma_g) = (w_g^2 + h_g^2) / 4, so the reward keeps a comparable slope on small
    and large targets. Here the GT arrives in original-image pixels.
    """
    _, Sigma_g = poly_to_gaussian(gt_obb)
    if Sigma_g is None:
        return C_TAU  # fall back to the global default
    tr_sigma = float(np.trace(Sigma_g))
    return C_TAU * math.sqrt(tr_sigma)


def calculate_r_wd(poly_gt: List[List[float]], poly_pred: List[List[float]],
                   tau: float = C_TAU) -> float:
    """R_WD = tau / (tau + W2), with W2 = sqrt(D_W). Both boxes must share a space,
    original-image pixels in this plugin."""
    mu_g, Sigma_g = poly_to_gaussian(poly_gt)
    mu_p, Sigma_p = poly_to_gaussian(poly_pred)

    if mu_g is None or mu_p is None:
        return 0.0

    dw = calculate_wd(mu_g, Sigma_g, mu_p, Sigma_p)

    if dw == float('inf'):
        return 0.0

    w2 = math.sqrt(dw)
    return tau / (tau + w2)


# Reward classes. ms-swift passes dataset columns as kwargs, so every __call__ takes:
#   completions                 model outputs, OBB in resized-image pixels
#   oriented_bbox               GT OBB in original-image pixels
#   image_width / image_height  original image size, used to map the prediction back
# and returns one reward in [0, 1] per completion.

class VG_IoU_Qwen25_ORM(ORM):
    """Rotated-IoU reward."""

    def __call__(self, completions, oriented_bbox, image_width, image_height, **kwargs) -> List[float]:
        rewards = []

        for completion, gt_obb, width, height in zip(completions, oriented_bbox, image_width, image_height):

            if gt_obb is None or len(gt_obb) != 4:
                rewards.append(0.0)
                continue

            pred_obb_resized = parse_obb_from_completion(completion)

            if pred_obb_resized is None:
                rewards.append(0.0)  # unparseable output earns nothing
                continue

            # Bring the prediction into the GT's coordinate space before scoring
            pred_obb_orig = map_resized_to_orig(pred_obb_resized, width, height)

            r_iou = calculate_rotated_iou(pred_obb_orig, gt_obb)

            rewards.append(r_iou)

        return rewards


class VG_WD_Qwen25_ORM(ORM):
    """Wasserstein reward with a fixed tau (control; prefer the adaptive variant)."""

    def __call__(self, completions, oriented_bbox, image_width, image_height, **kwargs) -> List[float]:
        rewards = []

        for completion, gt_obb, width, height in zip(completions, oriented_bbox, image_width, image_height):

            if gt_obb is None or len(gt_obb) != 4:
                rewards.append(0.0)
                continue

            pred_obb_resized = parse_obb_from_completion(completion)

            if pred_obb_resized is None:
                rewards.append(0.0)
                continue

            pred_obb_orig = map_resized_to_orig(pred_obb_resized, width, height)

            r_wd = calculate_r_wd(gt_obb, pred_obb_orig)

            rewards.append(r_wd)

        return rewards


class VG_WD_Adaptive_Qwen25_ORM(ORM):
    """Wasserstein reward with tau scaled to the GT size (the reward used by GDPO)."""

    def __call__(self, completions, oriented_bbox, image_width, image_height, **kwargs) -> List[float]:
        rewards = []

        for completion, gt_obb, width, height in zip(completions, oriented_bbox, image_width, image_height):

            if gt_obb is None or len(gt_obb) != 4:
                rewards.append(0.0)
                continue

            pred_obb_resized = parse_obb_from_completion(completion)

            if pred_obb_resized is None:
                rewards.append(0.0)
                continue

            pred_obb_orig = map_resized_to_orig(pred_obb_resized, width, height)

            tau = adaptive_tau(gt_obb)

            r_wd = calculate_r_wd(gt_obb, pred_obb_orig, tau=tau)

            rewards.append(r_wd)

        return rewards


# The _qwen2_5 suffix keeps these distinct from the norm1000 rewards in reward_plugin_qwen3vl.py
orms['external_vg_iou_qwen2_5'] = VG_IoU_Qwen25_ORM
orms['external_vg_wd_qwen2_5'] = VG_WD_Qwen25_ORM
orms['external_vg_wd_adaptive_qwen2_5'] = VG_WD_Adaptive_Qwen25_ORM

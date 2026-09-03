"""Geometric rewards for Qwen3-VL OBB grounding, loaded by gdpo.sh / gdpo_fixed_tau.sh.

Reward design (both terms are in [0, 1] and are mixed 0.5 / 0.5 by --reward_weights):
  R_IoU  rotated IoU between the predicted and the ground-truth quadrilateral. Direct, but
         it saturates at 0 as soon as the boxes stop overlapping, and then gives no signal.
  R_WD   tau / (tau + W2). Each box is viewed as a 2D Gaussian whose mean is the box
         centre and whose covariance is the second moment of its four corners, so W2, the
         2-Wasserstein distance between the two Gaussians, keeps growing smoothly with
         centre offset, size mismatch and rotation error even when the boxes are disjoint.
         tau sets the distance at which the reward drops to 0.5; it scales with the target
         size, tau = C_TAU * sqrt(Tr(Sigma_g)), because Tr(Sigma_g) = (w^2 + h^2) / 4 is
         the squared half-diagonal of the GT box, so the same relative error earns the same
         reward on small and large targets (a fixed tau under-rewards large ones; that
         variant is kept as VG_WD_ORM for the ablation).

Coordinate space: the policy emits norm1000 coordinates, so the pixel-space GT
(``oriented_bbox`` plus ``image_width`` / ``image_height`` from the dataset) is
normalised into the same 0-1000 space before scoring. Parsing is strict: only the last
``[{"oriented_bbox": [[x, y] x 4]}]`` block in a completion counts; anything else earns 0.
"""
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

# OBB format throughout this file: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


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

    Candidates are scanned in reverse so that a CoT trace followed by the final answer
    resolves to the answer.
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


def norm_obb(obb: List[List[float]], image_width, image_height) -> List[List[float]]:
    """Map a pixel-space OBB into the 0-1000 normalised space."""
    return [[p[0] * 1000.0 / image_width, p[1] * 1000.0 / image_height] for p in obb]


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


def adaptive_tau(gt_obb_norm: List[List[float]]) -> float:
    """Size-adaptive tau = C_TAU * sqrt(Tr(Sigma_g)), a distance like W2 itself.

    Tr(Sigma_g) = (w_g^2 + h_g^2) / 4, so the reward keeps a comparable slope on small
    and large targets instead of saturating on the large ones.
    """
    _, Sigma_g = poly_to_gaussian(gt_obb_norm)
    if Sigma_g is None:
        return C_TAU  # fall back to the global default
    tr_sigma = float(np.trace(Sigma_g))
    return C_TAU * math.sqrt(tr_sigma)


def calculate_r_wd(poly_gt: List[List[float]], poly_pred: List[List[float]],
                   tau: float = C_TAU) -> float:
    """R_WD = tau / (tau + W2), with W2 = sqrt(D_W). Both boxes must share a space.

    Mapping W2 rather than W2^2 doubles the reward resolution at small errors, which is
    where the policy spends most of training.
    """
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
#   completions                 model outputs, OBB already in norm1000
#   oriented_bbox               GT OBB in original-image pixels
#   image_width / image_height  original image size, used to normalise the GT
# and returns one reward in [0, 1] per completion.

class VG_IoU_ORM(ORM):
    """Rotated-IoU reward."""

    def __call__(self, completions, oriented_bbox, image_width, image_height, **kwargs) -> List[float]:

        rewards = []

        for completion, gt_obb, width, height in zip(completions, oriented_bbox, image_width, image_height):

            if gt_obb is None or len(gt_obb) != 4:
                rewards.append(0.0)
                continue

            pred_obb_norm = parse_obb_from_completion(completion)

            if pred_obb_norm is None:
                rewards.append(0.0)  # unparseable output earns nothing
                continue

            gt_obb_norm = norm_obb(gt_obb, width, height)

            r_iou = calculate_rotated_iou(pred_obb_norm, gt_obb_norm)

            rewards.append(r_iou)

        return rewards


class VG_WD_ORM(ORM):
    """Wasserstein reward with a fixed tau (the ablation control, see gdpo_fixed_tau.sh)."""

    def __call__(self, completions, oriented_bbox, image_width, image_height, **kwargs) -> List[float]:

        rewards = []

        for completion, gt_obb, width, height in zip(completions, oriented_bbox, image_width, image_height):

            if gt_obb is None or len(gt_obb) != 4:
                rewards.append(0.0)
                continue

            pred_obb_norm = parse_obb_from_completion(completion)

            if pred_obb_norm is None:
                rewards.append(0.0)
                continue

            gt_obb_norm = norm_obb(gt_obb, width, height)

            r_wd = calculate_r_wd(gt_obb_norm, pred_obb_norm)

            rewards.append(r_wd)

        return rewards


class VG_WD_Adaptive_ORM(ORM):
    """Wasserstein reward with tau scaled to the GT size (the reward used by GDPO)."""

    def __call__(self, completions, oriented_bbox, image_width, image_height, **kwargs) -> List[float]:
        rewards = []

        for completion, gt_obb, width, height in zip(completions, oriented_bbox, image_width, image_height):

            if gt_obb is None or len(gt_obb) != 4:
                rewards.append(0.0)
                continue

            pred_obb_norm = parse_obb_from_completion(completion)

            if pred_obb_norm is None:
                rewards.append(0.0)
                continue

            gt_obb_norm = norm_obb(gt_obb, width, height)

            tau = adaptive_tau(gt_obb_norm)

            r_wd = calculate_r_wd(gt_obb_norm, pred_obb_norm, tau=tau)

            rewards.append(r_wd)

        return rewards


orms['external_vg_iou'] = VG_IoU_ORM
orms['external_vg_wd'] = VG_WD_ORM
orms['external_vg_wd_adaptive'] = VG_WD_Adaptive_ORM

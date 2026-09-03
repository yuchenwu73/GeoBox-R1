"""Sweep C_TAU for the adaptive Wasserstein reward on the RL OBB subset.

The GDPO reward R_WD = tau / (tau + W2) uses tau = C_TAU * sqrt(Tr(Sigma_g)),
where Sigma_g is the covariance of the ground-truth box's corners. This script
is the analysis behind C_TAU = 8 in training/reward_plugin_qwen3vl.py; the
reward functions below are copies of the training ones (a unit test keeps them
in sync).

Reads data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl and prints eight sections:
  1-3  size statistics of the GT boxes in norm1000 space: overall, per source
       dataset, and bucketed by sqrt(Tr(Sigma_g));
  4    synthetic squares of several sizes, each shifted by exactly one width:
       a good C_TAU gives the same R_WD regardless of size (small std);
  5    R_WD against shift distance for several C_TAU values and for a fixed tau;
  6    the distribution of the resulting adaptive tau over the dataset;
  7-8  on 2000 real boxes, R_WD at shifts of 0.1w ... 5w; the final score
       discrimination / (1 + 10 * std) rewards a large gap between near and far
       misses and a small spread across targets at the same relative shift.

Run from anywhere: python visualization/analyze_tau_adaptive.py
"""

import json
import math
import numpy as np
from collections import defaultdict
from pathlib import Path


def norm_obb(obb, image_width, image_height):
    """Pixel corners -> norm1000 corners, so target scale is comparable across image sizes."""
    return [[p[0] * 1000.0 / image_width, p[1] * 1000.0 / image_height] for p in obb]


def poly_to_gaussian(poly):
    """Mean and covariance of the four corners; copy of the training reward's version."""
    points = np.asarray(poly, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"Expected four finite 2D corners, got shape {points.shape}")
    mu = np.mean(points, axis=0)
    centered = points - mu
    Sigma = np.dot(centered.T, centered) / len(points)
    # A small ridge keeps Cholesky stable for extremely thin but valid boxes.
    Sigma = Sigma + np.eye(2) * 1e-6
    return mu, Sigma


def calculate_wd(mu_g, Sigma_g, mu_p, Sigma_p):
    """Squared 2-Wasserstein distance between two 2D Gaussians; copy of the training reward's version."""
    mu_diff = mu_p - mu_g
    center_term = float(np.dot(mu_diff, mu_diff))
    trace_sum = np.trace(Sigma_p) + np.trace(Sigma_g)
    L_p = np.linalg.cholesky(Sigma_p)
    M = L_p.T @ Sigma_g @ L_p
    eigenvalues = np.linalg.eigvalsh(M)
    # Roundoff can produce tiny negative eigenvalues for a PSD matrix.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    cross_term = 2.0 * np.sum(np.sqrt(eigenvalues))
    dw = center_term + trace_sum - cross_term
    return max(0.0, float(dw))


def calculate_r_wd(poly_gt, poly_pred, tau):
    """R_WD = tau / (tau + W2), the bounded Wasserstein reward used by GDPO."""
    mu_g, Sigma_g = poly_to_gaussian(poly_gt)
    mu_p, Sigma_p = poly_to_gaussian(poly_pred)
    dw = calculate_wd(mu_g, Sigma_g, mu_p, Sigma_p)
    w2 = math.sqrt(dw)
    return tau / (tau + w2)


def obb_dimensions(obb_norm):
    """Width and height of an OBB in norm1000 space, from the edges between adjacent vertices."""
    pts = np.array(obb_norm)
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append(np.linalg.norm(pts[j] - pts[i]))
    e1 = (edges[0] + edges[2]) / 2
    e2 = (edges[1] + edges[3]) / 2
    w, h = min(e1, e2), max(e1, e2)
    return w, h


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "GeoBox-R1-Data" / "rl" / "rl_obb_20pct.jsonl"
print(f"Loading data: {DATA_PATH}")

# Work in norm1000 space so target scale is comparable across source resolutions.
records = []
with DATA_PATH.open('r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line.strip())
        records.append(rec)

print(f"Total samples: {len(records)}")

print("\n" + "=" * 80)
print("1. GT OBB scale distribution (norm1000 space)")
print("=" * 80)

widths = []
heights = []
areas = []
aspect_ratios = []
tr_sigmas = []
sqrt_tr_sigmas = []
dataset_stats = defaultdict(list)
valid_records = []
degenerate_by_dataset = defaultdict(int)

for rec in records:
    gt_obb = rec['oriented_bbox']
    img_w = rec['image_width']
    img_h = rec['image_height']
    dataset = rec.get('origin_dataset', 'unknown')

    gt_norm = norm_obb(gt_obb, img_w, img_h)

    w, h = obb_dimensions(gt_norm)
    if w < 1e-3:
        degenerate_by_dataset[dataset] += 1
        continue
    valid_records.append(rec)
    widths.append(w)
    heights.append(h)
    areas.append(w * h)
    aspect_ratios.append(h / w if w > 0 else 0)

    _, Sigma_g = poly_to_gaussian(gt_norm)
    tr = float(np.trace(Sigma_g))
    tr_sigmas.append(tr)
    sqrt_tr_sigmas.append(math.sqrt(tr))
    dataset_stats[dataset].append({
        'w': w, 'h': h, 'area': w * h,
        'tr_sigma': tr, 'sqrt_tr_sigma': math.sqrt(tr)
    })

skipped_degenerate = sum(degenerate_by_dataset.values())
skip_details = ", ".join(
    f"{name}={count}" for name, count in sorted(degenerate_by_dataset.items())
) or "none"
print(f"Skipped {skipped_degenerate} degenerate OBB(s): {skip_details}")
if not valid_records:
    raise RuntimeError("No valid OBB samples remain after filtering degenerate boxes")

widths = np.array(widths)
heights = np.array(heights)
areas = np.array(areas)
aspect_ratios = np.array(aspect_ratios)
tr_sigmas = np.array(tr_sigmas)
sqrt_tr_sigmas = np.array(sqrt_tr_sigmas)

def print_stats(name, arr):
    """One line of distribution statistics."""
    print(f"  {name:20s}: min={arr.min():8.2f}  p5={np.percentile(arr,5):8.2f}  "
          f"p25={np.percentile(arr,25):8.2f}  median={np.median(arr):8.2f}  "
          f"p75={np.percentile(arr,75):8.2f}  p95={np.percentile(arr,95):8.2f}  "
          f"max={arr.max():8.2f}  mean={arr.mean():8.2f}  std={arr.std():8.2f}")

print_stats("width (short edge)", widths)
print_stats("height (long edge)", heights)
print_stats("area (w*h)", areas)
print_stats("aspect ratio (h/w)", aspect_ratios)
print_stats("Tr(Σ_g)", tr_sigmas)
print_stats("√Tr(Σ_g)", sqrt_tr_sigmas)

print("\n" + "=" * 80)
print("2. Per-dataset statistics")
print("=" * 80)

for ds_name in sorted(dataset_stats.keys()):
    items = dataset_stats[ds_name]
    n = len(items)
    ws = np.array([x['w'] for x in items])
    hs = np.array([x['h'] for x in items])
    as_ = np.array([x['area'] for x in items])
    sqs = np.array([x['sqrt_tr_sigma'] for x in items])
    print(f"\n  [{ds_name}] (n={n})")
    print(f"    width:      median={np.median(ws):7.2f}  mean={ws.mean():7.2f}  min={ws.min():7.2f}  max={ws.max():7.2f}")
    print(f"    height:     median={np.median(hs):7.2f}  mean={hs.mean():7.2f}  min={hs.min():7.2f}  max={hs.max():7.2f}")
    print(f"    area:       median={np.median(as_):9.2f}  mean={as_.mean():9.2f}  min={as_.min():9.2f}  max={as_.max():9.2f}")
    print(f"    √Tr(Σ_g):  median={np.median(sqs):7.2f}  mean={sqs.mean():7.2f}  min={sqs.min():7.2f}  max={sqs.max():7.2f}")

print("\n" + "=" * 80)
print("3. Target-scale buckets (bucketed by √Tr(Σ_g))")
print("=" * 80)

bucket_edges = [0, 5, 10, 20, 50, 100, 200, 500, float('inf')]
bucket_names = ['<5', '5-10', '10-20', '20-50', '50-100', '100-200', '200-500', '>500']

for i, name in enumerate(bucket_names):
    lo, hi = bucket_edges[i], bucket_edges[i + 1]
    mask = (sqrt_tr_sigmas >= lo) & (sqrt_tr_sigmas < hi)
    count = mask.sum()
    pct = 100.0 * count / len(sqrt_tr_sigmas)
    if count > 0:
        subset = sqrt_tr_sigmas[mask]
        print(f"  {name:>8s}: n={count:6d} ({pct:5.1f}%)  "
              f"√Tr(Σ_g) median={np.median(subset):7.2f}  mean={subset.mean():7.2f}")
    else:
        print(f"  {name:>8s}: n={count:6d} ({pct:5.1f}%)")

print("\n" + "=" * 80)
print("4. C_TAU calibration")
print("=" * 80)
print("Goal: find a C_TAU that keeps R_WD equally discriminative across target scales")
print("Calibration condition: when the prediction is shifted by one target width, R_WD should hit a target value\n")

# Section 4: a square of side s centred at (500, 500), shifted right by exactly s.
# Each C_TAU row lists R_WD per size; the ideal row is flat (std close to 0).
test_c_tau_values = [0.5, 1, 2, 3, 5, 8, 10, 13, 15, 20]

test_sizes = [10, 30, 50, 100, 200, 500]

print(f"{'C_TAU':>6s}", end="")
for s in test_sizes:
    print(f"  size={s:>3d}", end="")
print("  | consistency(std)")
print("-" * 100)

best_c_tau = None
best_consistency = float('inf')

for c_tau in test_c_tau_values:
    r_values = []
    print(f"{c_tau:6.1f}", end="")

    for s in test_sizes:
        half = s / 2
        gt_obb = [
            [500 - half, 500 - half],
            [500 + half, 500 - half],
            [500 + half, 500 + half],
            [500 - half, 500 + half]
        ]
        pred_obb = [
            [500 - half + s, 500 - half],
            [500 + half + s, 500 - half],
            [500 + half + s, 500 + half],
            [500 - half + s, 500 + half]
        ]

        _, Sigma_g = poly_to_gaussian(gt_obb)
        tr_sg = float(np.trace(Sigma_g))
        tau = c_tau * math.sqrt(tr_sg)

        r = calculate_r_wd(gt_obb, pred_obb, tau=tau)
        r_values.append(r)
        print(f"  {r:8.4f}", end="")

    std = np.std(r_values)
    mean_r = np.mean(r_values)
    print(f"  | {std:.4f} (mean={mean_r:.4f})")

    if std < best_consistency:
        best_consistency = std
        best_c_tau = c_tau

print(f"\nMost consistent C_TAU = {best_c_tau} (std = {best_consistency:.4f})")

print("\n" + "=" * 80)
print("5. Shift ratio vs R_WD curve (several C_TAU values)")
print("=" * 80)
print("shift = d * (target edge length); several values of d are tested\n")

# Section 5: reward against shift distance on a 50x50 target; tau=5000 is the
# fixed-scale reference that barely reacts to shifts at this size.
selected_c_taus = [1, 3, 5, 8, 13]
ref_size = 50
d_ratios = [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

print(f"Reference target: {ref_size}x{ref_size} square (norm1000 space)")
print(f"{'d/w':>6s}", end="")
for ct in selected_c_taus:
    print(f"  C={ct:>2d}", end="")
print("  | fixed_tau=5000")
print("-" * 80)

for d_ratio in d_ratios:
    half = ref_size / 2
    gt_obb = [
        [500 - half, 500 - half],
        [500 + half, 500 - half],
        [500 + half, 500 + half],
        [500 - half, 500 + half]
    ]
    d = d_ratio * ref_size
    pred_obb = [
        [500 - half + d, 500 - half],
        [500 + half + d, 500 - half],
        [500 + half + d, 500 + half],
        [500 - half + d, 500 + half]
    ]

    print(f"{d_ratio:6.1f}", end="")
    for ct in selected_c_taus:
        _, Sigma_g = poly_to_gaussian(gt_obb)
        tr_sg = float(np.trace(Sigma_g))
        tau = ct * math.sqrt(tr_sg)
        r = calculate_r_wd(gt_obb, pred_obb, tau=tau)
        print(f"  {r:5.3f}", end="")

    r_fixed = calculate_r_wd(gt_obb, pred_obb, tau=5000)
    print(f"  | {r_fixed:5.3f}")

print("\n" + "=" * 80)
print("6. Whole-dataset simulation: tau distribution under different C_TAU")
print("=" * 80)

for c_tau_test in [1, 3, 5, best_c_tau]:
    taus = []
    for sq in sqrt_tr_sigmas:
        taus.append(c_tau_test * sq)
    taus = np.array(taus)
    print(f"\n  C_TAU={c_tau_test}:")
    print_stats("    adaptive tau", taus)

print("\n" + "=" * 80)
print("7. Discriminability on real data")
print("=" * 80)
print("For every GT, simulate 5 shifts (0.1w, 0.5w, 1w, 2w, 5w) and compare the discriminability of each C_TAU\n")

# The deterministic subset keeps calibration runs reproducible without scanning every sample.
np.random.seed(42)
sample_indices = np.random.choice(len(valid_records), size=min(2000, len(valid_records)), replace=False)

# Section 7: the same experiment on real GT boxes; the shift is a multiple of each
# box's own width, so all targets are compared at the same relative error.
shift_ratios = [0.1, 0.5, 1.0, 2.0, 5.0]
c_tau_candidates = [1, 3, 5, 8, 13]

results_by_ctau = {ct: {sr: [] for sr in shift_ratios} for ct in c_tau_candidates}
results_fixed = {sr: [] for sr in shift_ratios}

for idx in sample_indices:
    rec = valid_records[idx]
    gt_obb = rec['oriented_bbox']
    img_w = rec['image_width']
    img_h = rec['image_height']
    gt_norm = norm_obb(gt_obb, img_w, img_h)

    w, h = obb_dimensions(gt_norm)
    _, Sigma_g = poly_to_gaussian(gt_norm)
    tr_sg = float(np.trace(Sigma_g))
    sq_tr = math.sqrt(tr_sg)

    for sr in shift_ratios:
        d = sr * w  # shift, in multiples of the width
        pred_norm = [[p[0] + d, p[1]] for p in gt_norm]

        r_fixed = calculate_r_wd(gt_norm, pred_norm, tau=5000)
        results_fixed[sr].append(r_fixed)

        for ct in c_tau_candidates:
            tau = ct * sq_tr
            r = calculate_r_wd(gt_norm, pred_norm, tau=tau)
            results_by_ctau[ct][sr].append(r)

print(f"{'shift':>6s}", end="")
for ct in c_tau_candidates:
    print(f"  C={ct:>2d}(mean±std)    ", end="")
print("  | fixed(mean±std)")
print("-" * 130)

for sr in shift_ratios:
    print(f"  {sr:4.1f}w", end="")
    for ct in c_tau_candidates:
        arr = np.array(results_by_ctau[ct][sr])
        print(f"  {arr.mean():.3f}±{arr.std():.3f}       ", end="")
    arr_f = np.array(results_fixed[sr])
    print(f"  | {arr_f.mean():.3f}±{arr_f.std():.3f}")

print("\nR_WD gap between adjacent shifts (discriminability):")
print(f"{'comparison':>12s}", end="")
for ct in c_tau_candidates:
    print(f"  C={ct:>2d}", end="")
print("  | fixed")
print("-" * 80)

for i in range(len(shift_ratios) - 1):
    sr1, sr2 = shift_ratios[i], shift_ratios[i + 1]
    print(f"  {sr1:.1f}w→{sr2:.1f}w", end="")
    for ct in c_tau_candidates:
        arr1 = np.array(results_by_ctau[ct][sr1])
        arr2 = np.array(results_by_ctau[ct][sr2])
        diff = (arr1 - arr2).mean()
        print(f"  {diff:5.3f}", end="")
    arr_f1 = np.array(results_fixed[sr1])
    arr_f2 = np.array(results_fixed[sr2])
    diff_f = (arr_f1 - arr_f2).mean()
    print(f"  | {diff_f:5.3f}")

print("\n" + "=" * 80)
print("8. Final recommendation")
print("=" * 80)

print("\nOverall assessment - cross-scale consistency (smaller std is better) + discriminability (larger gap is better):")
print(f"{'C_TAU':>6s}  {'std@1w':>10s}  {'discrim':>8s}  {'score':>8s}")
print("-" * 50)

# Section 8: consistency = std of R_WD across targets at a 1w shift (lower is better);
# discrimination = mean R_WD gap between a 0.1w and a 5w shift (higher is better).
for ct in c_tau_candidates:
    arr_1w = np.array(results_by_ctau[ct][1.0])
    consistency = arr_1w.std()

    arr_01 = np.array(results_by_ctau[ct][0.1])
    arr_5 = np.array(results_by_ctau[ct][5.0])
    discrimination = (arr_01 - arr_5).mean()

    # Prefer separation between shifts while penalizing cross-scale variance.
    score = discrimination / (1 + consistency * 10)

    print(f"{ct:6d}  {consistency:10.4f}  {discrimination:8.4f}  {score:8.4f}")

arr_01_f = np.array(results_fixed[0.1])
arr_5_f = np.array(results_fixed[5.0])
arr_1w_f = np.array(results_fixed[1.0])
consistency_f = arr_1w_f.std()
discrimination_f = (arr_01_f - arr_5_f).mean()
score_f = discrimination_f / (1 + consistency_f * 10)
print(f"{'fixed':>6s}  {consistency_f:10.4f}  {discrimination_f:8.4f}  {score_f:8.4f}  (tau=5000)")

print("\nNotes:")
print("- C_TAU too small -> tau too small -> R_WD separates near misses well but saturates to 0 far away (loses long-range ordering)")
print("- C_TAU too large -> tau too large -> R_WD separates far misses well but saturates to 1 nearby (loses short-range ordering)")
print("- The right value balances the two")

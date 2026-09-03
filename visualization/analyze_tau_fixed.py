"""Show that one fixed tau cannot serve targets of different sizes (the fixed-tau control).

Companion of analyze_tau_adaptive.py using the same reward functions and the same
RL subset. Boxes are split into small / medium / large groups at the 33rd and 66th
percentiles of sqrt(Tr(Sigma_g)). For every fixed tau in the sweep, and for the
adaptive tau with C_TAU_REF = 8 (the training value), the script prints the mean
R_WD of each group at a one-width shift, the spread across groups (crossStd), the
gap between a 0.2w and a 2.0w shift (discrim) and the composite score
discrim / (1 + 10 * crossStd). A fixed tau tuned to small targets saturates on
large ones and vice versa; the closing distribution table shows the adaptive tau
keeping the spread small.

The script raises when a scale group is empty, because the cross-scale comparison
is meaningless without all three sizes.

Run from anywhere: python visualization/analyze_tau_fixed.py
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
    """(short edge, long edge) of an OBB, averaging opposite edges."""
    pts = np.array(obb_norm)
    edges = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
    e1 = (edges[0] + edges[2]) / 2
    e2 = (edges[1] + edges[3]) / 2
    return min(e1, e2), max(e1, e2)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "GeoBox-R1-Data" / "rl" / "rl_obb_20pct.jsonl"
print(f"Loading data: {DATA_PATH}")

# Normalize all samples before grouping so image resolution cannot drive the comparison.
records = []
with DATA_PATH.open('r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line.strip()))
print(f"Total samples: {len(records)}")

samples = []
degenerate_by_dataset = defaultdict(int)
for rec in records:
    gt_obb = rec['oriented_bbox']
    gt_norm = norm_obb(gt_obb, rec['image_width'], rec['image_height'])
    w, h = obb_dimensions(gt_norm)
    _, Sigma_g = poly_to_gaussian(gt_norm)
    if w < 1e-3:
        degenerate_by_dataset[rec.get('origin_dataset', 'unknown')] += 1
        continue
    tr_sg = float(np.trace(Sigma_g))
    samples.append({
        'gt_norm': gt_norm, 'w': w, 'h': h,
        'sqrt_tr': math.sqrt(tr_sg)
    })

skipped_degenerate = sum(degenerate_by_dataset.values())
skip_details = ", ".join(
    f"{name}={count}" for name, count in sorted(degenerate_by_dataset.items())
) or "none"
print(f"Skipped {skipped_degenerate} degenerate OBB(s): {skip_details}")
if not samples:
    raise RuntimeError("No valid OBB samples remain after filtering degenerate boxes")
print(f"Valid samples: {len(samples)}")

sqrt_trs = np.array([s['sqrt_tr'] for s in samples])
print(f"√Tr(Σ_g) range: [{sqrt_trs.min():.2f}, {sqrt_trs.max():.2f}], "
      f"median={np.median(sqrt_trs):.2f}, p5={np.percentile(sqrt_trs,5):.2f}, "
      f"p95={np.percentile(sqrt_trs,95):.2f}")

# Quantile groups keep the scale comparison balanced despite a skewed size distribution.
p33 = np.percentile(sqrt_trs, 33)
p66 = np.percentile(sqrt_trs, 66)
groups = {'small': [], 'medium': [], 'large': []}
for s in samples:
    if s['sqrt_tr'] <= p33:
        groups['small'].append(s)
    elif s['sqrt_tr'] <= p66:
        groups['medium'].append(s)
    else:
        groups['large'].append(s)

print("\nScale groups (split at the p33/p66 of √Tr(Σ_g)):")
empty_groups = []
for name, grp in groups.items():
    sqs = [s['sqrt_tr'] for s in grp]
    if not sqs:
        empty_groups.append(name)
        print(f"  {name}: n=0")
        continue
    print(f"  {name}: n={len(grp)}, √Tr(Σ_g) ∈ [{min(sqs):.2f}, {max(sqs):.2f}], median={np.median(sqs):.2f}")

# Cross-scale comparisons are meaningful only when every quantile group is represented.
if empty_groups:
    raise RuntimeError(
        "Fixed-tau analysis requires three non-empty scale groups; "
        f"empty groups: {', '.join(empty_groups)}"
    )

print("\n" + "=" * 80)
print("Sweeping fixed tau: cross-scale consistency + discriminability")
print("=" * 80)

# Use a fixed subsample so repeated sweeps are fast and directly comparable.
np.random.seed(42)
sample_per_group = 500
eval_groups = {}
for name, grp in groups.items():
    n = min(sample_per_group, len(grp))
    indices = np.random.choice(len(grp), size=n, replace=False)
    eval_groups[name] = [grp[i] for i in indices]

shift_ratios = [0.2, 0.5, 1.0, 2.0]  # shift, in multiples of the target width
fixed_taus = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
C_TAU_REF = 8  # Must match the reward used for training.

def eval_tau_for_group(group_samples, tau_val, adaptive=False, c_tau=C_TAU_REF):
    """Mean R_WD of one group of samples under each shift."""
    results = {sr: [] for sr in shift_ratios}
    for s in group_samples:
        gt = s['gt_norm']
        w = s['w']
        for sr in shift_ratios:
            d = sr * w
            pred = [[p[0] + d, p[1]] for p in gt]
            if adaptive:
                t = c_tau * s['sqrt_tr']
            else:
                t = tau_val
            r = calculate_r_wd(gt, pred, tau=t)
            results[sr].append(r)
    return {sr: np.mean(results[sr]) for sr in shift_ratios}

print("\nMean R_WD of each fixed tau over the 3 scale groups (shift = 1.0w):")
print(f"{'tau':>8s}  {'small':>8s}  {'medium':>8s}  {'large':>8s}  {'crossStd':>8s}  {'discrim':>8s}  {'score':>8s}")
print("-" * 75)

best_score = -1
best_tau = None

# Sweep: per fixed tau, mean R_WD of each scale group at a 1.0w shift. crossStd is
# the spread across the three groups (lower is better); discrim is R(0.2w) - R(2.0w)
# averaged over groups (higher is better).
for tau_val in fixed_taus:
    r_1w = []
    total_disc = []
    for name in ['small', 'medium', 'large']:
        res = eval_tau_for_group(eval_groups[name], tau_val, adaptive=False)
        r_1w.append(res[1.0])
        total_disc.append(res[0.2] - res[2.0])

    cross_std = np.std(r_1w)
    avg_disc = np.mean(total_disc)
    # Composite score: discriminability / (1 + cross-group std * 10)
    score = avg_disc / (1 + cross_std * 10)

    print(f"{tau_val:8d}  {r_1w[0]:8.4f}  {r_1w[1]:8.4f}  {r_1w[2]:8.4f}  "
          f"{cross_std:8.4f}  {avg_disc:8.4f}  {score:8.4f}")

    if score > best_score:
        best_score = score
        best_tau = tau_val

# The same measurement with the adaptive tau used in training.
r_1w_adapt = []
disc_adapt = []
for name in ['small', 'medium', 'large']:
    res = eval_tau_for_group(eval_groups[name], None, adaptive=True, c_tau=C_TAU_REF)
    r_1w_adapt.append(res[1.0])
    disc_adapt.append(res[0.2] - res[2.0])

cross_std_adapt = np.std(r_1w_adapt)
avg_disc_adapt = np.mean(disc_adapt)
score_adapt = avg_disc_adapt / (1 + cross_std_adapt * 10)

print(f"{'adapt':>8s}  {r_1w_adapt[0]:8.4f}  {r_1w_adapt[1]:8.4f}  {r_1w_adapt[2]:8.4f}  "
      f"{cross_std_adapt:8.4f}  {avg_disc_adapt:8.4f}  {score_adapt:8.4f}  (C_TAU={C_TAU_REF})")

print(f"\nBest fixed tau = {best_tau} (score = {best_score:.4f})")
print(f"Adaptive tau score = {score_adapt:.4f}")
if score_adapt > best_score:
    print(f"Adaptive tau beats the best fixed tau by {(score_adapt/best_score - 1)*100:.1f}%")

print("\n" + "=" * 80)
print(f"Detailed comparison: fixed tau={best_tau} vs adaptive tau (C_TAU={C_TAU_REF})")
print("=" * 80)

print(f"\n{'shift':>6s}", end="")
for name in ['small', 'medium', 'large']:
    print(f"  {name}(fixed/adaptive)", end="")
print()
print("-" * 90)

for sr in shift_ratios:
    print(f"  {sr:4.1f}w", end="")
    for name in ['small', 'medium', 'large']:
        res_fixed = eval_tau_for_group(eval_groups[name], best_tau, adaptive=False)
        res_adapt = eval_tau_for_group(eval_groups[name], None, adaptive=True, c_tau=C_TAU_REF)
        print(f"     {res_fixed[sr]:.3f} / {res_adapt[sr]:.3f}    ", end="")
    print()

print("\n" + "=" * 80)
print("Core problem of a fixed tau: the same relative shift yields different R_WD across target scales")
print("=" * 80)

print("\nR_WD distribution over all samples at shift = 1.0w:")
print(f"{'setting':>12s}  {'mean':>8s}  {'std':>8s}  {'p5':>8s}  {'p25':>8s}  {'median':>8s}  {'p75':>8s}  {'p95':>8s}")
print("-" * 85)

# Closing table: R_WD at a 1.0w shift over 3000 boxes; the std column is the
# cross-scale consistency of each setting.
all_eval = []
np.random.seed(42)
n_all = min(3000, len(samples))
all_idx = np.random.choice(len(samples), size=n_all, replace=False)
for i in all_idx:
    all_eval.append(samples[i])

for tau_label, tau_val in [(f"fixed tau={best_tau}", best_tau), ("fixed tau=5000", 5000)]:
    rwd_list = []
    for s in all_eval:
        d = 1.0 * s['w']
        pred = [[p[0] + d, p[1]] for p in s['gt_norm']]
        r = calculate_r_wd(s['gt_norm'], pred, tau=tau_val)
        rwd_list.append(r)
    arr = np.array(rwd_list)
    print(f"{tau_label:>12s}  {arr.mean():8.4f}  {arr.std():8.4f}  {np.percentile(arr,5):8.4f}  "
          f"{np.percentile(arr,25):8.4f}  {np.median(arr):8.4f}  {np.percentile(arr,75):8.4f}  "
          f"{np.percentile(arr,95):8.4f}")

rwd_list = []
for s in all_eval:
    d = 1.0 * s['w']
    pred = [[p[0] + d, p[1]] for p in s['gt_norm']]
    t = C_TAU_REF * s['sqrt_tr']
    r = calculate_r_wd(s['gt_norm'], pred, tau=t)
    rwd_list.append(r)
arr = np.array(rwd_list)
print(f"{'adaptive tau':>12s}  {arr.mean():8.4f}  {arr.std():8.4f}  {np.percentile(arr,5):8.4f}  "
      f"{np.percentile(arr,25):8.4f}  {np.median(arr):8.4f}  {np.percentile(arr,75):8.4f}  "
      f"{np.percentile(arr,95):8.4f}")

print("\nConclusion: a smaller std means better cross-scale consistency; the adaptive tau std should be clearly below the fixed-tau std.")

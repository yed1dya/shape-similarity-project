import pandas as pd
import numpy as np
import os
import math
import time
from datetime import datetime
from scipy.stats import skew
from scipy.spatial import procrustes

# --- CONFIG ---
INPUT_SEGMENTS_CSV = "shape9_full_segments_20250714_005442.csv"
LABELS_CSV = "shape9_scores.csv"
PROGRESS_PRINT_INTERVAL = 20

# --- LOAD DATA ---
segments_df = pd.read_csv(INPUT_SEGMENTS_CSV)
labels_df = pd.read_csv(LABELS_CSV)[["child_id", "label"]]

# --- PREPROCESS ---
segments_df = segments_df.dropna(subset=["child_id"])
segments_df = segments_df.sort_values(by=["child_id", "component_num"])
segments_df["x_values"] = segments_df["x_values"].apply(lambda x: list(map(int, str(x).split(","))))
segments_df["y_values"] = segments_df["y_values"].apply(lambda y: list(map(int, str(y).split(","))))

# --- GROUP BY IMAGE ---
grouped = segments_df.groupby("child_id")
total = len(grouped)

# --- PRECOMPUTE REFERENCE POINTS FOR PROCRUSTES ---
if "0000" in grouped.groups:
    ref_group = grouped.get_group("0000")
    ref_segs = list(zip(ref_group["x_values"], ref_group["y_values"]))
    pts_ref = [pt for (xs, ys) in ref_segs for pt in zip(xs, ys)]
    ref_xs = [p[0] for p in pts_ref]
    # determine image horizontal center
    img_xmin, img_xmax = min(ref_xs), max(ref_xs)
    center_x_img = (img_xmin + img_xmax) / 2.0
else:
    pts_ref = []
    center_x_img = 0

# --- GEOMETRY HELPERS ---
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def curvature(contour):
    curvatures = []
    for i in range(1, len(contour) - 1):
        p1, p2, p3 = contour[i - 1], contour[i], contour[i + 1]
        a = euclidean(p1, p2)
        b = euclidean(p2, p3)
        c = euclidean(p1, p3)
        if a * b == 0:
            continue
        angle = math.acos(max(min((a**2 + b**2 - c**2) / (2 * a * b), 1), -1))
        curvatures.append(angle)
    return curvatures

def convexity_defect(contour):
    x = [p[0] for p in contour]
    y = [p[1] for p in contour]
    area = abs(sum(x[i]*y[(i+1)%len(contour)] - x[(i+1)%len(contour)]*y[i] for i in range(len(contour)))) / 2.0
    hull_area = (max(x) - min(x)) * (max(y) - min(y))
    return max(hull_area - area, 0)

# --- MAIN FEATURE EXTRACTION ---
rows = []
for idx, (child_id, group) in enumerate(grouped):
    segments = list(zip(group["x_values"], group["y_values"]))
    num_contours = len(segments)

    # per-segment lists
    lengths = []
    aspects = []
    avg_curvs = []
    std_curvs = []
    inflection_counts = []
    centers_x = []
    centers_y = []
    concavity = []
    slopes = []
    widths = []

    for x_vals, y_vals in segments:
        pts = list(zip(x_vals, y_vals))
        if len(pts) < 2:
            continue

        # Length
        d = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = max(d, euclidean(pts[i], pts[j]))
        lengths.append(d)

        # Aspect ratio
        w = max(x_vals) - min(x_vals) + 1
        h = max(y_vals) - min(y_vals) + 1
        print(str(child_id), w, h, d)
        if h > 0:
            aspects.append(w / h)
        widths.append(w)

        # Curvature
        curvs = curvature(pts)
        if curvs:
            avg_curvs.append(np.mean(np.abs(curvs)))
            std_curvs.append(np.std(curvs))
            signs = np.sign(np.diff(curvs))
            inflection_counts.append(np.sum(np.diff(signs) != 0))
        else:
            avg_curvs.append(0)
            std_curvs.append(0)
            inflection_counts.append(0)

        # Centers
        centers_x.append(np.mean(x_vals))
        centers_y.append(np.mean(y_vals))

        # Concavity
        if len(pts) >= 3:
            concavity.append(convexity_defect(pts))

        # Slope (linear fit)
        m = np.polyfit(x_vals, y_vals, 1)[0]
        slopes.append(m)

    # spacing & alignment (existing)
    spacing_std = 0
    alignment_std = 0
    if len(centers_y) >= 2:
        sorted_idx = np.argsort(centers_y)
        sorted_y = np.array(centers_y)[sorted_idx]
        sorted_x = np.array(centers_x)[sorted_idx]
        spacing_std = np.std(np.diff(sorted_y))
        alignment_std = np.std(sorted_x)

    # 1. Left-right symmetry (mean abs deviation of segment centers from image center)
    if centers_x:
        left_right_symmetry = np.mean(np.abs(np.array(centers_x) - center_x_img))
    else:
        left_right_symmetry = 0

    # 3. Slope stats
    mean_slope = np.mean(slopes) if slopes else 0
    std_slope = np.std(slopes) if slopes else 0

    # 4. Width range
    if widths:
        width_range = max(widths) - min(widths)
    else:
        width_range = 0

    # 5. Vertical skewness
    vertical_skewness = skew(centers_y) if len(centers_y) >= 3 else 0

    rows.append({
        "child_id": child_id,
        "num_contours": num_contours,
        "std_length": np.std(lengths) if lengths else 0,
        "avg_aspect": np.mean(aspects) if aspects else 0,
        "avg_curvature": np.mean(avg_curvs) if avg_curvs else 0,
        "std_curvature": np.mean(std_curvs) if std_curvs else 0,
        "total_inflections": np.sum(inflection_counts),
        "spacing_std": spacing_std,
        "alignment_std": alignment_std,
        "avg_concavity": np.mean(concavity) if concavity else 0,
        "left_right_symmetry": left_right_symmetry,
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "width_range": width_range,
        "vertical_skewness": vertical_skewness
    })

    if (idx + 1) % PROGRESS_PRINT_INTERVAL == 0:
        print(f"Processed {idx + 1} / {total} images")

# --- MERGE LABELS AND SAVE ---
features_df = pd.DataFrame(rows)
features_df = features_df.merge(labels_df, on="child_id", how="left")
cols = ["child_id", "label"] + [c for c in features_df.columns if c not in ("child_id", "label")]
features_df = features_df[cols]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = f"shape9_features_{timestamp}.csv"
features_df.to_csv(out_csv, index=False)
print(f"Done. Output saved to {out_csv}")

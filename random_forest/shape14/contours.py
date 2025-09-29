import winsound
import cv2
import os
import csv
import numpy as np
import networkx as nx
from datetime import datetime

# --- CONFIG ---
SHAPE = 14
IMG_DIR = "images/shape14"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT = f"shape{SHAPE}_contours_{timestamp}.csv"
ANNOTATED_DIR = f"annotated_shape{SHAPE}_{timestamp}"

# Optional: specify which images to process
IMAGES_TO_PROCESS = [
    "4641", "7502", "7504", "7510", "7646", "7761", "10147",
    "10410", "10481", "10643", "10702-10703", "10793", "10957", "10361"
]
IMAGE_LIMIT = None

# Hole filtering thresholds
HOLE_MIN_SIZE = 80            # Minimum pixel count
ELLIPSE_MAX_AXIS_RATIO = 3    # Max major/minor axis ratio
ELLIPSE_MIN_AREA_RATIO = 0.8  # Min area fill ratio inside ellipse
DENSITY_MAX = 0.75             # Max fill density (raw fill inside contour)
HOLE_COVERAGE_RATIO = 0.3     # Min total hole area fraction to replace contour
DIST_THRESHOLD = 10           # Max distance to connect contours

# Morphology kernel size for closing individual components
CLOSE_KERNEL_SIZE = (20, 20)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL_SIZE)

# Helper: contour distance (min pairwise distance)
def contour_distance(c1, c2):
    pts1 = c1.reshape(-1, 2).astype(float)
    pts2 = c2.reshape(-1, 2).astype(float)
    d = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
    return d.min()

# Parse image list helper
def parse_image_list(image_list):
    if image_list is None:
        return None
    ids = set()
    for item in image_list:
        if "-" in item:
            start, end = map(int, item.split("-"))
            ids.update(str(i) for i in range(start, end + 1))
        else:
            ids.add(item)
    return ids


valid_ids = parse_image_list(IMAGES_TO_PROCESS)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Write CSV header
with open(CSV_OUT, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(["child_id", "component_num", "hole_num", "x_values", "y_values"])

# Palette for graph-coloring; full palette available, will slice later
PALETTE = [
    (0, 0, 255), (255, 0, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 128), (128, 128, 0)
]

# Process each image
all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(f"_{SHAPE}mnist.png")]
selected = [f for f in all_files if valid_ids is None or f.split("_")[0] in valid_ids]
if IMAGE_LIMIT:
    selected = selected[:IMAGE_LIMIT]

try:
    for idx, fname in enumerate(selected, 1):
        print(f"Processing {fname} ({idx}/{len(selected)})")
        child = fname.split("_")[0]
        img = cv2.imread(os.path.join(IMG_DIR, fname), cv2.IMREAD_GRAYSCALE)

        # Binarize image
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Initial contour detection on raw binary
        raw_contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        final_contours = []
        ellipse_list = []

        for comp_idx, cnt in enumerate(raw_contours, 1):
            # Create component mask (raw fill) for density check
            comp_mask = np.zeros_like(bin_img)
            cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=-1)

            # Compute fill density: fraction of raw contour area actually white
            filled_raw = cv2.bitwise_and(comp_mask, bin_img)
            filled_area = cv2.countNonZero(filled_raw)
            mask_area = cv2.countNonZero(comp_mask)
            density = filled_area / mask_area if mask_area > 0 else 0
            print(f"  contour {comp_idx}: density={density:.3f}", end="")
            if density > DENSITY_MAX:
                final_contours.append((comp_idx, 0, cnt))
                print("  skipping")
                continue
            print()

            # Close this single component to fill its own gaps
            comp_closed = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, kernel)
            closed_cnts, _ = cv2.findContours(comp_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not closed_cnts:
                continue
            closed_cnt = max(closed_cnts, key=lambda x: cv2.contourArea(x))

            # Hole detection inside closed component
            interior_bg = cv2.bitwise_and(comp_closed, cv2.bitwise_not(bin_img))
            n_labels, labels = cv2.connectedComponents(interior_bg)
            valid_holes = []
            total_hole_area = 0.0
            print("  checking holes")
            for lbl in range(1, n_labels):
                hole_mask = (labels == lbl).astype(np.uint8) * 255
                size = cv2.countNonZero(hole_mask)
                if size < HOLE_MIN_SIZE:
                    continue
                hole_conts, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for hcnt in hole_conts:
                    if hcnt.shape[0] < 5:
                        continue
                    (ex, ey), (a1, a2), ang = cv2.fitEllipse(hcnt)
                    major, minor = max(a1, a2), min(a1, a2)
                    ellipse_area = np.pi * (major / 2) * (minor / 2)
                    hole_area = cv2.contourArea(hcnt)
                    ratio = major / minor if minor != 0 else float("inf")
                    fill_ratio = hole_area / ellipse_area if ellipse_area != 0 else 1
                    if ratio > ELLIPSE_MAX_AXIS_RATIO:
                        print(f"    ellipse ratio check failed: ratio={ratio}")
                        continue

                    if fill_ratio <= ELLIPSE_MIN_AREA_RATIO:
                        print(f"    fill ratio check failed: fill_ratio={fill_ratio}")
                        continue

                    valid_holes.append((lbl, hcnt))
                    total_hole_area += hole_area
                    ellipse_list.append(((ex, ey), (a1, a2), ang))

            # Decide whether to replace contour or keep along with holes
            contour_area = cv2.contourArea(cnt)
            if total_hole_area >= HOLE_COVERAGE_RATIO * contour_area and len(valid_holes) > 1:
                # Replace: only draw holes
                print(f"  replacing contour with holes")
                for lbl, hcnt in valid_holes:
                    final_contours.append((comp_idx, lbl, hcnt))
            else:
                # Keep contour and holes
                print("  keeping contour")
                final_contours.append((comp_idx, 0, cnt))

        # Build adjacency graph and color
        G = nx.Graph()
        for i in range(len(final_contours)):
            G.add_node(i)
        for i in range(len(final_contours)):
            for j in range(i):
                if contour_distance(final_contours[i][2], final_contours[j][2]) < DIST_THRESHOLD:
                    G.add_edge(i, j)
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        num_colors = max(coloring.values()) + 1 if coloring else 1
        limited_palette = PALETTE[:num_colors]

        # Draw and save CSV
        with open(CSV_OUT, 'a', newline='') as cf:
            writer = csv.writer(cf)
            for node_idx, (comp_idx, lbl, cnt) in enumerate(final_contours):
                pts = cnt.squeeze()
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                xs, ys = pts[:, 0].tolist(), pts[:, 1].tolist()
                writer.writerow([child, comp_idx, lbl,
                                 ",".join(map(str, xs)),
                                 ",".join(map(str, ys))])
                color = limited_palette[coloring.get(node_idx, 0) % len(limited_palette)]
                print(f"node_idx: {node_idx}, color: {color}")
                cv2.drawContours(annotated, [cnt], -1, color, thickness=1)

        # Draw ellipses in green
        for (ex, ey), (a1, a2), ang in ellipse_list:
            center = (int(ex), int(ey))
            axes = (int(a1 / 2), int(a2 / 2))
            cv2.ellipse(annotated, center, axes, ang, 0, 360, (0, 204, 0), 1)

        cv2.imwrite(os.path.join(ANNOTATED_DIR, fname), annotated)
    print("Done.")
    winsound.Beep(400, 1000)
except Exception as e:
    print(e)
    winsound.MessageBeep()

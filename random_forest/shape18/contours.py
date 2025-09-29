import traceback

import winsound
import cv2
import os
import csv
import numpy as np
import networkx as nx
from datetime import datetime

# --- CONFIG ---
SHAPE = 18
IMG_DIR = "images/shape18"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT = f"shape{SHAPE}_contours_{timestamp}.csv"
ANNOTATED_DIR = f"annotated_shape{SHAPE}_{timestamp}"
DIST_THRESHOLD = 50

# Optional: specify which images to process
IMAGES_TO_PROCESS = None  # ["4641", "7502", "7504", "7510", "7646", "7761", "10147", "10410", "10481", "10643", "10702-10703", "10793", "10957", "10361"]
IMAGE_LIMIT = None


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
    writer.writerow(["child_id", "component_num", "x_values", "y_values"])

# Palette for graph-coloring; full palette available, will slice later
PALETTE = [
    (255, 0, 0), (0, 128, 255), (255, 0, 255),
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
        _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Initial contour detection on raw binary
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Build adjacency graph and color
        G = nx.Graph()
        for i in range(len(contours)):
            G.add_node(i)
        for i in range(len(contours)):
            for j in range(i):
                if contour_distance(contours[i], contours[j]) < DIST_THRESHOLD:
                    G.add_edge(i, j)
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        num_colors = max(coloring.values()) + 1 if coloring else 1
        limited_palette = PALETTE[:num_colors]

        # Draw and save CSV
        with open(CSV_OUT, 'a', newline='') as cf:
            writer = csv.writer(cf)
            for i, cnt in enumerate(contours):
                pts = cnt.squeeze()
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                xs, ys = pts[:, 0].tolist(), pts[:, 1].tolist()
                writer.writerow([child, i,
                                 ",".join(map(str, xs)),
                                 ",".join(map(str, ys))])
                color = limited_palette[coloring.get(i, 0) % len(limited_palette)]
                cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), thickness=1)
                # Draw convex hull
                hull = cv2.convexHull(cnt)
                cv2.drawContours(annotated, [hull], -1, color, thickness=1)
                M = cv2.moments(cnt)
                if M.get("m00", 0) != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.drawMarker(annotated, (cX, cY), (0, 0, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
        cv2.imwrite(os.path.join(ANNOTATED_DIR, fname), annotated)
    print("Done.")
    winsound.Beep(400, 1000)
except Exception as e:
    print(traceback.format_exc())
    winsound.MessageBeep()

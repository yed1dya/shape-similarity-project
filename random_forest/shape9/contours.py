import cv2
import os
import csv
import numpy as np
from datetime import datetime
from skimage.morphology import skeletonize

# --- CONFIG ---
IMG_DIR = "images/shape9"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT = f"shape9_full_segments_{timestamp}.csv"
ANNOTATED_DIR = f"annotated_shape9_{timestamp}"
SKELETON_DIR = f"skeleton_shape9_{timestamp}"

# Optional: specify which images to process
# Format: list of strings, e.g. ["7513", "9500-9550", "10456"] or None to process all
IMAGES_TO_PROCESS = None  # process all images
IMAGE_LIMIT = None  # e.g., 10 to process only the first 10 matching images

# --- Helper to parse image IDs ---
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

# Create output directories
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(SKELETON_DIR, exist_ok=True)

# --- Collect valid file list ---
all_files = [f for f in os.listdir(IMG_DIR) if f.endswith("_9mnist.png")]
selected_files = []
for f in all_files:
    cid = f.split("_")[0]
    if valid_ids is None or cid in valid_ids:
        selected_files.append(f)

if IMAGE_LIMIT is not None:
    selected_files = selected_files[:IMAGE_LIMIT]

total_files = len(selected_files)
processed_count = 0

# --- Fixed color palette with good visibility ---
PALETTE = [
    (0, 128, 255),   # Orange-ish blue
    (0, 200, 100),   # Medium green
    (255, 100, 0),   # Orange
    (200, 0, 200),   # Magenta
    (0, 180, 180),   # Teal
    (180, 180, 0),   # Olive
    (255, 80, 80),   # Coral
    (120, 0, 200),   # Purple
    (0, 150, 255),   # Sky blue
]

# --- Main processing ---
with open(CSV_OUT, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["child_id", "component_num", "x_values", "y_values"])

    for fname in selected_files:
        processed_count += 1
        print(f"Processing {fname} ({processed_count}/{total_files})")

        child_id = fname.split("_")[0]
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Threshold original image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Skeletonize the binary image
        skel = skeletonize(binary // 255)  # convert to boolean, then skeletonize
        skel = skel.astype(np.uint8) * 255  # back to uint8 for OpenCV

        # Save skeleton image
        skeleton_path = os.path.join(SKELETON_DIR, fname)
        cv2.imwrite(skeleton_path, skel)

        # Find contours on the skeleton
        contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # For drawing
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            coords = cnt.squeeze()
            if len(coords.shape) != 2 or coords.shape[0] < 2:
                continue  # skip degenerate contours

            x_vals = coords[:, 0].tolist()
            y_vals = coords[:, 1].tolist()

            writer.writerow([
                child_id,
                i + 1,
                ",".join(map(str, x_vals)),
                ",".join(map(str, y_vals))
            ])

            # Draw contour with fixed palette
            color = PALETTE[i % len(PALETTE)]
            cv2.drawContours(annotated, [cnt], -1, color, 1)

        out_path = os.path.join(ANNOTATED_DIR, fname)
        cv2.imwrite(out_path, annotated)

print("Done.")

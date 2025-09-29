# stardist_detection.py

import os
from glob import glob
from datetime import datetime
import numpy as np
from skimage.io import imread, imsave
from skimage.color import label2rgb
from csbdeep.utils import normalize
from stardist.models import StarDist2D

# --- CONFIG ---
INPUT_DIR    = r"annotated_shape14_20250718_151016"
MODEL_NAME   = "2D_versatile_fluo"  # Versatile fluorescent-nuclei model
PROB_THRESH  = 0.5
NMS_THRESH   = 0.3

# Create timestamped output folder
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"stardist_output_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the pre-trained StarDist2D model
model = StarDist2D.from_pretrained(MODEL_NAME)  # :contentReference[oaicite:0]{index=0}

# Process each PNG in the input directory
for img_path in glob(os.path.join(INPUT_DIR, "*.png")):
    fname = os.path.basename(img_path)

    # --- READ & NORMALIZE ---
    img = imread(img_path)
    if img.ndim == 3:            # if RGB, take the first channel
        img = img[..., 0]
    img_norm = normalize(img, 1, 99.8, axis=(0, 1))

    # --- PREDICT INSTANCES ---
    # returns: labels (H×W int mask), details (polygon coords, probs…)
    labels, _ = model.predict_instances(
        img_norm,
        prob_thresh=PROB_THRESH,
        nms_thresh=NMS_THRESH
    )  # :contentReference[oaicite:1]{index=1}

    # Report
    n_objs = labels.max()
    print(f"{fname}: {n_objs} objects detected")

    # --- SAVE RESULTS ---
    # 1) raw label mask
    lbl_path = os.path.join(OUTPUT_DIR, fname.replace(".png", "_labels.png"))
    imsave(lbl_path, labels.astype(np.uint16))

    # 2) color overlay for quick inspection
    overlay = label2rgb(labels, image=img, bg_label=0)
    ov_path  = os.path.join(OUTPUT_DIR, fname.replace(".png", "_overlay.png"))
    imsave(ov_path, (overlay * 255).astype(np.uint8))

print("Done. Outputs in:", OUTPUT_DIR)

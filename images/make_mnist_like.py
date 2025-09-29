import os
import cv2
import numpy as np
from datetime import datetime

for SHAPE in range(18, 19):
    # --- CONFIG ---
    ROOT_DIR = r"C:\Users\user\Documents\Sample Shapes\children"
    TARGET_NAME = f"{SHAPE}.png"
    KERNEL_WIDTH = 1  # Width of dilation kernel (horizontal thickening)
    KERNEL_HEIGHT = 1  # Height of dilation kernel (vertical thickening)
    LIMIT = None  # Set to an integer to limit number of images processed
    MARGIN = 20  # Margin in pixels around the square crop
    LOW = 50
    HIGH = 250

    # Create output directory with timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), f"shape_{SHAPE}_"
                                                         f"kernel_{KERNEL_WIDTH}_{KERNEL_HEIGHT}_"
                                                         f"low_thresh_{LOW}_"
                                                         f"up_thresh_{HIGH}_"
                                                         f"{TIMESTAMP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each child folder
    processed_count = 0
    for folder in os.listdir(ROOT_DIR):
        if LIMIT is not None and processed_count >= LIMIT:
            break

        child_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(child_path):
            continue

        extracted_path = os.path.join(child_path, "Extracted images")
        if not os.path.isdir(extracted_path):
            continue

        img_path = os.path.join(extracted_path, TARGET_NAME)
        if not os.path.isfile(img_path):
            continue

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Create mask for blue pixels (relaxed range)
        lower_blue = np.array([LOW, 0, 0])
        if folder == "0000":
            HIGH_t = 100
        else:
            HIGH_t = HIGH
        upper_blue = np.array([255, HIGH_t, HIGH_t])
        mask = cv2.inRange(img, lower_blue, upper_blue)

        # Dilate to thicken lines
        kernel = np.ones((KERNEL_HEIGHT, KERNEL_WIDTH), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Close gaps in lines
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closing_kernel)

        # Create and save thickened blue image
        # thick_blue_img = np.ones_like(img) * 255
        # thick_blue_img[closed > 0] = [255, 0, 0]
        # blue_output_filename = f"{folder}_{SHAPE}blue.png"
        # blue_output_path = os.path.join(OUTPUT_DIR, blue_output_filename)
        # cv2.imwrite(blue_output_path, thick_blue_img)

        # Convert to white lines on black background
        binary_image = np.zeros_like(mask)
        binary_image[closed > 0] = 255

        # Find bounding box of white pixels
        coords = cv2.findNonZero(binary_image)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Center of bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Side of the square crop (max of w, h) + margin
            side = max(w, h) + 2 * MARGIN

            # Calculate top-left coordinates of the square crop
            x_new = center_x - side // 2
            y_new = center_y - side // 2

            # Check if square crop with margin fits within the image
            if x_new >= 0 and y_new >= 0 and x_new + side <= binary_image.shape[1] and y_new + side <= \
                    binary_image.shape[0]:
                binary_image = binary_image[y_new:y_new + side, x_new:x_new + side]

        # Save processed image
        output_filename = f"{folder}_{SHAPE}mnist.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        binary_image = cv2.resize(binary_image, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, binary_image)
        processed_count += 1
    print(f"shape {SHAPE} done")

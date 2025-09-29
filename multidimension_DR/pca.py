import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from datetime import datetime
import base64
import io

# Parameters
SHAPES_FOLDER = r"C:\Users\user\Documents\Sample Shapes\shapes"  # Parent folder containing shape folders
OUTPUT_DIM = 100  # Number of PCA components (adjust as needed)
CURRENT_DATETIME = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
OUTPUT_FOLDER = r"C:\Users\user\Documents\Sample Shapes\each shape\multidimension\PCA"  # Directory to save output
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# List of shapes to process
NUMBERS = [5]
SELECTED_SHAPES = [f"shape{i}" for i in NUMBERS]  # Modify this list as needed


# Function to calculate distance (default: Euclidean)
def calculate_distance(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


# Function to load images and preprocess
def load_images_as_vectors_and_base64(folder):
    images = []
    titles = []
    base64_images = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).convert("L")  # Grayscale
            img_vector = np.array(img).flatten()  # Flatten the image
            images.append(img_vector)
            titles.append(file.split(".")[0])  # Extract filename without extension

            # Convert image to Base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_base64)
    return np.array(images), titles, base64_images


# Function to perform PCA and save results
def perform_pca_on_shape(shape_folder, output_dim):
    print(f"Processing {shape_folder}...")
    image_data, image_titles, image_base64 = load_images_as_vectors_and_base64(shape_folder)
    print(f"Loaded {len(image_data)} images with shape {image_data[0].shape}")

    # Normalize the data
    print("Normalizing data...")
    image_data = image_data / 255.0  # Scale pixels to [0, 1]

    # Perform PCA
    print("Performing PCA...")
    pca = PCA(n_components=output_dim)
    reduced_data = pca.fit_transform(image_data)
    print(f"PCA completed. Reduced dimensions: {reduced_data.shape}")

    # Identify reference image (0000_<shape_number>)
    shape_number = os.path.basename(shape_folder).replace("shape", "")
    reference_image_name = f"0000_{shape_number}mnist"
    reference_index = image_titles.index(reference_image_name)
    reference_vector = reduced_data[reference_index]

    # Calculate distances to reference image
    distances = [calculate_distance(vector, reference_vector) for vector in reduced_data]
    max_distance = max(distances)
    normalized_distances = [dist / max_distance for dist in distances]  # Normalize distances

    # Save main PCA results to CSV
    shape_name = os.path.basename(shape_folder)
    output_file = f"{shape_name}_pca_{OUTPUT_DIM}-dims_output_{CURRENT_DATETIME}.csv"
    output_df = pd.DataFrame(reduced_data, columns=[f"dim{i + 1}" for i in range(output_dim)])
    output_df.insert(0, "serial_number", pd.Series(image_titles))
    output_df["image_base64"] = image_base64  # Add Base64 images to the DataFrame
    output_df["distance_to_0000"] = distances  # Add distance column
    output_df["normalized_distance"] = normalized_distances  # Add normalized distance column
    output_path = os.path.join(OUTPUT_FOLDER, output_file)
    output_df.to_csv(output_path, index=False)
    print(f"PCA results saved to {output_path}\n")

    # Save sorted results to an HTML file
    sorted_df = output_df.sort_values(by="distance_to_0000").reset_index(drop=True)
    html_output_file = f"{shape_name}_sorted_by_distance_{OUTPUT_DIM}-dims_pca_{CURRENT_DATETIME}.html"
    html_output_path = os.path.join(OUTPUT_FOLDER, html_output_file)

    # Generate HTML content
    html_content = f"""
    <html>
    <head><title>Sorted Results by Distance</title></head>
    <body>
    <h1>Sorted Results by Distance for {shape_name}</h1>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Serial Number</th>
            <th>Distance to 0000</th>
            <th>Normalized Distance</th>
            <th>Image</th>
        </tr>
    """
    for _, row in sorted_df.iterrows():
        html_content += f"""
        <tr>
            <td>{row['serial_number']}</td>
            <td>{row['distance_to_0000']:.4f}</td>
            <td>{row['normalized_distance']:.4f}</td>
            <td><img src="data:image/png;base64,{row['image_base64']}" width="100" /></td>
        </tr>
        """
    html_content += """</table></body></html>"""

    # Write HTML to file
    with open(html_output_path, "w") as html_file:
        html_file.write(html_content)
    print(f"Sorted HTML results saved to {html_output_path}\n")


# Process selected shape folders
for shape_folder in SELECTED_SHAPES:
    shape_path = os.path.join(SHAPES_FOLDER, shape_folder)
    if os.path.isdir(shape_path):
        perform_pca_on_shape(shape_path, OUTPUT_DIM)

import base64
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Parameters
SHAPES_FOLDER = r"C:\Users\user\Documents\Sample Shapes\shapes"  # Input parent folder
AUTOENCODER_OUTPUT_DIM = 50  # Final autoencoder dimensions (increased further)
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.0005
CURRENT_DATETIME = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
OUTPUT_FOLDER = r"C:\Users\user\Documents\Sample Shapes\each shape\multidimension\Autoencoder_Only"  # Output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# List of shapes to process
NUMBERS = [5]
SELECTED_SHAPES = [f"shape{i}" for i in NUMBERS]

# Function to save parameters to text file
def save_parameters():
    params = {
        "SHAPES_FOLDER": SHAPES_FOLDER,
        "AUTOENCODER_OUTPUT_DIM": AUTOENCODER_OUTPUT_DIM,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "VALIDATION_SPLIT": VALIDATION_SPLIT,
        "LEARNING_RATE": LEARNING_RATE,
        "OUTPUT_FOLDER": OUTPUT_FOLDER,
        "DATE_TIME": CURRENT_DATETIME
    }
    param_file = os.path.join(OUTPUT_FOLDER, f"parameters_{CURRENT_DATETIME}.txt")
    with open(param_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved to {param_file}\n")

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
            img_vector = np.array(img).flatten()
            images.append(img_vector)
            titles.append(file.split(".")[0])

            # Convert image to Base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_base64)
    return np.array(images), titles, base64_images

# Function to build and train denoising autoencoder
def train_autoencoder(input_dim, encoding_dim, data):
    print("Training denoising autoencoder with increased latent space and lower learning rate...")
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(512, activation='relu')(input_layer)
    encoded = layers.Dense(256, activation='relu')(encoded)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck

    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(256, activation='relu')(decoded)
    decoded = layers.Dense(512, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    # Compile with reduced learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    # Add noise to input data
    noisy_data = data + np.random.normal(0, 0.05, data.shape)
    noisy_data = np.clip(noisy_data, 0., 1.)  # Ensure data remains in valid range

    # Track loss history
    history = autoencoder.fit(
        noisy_data, data, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_split=VALIDATION_SPLIT
    )

    # Plot loss history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(OUTPUT_FOLDER, f"loss_plot_{CURRENT_DATETIME}.png")
    plt.savefig(loss_plot_path)
    plt.show()
    print(f"Loss plot saved to {loss_plot_path}")

    print("Autoencoder training complete.")
    return encoder

# Function to perform Autoencoder-only and save results
def perform_autoencoder_on_shape(shape_folder, autoencoder_dim):
    print(f"Processing {shape_folder}...")
    image_data, image_titles, image_base64 = load_images_as_vectors_and_base64(shape_folder)
    print(f"Loaded {len(image_data)} images with shape {image_data[0].shape}")

    # Normalize the data
    print("Normalizing data...")
    scaler = MinMaxScaler()
    image_data = scaler.fit_transform(image_data)

    # Train Autoencoder
    encoder = train_autoencoder(image_data.shape[1], autoencoder_dim, image_data)
    encoded_data = encoder.predict(image_data)
    print(f"Autoencoder reduced dimensions: {encoded_data.shape}")

    # Identify reference image
    shape_number = os.path.basename(shape_folder).replace("shape", "")
    reference_image_name = f"0000_{shape_number}mnist"
    reference_index = image_titles.index(reference_image_name)
    reference_vector = encoded_data[reference_index]

    # Calculate distances
    distances = [calculate_distance(vector, reference_vector) for vector in encoded_data]
    max_distance = max(distances)
    normalized_distances = [dist / max_distance for dist in distances]

    # Save Autoencoder results to CSV
    shape_name = os.path.basename(shape_folder)
    output_file = f"{shape_name}_autoencoder_{autoencoder_dim}_output_{CURRENT_DATETIME}.csv"
    output_df = pd.DataFrame(encoded_data, columns=[f"dim{i + 1}" for i in range(autoencoder_dim)])
    output_df.insert(0, "serial_number", pd.Series(image_titles))
    output_df["image_base64"] = image_base64
    output_df["distance_to_0000"] = distances
    output_df["normalized_distance"] = normalized_distances
    output_path = os.path.join(OUTPUT_FOLDER, output_file)
    output_df.to_csv(output_path, index=False)
    print(f"Autoencoder results saved to {output_path}\n")

    # Save sorted results to an HTML file
    sorted_df = output_df.sort_values(by="distance_to_0000").reset_index(drop=True)
    html_output_file = f"{shape_name}_sorted_by_distance_autoencoder_{autoencoder_dim}_{CURRENT_DATETIME}.html"
    html_output_path = os.path.join(OUTPUT_FOLDER, html_output_file)

    html_content = f"""
    <html>
    <head><title>Sorted Results by Distance</title></head>
    <body>
    <h1>Sorted Results by Distance for {shape_name}, Autoencoder ({autoencoder_dim} dimensions)</h1>
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

    with open(html_output_path, "w") as html_file:
        html_file.write(html_content)
    print(f"Sorted HTML results saved to {html_output_path}\n")

# Save parameters to a text file
save_parameters()

# Process selected shape folders
for shape_folder in SELECTED_SHAPES:
    shape_path = os.path.join(SHAPES_FOLDER, shape_folder)
    if os.path.isdir(shape_path):
        perform_autoencoder_on_shape(shape_path, AUTOENCODER_OUTPUT_DIM)

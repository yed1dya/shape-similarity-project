# Shape Similarity Project

This repository contains the codebase for analyzing, classifying, and measuring the similarity between drawn shapes. The project evaluates shape representations using a combination of traditional computer vision metrics, supervised machine learning, and unsupervised deep learning techniques.

## Repository Structure

* **`image_processing/` & `images/`**: Data preparation pipelines. Includes scripts to organize raw data and standardize images (e.g., converting to MNIST-like formats).
* **`unsupervised_IoU_Hausdorff/` & `unsupervides_ssim/`**: Baselines using traditional computer vision metrics for structural similarity, specifically Intersection over Union (IoU), Hausdorff Distance, and Structural Similarity Index (SSIM).
* **`random_forest/`**: Supervised classification baselines. Contains scripts for geometric feature extraction (contours) and training Random Forest classifiers (segmented by shape types).
* **`cnn.ipynb`**: Convolutional Neural Network implementations for standard shape classification.
* **`fine_tuning/siamis/`**: Siamese Network implementations for metric learning, focused on directly computing similarity scores between shape pairs.
* **`unsuper_vae_autoencoder/`**: Unsupervised representation learning models utilizing standard Autoencoders and Variational Autoencoders (VAEs) to learn latent shape features.
* **`dm_red_each_shpae/` & `clustring_for_all_shapes/`**: Dimensionality reduction and clustering scripts. Applies PCA, t-SNE, and UMAP to the latent representations extracted by the autoencoders for visualization and grouping.

## Core Methodologies

1.  **Geometric & Pixel-Level Metrics**: Establishing similarity using direct mathematical comparisons (SSIM, IoU, Hausdorff).
2.  **Feature Engineering**: Extracting explicit contours and topological features for classical ML models.
3.  **Latent Space Analysis**: Compressing shapes into lower-dimensional embeddings via VAEs/Autoencoders to evaluate similarity based on proximity in the latent space.
4.  **Metric Learning**: Training Siamese networks to optimize the distance between similar and dissimilar shapes dynamically.

## Setup and Usage

1.  Clone the repository.
2.  Install the required dependencies. The notebooks and scripts primarily rely on:
    * `numpy`, `pandas`, `matplotlib`, `seaborn`
    * `scikit-learn`
    * `opencv-python` (cv2)
    * `tensorflow` / `keras` / `pytorch` 
3.  Execute the preprocessing scripts in `image_processing/` to format the dataset.
4.  Navigate to the specific methodology folders (e.g., `random_forest/shape14/train_RF.py` or the Jupyter notebooks) to train models, extract features, or generate evaluation HTML outputs.

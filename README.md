# Satellite-Based Flood Detection Pipeline

![Flood Detection Banner](https://placehold.co/1200x400/007BFF/FFFFFF?text=Flood+Detection+with+SAR+%26+Deep+Learning)

A comprehensive, end-to-end pipeline for detecting and mapping floodwater from satellite imagery. This project implements state-of-the-art methodologies, from data preprocessing to deep learning-based semantic segmentation, based on a thorough research plan for robust and accurate flood analysis.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Path A: Rapid Response (Classical ML)](#path-a-rapid-response--proof-of-concept)
  - [Path B: High-Accuracy (Deep Learning)](#path-b-high-accuracy--research-grade)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Dataset Acquisition Guide](#dataset-acquisition-guide)
- [Usage](#usage)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training a Model](#2-training-a-model)
  - [3. Running Inference](#3-running-inference)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Project Description

This repository provides the tools to build, train, and evaluate models for flood mapping using publicly available satellite data. The primary goal is to leverage the all-weather capabilities of Synthetic Aperture Radar (SAR) data from Sentinel-1, with optional fusion of optical data from Sentinel-2.

The project is designed with two distinct pathways: a rapid, proof-of-concept model using classical machine learning for quick deployment, and a research-grade, high-accuracy model using deep learning (U-Net) for state-of-the-art performance.

## Features

- **End-to-End Workflow:** From data acquisition and preprocessing to model training, evaluation, and inference.
- **Dual-Sensor Capability:** Primarily focused on **Sentinel-1 (SAR)** for its all-weather reliability, with support for **Sentinel-2 (Optical)** data fusion.
- **Multi-Modal Approach:** Implements both classical machine learning (Random Forest) and state-of-the-art deep learning (U-Net) models.
- **Standardized Preprocessing:** Includes scripts and guidelines for creating Analysis-Ready Data (ARD) using tools like the ESA SNAP Engine.
- **Benchmark Dataset Integration:** Supports training and evaluation on standard public datasets like **Sen1Floods11** and **WorldFloods**.
- **Robust Evaluation:** Employs industry-standard metrics for imbalanced segmentation tasks, including **Intersection over Union (IoU)** and **F1-Score**.

## Project Structure

The project is structured as a Python package for easy installation and use.

```
flood-detection-pipeline/
├── data/                    # Data directory (not tracked by Git)
│   ├── raw/                 # Raw downloaded satellite imagery
│   ├── processed/           # Preprocessed, analysis-ready data
│   └── training_data/       # Labeled datasets (e.g., Sen1Floods11)
│
├── notebooks/               # Jupyter notebooks for exploration and analysis
│
├── results/                 # Output directory for models and flood maps
│   ├── models/              # Saved model weights
│   └── maps/                # Generated flood extent maps (GeoTIFFs)
│
├── src/
│   └── flood_detector/      # Main source code package
│       ├── __init__.py
│       ├── config.py        # Project configuration and constants
│       ├── data_handler.py  # Data loading and preparation
│       ├── evaluate.py      # Model evaluation logic and metrics
│       ├── models/          # Model architecture definitions
│       │   ├── __init__.py
│       │   ├── random_forest.py
│       │   └── unet.py
│       ├── preprocess.py    # Preprocessing pipeline scripts (e.g., SNAP integration)
│       ├── train.py         # Model training script
│       └── predict.py       # Inference script to generate flood maps
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt         # Python dependencies
```

## Methodology

This project implements two distinct pathways as outlined in the foundational research document.

### Path A: Rapid Response / Proof-of-Concept

-   **Algorithm:** Random Forest Classifier.
-   **Data:** Sentinel-1 GRD (pre- and post-flood).
-   **Features:** VV & VH backscatter, change detection features (difference/ratio), and DEM derivatives (slope, elevation).
-   **Use Case:** Ideal for rapid deployment during an emergency, requiring less data and computational power.

### Path B: High-Accuracy / Research-Grade

-   **Algorithm:** U-Net (or variants like Attention U-Net).
-   **Data:** Fused Sentinel-1 (SAR) and Sentinel-2 (Optical) data.
-   **Features:** Multi-channel input tensor including SAR backscatter, optical bands, and ancillary data like the HAND index.
-   **Use Case:** Aims for state-of-the-art accuracy, suitable for research and developing operational systems. Requires a GPU and a large labeled dataset.

## Data Sources

### Understanding the Two-Phase Data Strategy

This project operates with a two-phase data approach:

#### **Training Phase (The "Textbook")**
- **Dataset**: Sen1Floods11 and WorldFloods
- **Purpose**: These are massive, pre-labeled datasets containing thousands of examples of "water" and "not water" with ground truth labels. Think of these as comprehensive textbooks that teach your model the "language" of what floods look like in satellite imagery. This is the "studying" phase where the model learns flood detection patterns.

#### **Prediction Phase (The "Real-World Application")**
- **Dataset**: Your own downloaded satellite images (e.g., Limburg flood event)
- **Purpose**: After your model has "studied" the textbook datasets and become "fluent" in flood detection, you test it on new, unseen events. You provide the trained model with a specific flood event image and ask: "Based on everything you've learned, where is the flood in this image?"

### Data Requirements by Model Type

**⚠️ Important**: The number of images required depends on your chosen approach:

- **Random Forest (Path A)**: Requires **TWO images** - a pre-flood baseline and post-flood image for change detection
- **U-Net (Path B)**: Requires **ONE image** - only the post-flood scene, as it learns intrinsic water patterns

### Data Sources

-   **Primary Imagery:**
    -   [Sentinel-1 (SAR)](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1) - All-weather capability, primary data source
    -   [Sentinel-2 (Optical)](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2) - High-resolution optical data (weather dependent)
-   **Ancillary Data:**
    -   [Copernicus DEM](https://registry.opendata.aws/copernicus-dem/) - Digital Elevation Model for topographic analysis
-   **Training/Benchmark Datasets:**
    -   [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11): ~14GB dataset for SAR-based models (Sentinel-1)
    -   [WorldFloods](https://www.worldfloods.org/): ~76GB dataset for optical-based models (Sentinel-2)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/flood-detection-pipeline.git
    cd flood-detection-pipeline
    ```

2.  **Install ESA SNAP:**
    This project relies on the ESA SNAP Engine for SAR preprocessing. Please [download and install SNAP](http://step.esa.int/main/download/snap-download/) and configure the Python interface (`snappy`). Follow the instructions provided by ESA.

TODO: We are using poetry and not conde/venv.
3.  **Create a Python environment and install dependencies:**
    It is highly recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    **`requirements.txt`:**
    ```
    # Geospatial Libraries
    gdal
    rasterio
    geopandas
    shapely

    # Machine Learning
    scikit-learn
    tensorflow  # or torch, torchvision, torchaudio

    # Core Libraries
    numpy
    pandas
    matplotlib
    tqdm
    opencv-python-headless
    ```

## Dataset Acquisition Guide

This section provides a complete A-to-Z guide for downloading both training datasets and satellite imagery for your specific flood events.

### Part 1: Downloading Benchmark Training Datasets

#### A. Sen1Floods11 Dataset (for SAR Models)

The Sen1Floods11 dataset (~14GB) is specifically designed for Sentinel-1 based flood segmentation and is hosted on Google Cloud Storage.

**Prerequisites:**
1. Install Google Cloud SDK (includes `gsutil` tool):
   - Download from: https://cloud.google.com/sdk/docs/install
   - Follow installation instructions for Windows

**Download Steps:**
```bash
# Navigate to your project directory
cd C:\Users\Soheil\Documents\GitHub\flood-detection-pipeline

# Create target directory
mkdir data\training_data\Sen1Floods11

# Download the complete dataset (this will take time - ~14GB)
gsutil -m rsync -r gs://sen1floods11 data\training_data\Sen1Floods11\
```

#### B. WorldFloods Dataset (for Optical Models)

The WorldFloods dataset (~76GB) provides Sentinel-2 optical images and flood masks, hosted on Hugging Face.

**Prerequisites:**
```bash
# Install Hugging Face Hub
pip install huggingface_hub
```

**Download Steps:**
```bash
# Login to Hugging Face (optional but recommended)
huggingface-cli login

# Create target directory
mkdir data\training_data\WorldFloods

# Download the dataset
huggingface-cli download isp-uv-es/WorldFloodsv2 --repo-type dataset --local-dir data\training_data\WorldFloods
```

### Part 2: Downloading Satellite Imagery for Specific Events

#### Recommended Case Study: July 2021 Limburg Floods, Netherlands

For a powerful case study with clear flood signals, use the July 2021 floods in Limburg province, which caused significant riverine flooding along the Geul and Meuse rivers.

**Target Location:** Valkenburg aan de Geul, Netherlands
**Coordinates:** 50.8661°N, 5.8308°E

#### A. Create Copernicus Account

1. Navigate to: https://dataspace.copernicus.eu/
2. Click "Register" in the top-right corner
3. Fill out registration form and verify your email
4. Login to your new account

#### B. Search and Download Data

1. **Access the Browser:**
   - From the main page, select "Explore the data" or launch "Copernicus Browser"

2. **Define Area of Interest:**
   - Search for "Valkenburg aan de Geul, Netherlands" in the search bar
   - OR manually draw a rectangle around the area using the drawing tools

3. **Set Search Filters (Left Panel):**

   **For Sentinel-1 (SAR) Data:**
   ```
   Mission: Sentinel-1
   Product Type: GRD (Ground Range Detected)
   Sensing Period:
     - Pre-flood: June 1, 2021 - July 10, 2021
     - Post-flood: July 14, 2021 - July 18, 2021
   ```

   **For Sentinel-2 (Optical) Data:**
   ```
   Mission: Sentinel-2
   Product Type: Level-2A (or Level-1C)
   Cloud Cover: 0% - 10% (crucial for usable images)
   Sensing Period:
     - Pre-flood: June 1, 2021 - July 10, 2021
     - Post-flood: July 14, 2021 - July 18, 2021
   ```

4. **Download Process:**
   - Click the search button to see available images
   - Review results by clicking each to see footprint and preview
   - Select images covering your timeframes (you need at least 2 for Random Forest, 1 for U-Net)
   - Click download icon for each selected scene
   - Files will download as .zip archives (several GB each)

5. **Organize Downloaded Data:**
   ```bash
   # Move downloaded files to your project structure
   move *.zip data\raw\
   ```

### Data Requirements Summary

| Model Type | Training Data | Prediction Data | Notes |
|------------|---------------|-----------------|-------|
| **Random Forest** | Sen1Floods11 | 2 images (pre + post flood) | Requires change detection |
| **U-Net** | Sen1Floods11 | 1 image (post flood only) | Learns intrinsic water patterns |
| **Optical Models** | WorldFloods | 1-2 images (depending on model) | Weather dependent |

### Troubleshooting Download Issues

**Large File Downloads:**
- Downloads may take hours due to file sizes (2-8GB per scene)
- Use stable internet connection and consider downloading overnight
- If downloads fail, restart from the Copernicus Browser

**Storage Requirements:**
- Sen1Floods11: ~14GB
- WorldFloods: ~76GB
- Raw satellite scenes: ~2-8GB each
- Ensure adequate disk space before starting downloads

**Authentication Issues:**
- For Google Cloud: Ensure `gsutil` is properly configured
- For Hugging Face: Create account and obtain access token if needed
- For Copernicus: Verify email and ensure account is activated

## Usage

The pipeline is executed via scripts within the `src/flood_detector/` directory.

### 1. Preprocessing

Use the `preprocess.py` script to convert raw Sentinel-1 data into Analysis-Ready Data. This script typically calls a SNAP Graph Processing Tool (GPT) graph.

```bash
python src/flood_detector/preprocess.py \
  --input_path data/raw/S1A_IW_GRDH_1SDV_...zip \
  --output_path data/processed/S1A_processed.tif \
  --graph_xml path/to/your/sar_preprocessing_graph.xml
```

### 2. Training a Model

Use the `train.py` script to train a new flood detection model.

**Example: Training a Random Forest model (Path A)**

```bash
python src/flood_detector/train.py \
  --model_type random_forest \
  --training_data data/training_data/my_rf_samples.csv \
  --model_output_path results/models/rf_model_v1.joblib
```

**Example: Training a U-Net model (Path B) with Enhanced Monitoring**

```bash
# Enhanced U-Net training with visual sample inference monitoring
poetry run python -m src.flood_detector.train \
  --model_type unet \
  --training_data data/training_data/Sen1Floods11/ \
  --model_output_path results/models/unet_model_v1.h5 \
  --epochs 10 \
  --batch_size 1
```

**New Feature: Sample Inference During Training**
The enhanced training pipeline now includes automatic sample inference after each epoch:
- Generates flood maps from validation samples to visually monitor training progress
- Saves individual sample results with input/ground truth/prediction comparisons
- Creates epoch summaries with IoU and accuracy metrics
- Results saved to `results/training_logs/sample_inference/epoch_XXX/`

### 3. Running Inference

Use the `predict.py` script to generate a flood map from a preprocessed satellite image.

# TODO: Convert to poetry run
```bash
# python src/flood_detector/predict.py \
#   --model_path results/models/unet_model_v1.h5 \
#   --input_image data/processed/post_flood_event.tif \
#   --output_map results/maps/flood_map_event.tif

poetry run python -m src.flood_detector.predict --model_path results/models/unet_final_test_best.h5 --post_flood_image data/training_data/Sen1Floods11/train/s1_post/Bolivia_76104_S1Hand.tif --output_map results/maps/flood_map_sample.tif
```

## Evaluation

Model performance is evaluated using metrics suitable for imbalanced datasets. The `evaluate.py` script calculates these from a set of predictions and ground truth labels.

- **Intersection over Union (IoU):** The primary metric for segmentation quality.
- **F1-Score:** The harmonic mean of Precision and Recall.
- **Precision:** Measures the accuracy of positive predictions (low false positives).
- **Recall (Sensitivity):** Measures the ability to find all positive samples (low false negatives). This is often the most critical metric for disaster response.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- This project would not be possible without the free and open data provided by the European Space Agency (ESA) through the Copernicus Programme.
- Data from the NASA/USGS Landsat Program.
- Benchmark datasets provided by Cloud to Street (Sen1Floods11) and other research institutions.

## Contact

Soheil Mohammadpour - [Your Email/LinkedIn]

Project Link: https://github.com/your-username/flood-detection-pipeline

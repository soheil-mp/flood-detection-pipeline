"""
config.py

Central configuration file for the Flood Detection Pipeline project.
This file contains constants, file paths, and model parameters
that are used across the various modules of the project.
"""

import os

# --- Project Root ---
# Assumes this config.py is in src/flood_detector/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Directory Paths ---
# Use os.path.join to ensure cross-platform compatibility
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")

# --- SNAP (ESA SNAP Toolbox) Configuration ---
# Path to the Graph Processing Tool (GPT) executable.
# This needs to be set by the user based on their SNAP installation.
# Example for Linux: '/usr/local/snap/bin/gpt'
# Example for Windows: 'C:\\Program Files\\snap\\bin\\gpt.exe'
# Example for macOS: '/Applications/snap/bin/gpt'
SNAP_GPT_PATH = "gpt"  # Assumes 'gpt' is in the system's PATH

# Path to the XML graph file for SAR preprocessing.
# This graph should be created using the SNAP Desktop application.
SAR_PREPROCESSING_GRAPH_XML = os.path.join(
    PROJECT_ROOT, "src", "flood_detector", "snap_graphs", "sar_preprocessing.xml"
)


# --- Data Parameters ---
# Sentinel-1 bands used in the project
S1_BANDS = ["VV", "VH"]

# Sentinel-2 bands used in the project
# Blue, Green, Red, NIR, SWIR1, SWIR2
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# --- U-Net Model Parameters ---
# Input image dimensions for the U-Net model
# Sen1Floods11 dataset uses 512x512 chips
IMG_HEIGHT = 512
IMG_WIDTH = 512
# Number of input channels. For a simple S1 model, this is 2 (VV, VH).
# For a fused model, this would be the sum of S1 and S2 bands.
IMG_CHANNELS = 2

# Training hyperparameters - Enhanced for stability
UNET_LEARNING_RATE = 5e-5  # Reduced from 1e-4 for better stability
UNET_BATCH_SIZE = 4  # Reduced from 8 to prevent memory issues
UNET_EPOCHS = 50

# Enhanced training parameters for stability
UNET_PATIENCE = 10  # Early stopping patience
UNET_LR_REDUCTION_FACTOR = 0.5  # Learning rate reduction factor
UNET_LR_REDUCTION_PATIENCE = 5  # Patience for learning rate reduction
UNET_MIN_LEARNING_RATE = 1e-7  # Minimum learning rate

# --- Random Forest Model Parameters ---
# Number of trees in the forest
RF_N_ESTIMATORS = 100
# Number of jobs to run in parallel (-1 means using all available processors)
RF_N_JOBS = -1
# Maximum number of samples to use for training to avoid memory issues
RF_MAX_SAMPLES = 100000  # Use a subset of pixels for training

# --- General ---
# Seed for random number generators to ensure reproducibility
RANDOM_SEED = 42


# --- Function to create directories if they don't exist ---
def create_project_directories():
    """
    Creates all the necessary directories for the project if they
    do not already exist.
    """
    dirs_to_create = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TRAINING_DATA_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        MAPS_DIR,
        os.path.join(PROJECT_ROOT, "src", "flood_detector", "snap_graphs"),
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    print("Project directories checked/created.")


if __name__ == "__main__":
    # When this script is run directly, it will create the project structure.
    create_project_directories()

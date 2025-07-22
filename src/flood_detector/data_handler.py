"""
data_handler.py

This module contains functions for loading, preparing, and augmenting
satellite imagery and label data for model training and inference.
It handles reading GeoTIFF files, generating training samples for
classical machine learning models, and creating data generators for
deep learning models.
"""

import os
import numpy as np
import rasterio
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

from flood_detector import config

# --- GeoTIFF I/O Functions ---


def read_geotiff(file_path):
    """
    Reads a GeoTIFF file and returns its data array and metadata.

    Args:
        file_path (str): Path to the GeoTIFF file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The image data as a NumPy array.
            - dict: The metadata of the GeoTIFF file (profile).
    """
    try:
        with rasterio.open(file_path) as src:
            # Rasterio reads bands first, so shape is (bands, height, width)
            # We will transpose it to (height, width, bands) for consistency
            data = src.read()
            profile = src.profile

            # Transpose from (bands, height, width) to (height, width, bands)
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))

            return data, profile
    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading GeoTIFF file {file_path}: {e}")
        return None, None


def write_geotiff(data, profile, file_path):
    """
    Writes a NumPy array to a GeoTIFF file.

    Args:
        data (np.ndarray): The image data array. Assumes shape
        (height, width) for single band
                           or (height, width, bands) for multi-band.
        profile (dict): The metadata profile for the output GeoTIFF.
        file_path (str): The path to save the output file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Update profile based on data array shape
        if data.ndim == 2:  # Single band
            profile.update(count=1, dtype=data.dtype)
        elif data.ndim == 3:  # Multi-band
            profile.update(count=data.shape[2], dtype=data.dtype)
        else:
            raise ValueError("Data must be a 2D or 3D array.")

        with rasterio.open(file_path, "w", **profile) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                # Transpose from (height, width, bands) to
                # (bands, height, width) for writing
                data_to_write = np.transpose(data, (2, 0, 1))
                for i in range(data_to_write.shape[0]):
                    dst.write(data_to_write[i], i + 1)
        print(f"Successfully wrote GeoTIFF to {file_path}")
    except Exception as e:
        print(f"Error writing GeoTIFF file {file_path}: {e}")


# --- Data Preparation for Random Forest ---


def create_rf_training_data(pre_flood_img, post_flood_img, labels):
    """
    Creates a feature array (X) and label vector (y) for training a
    Random Forest model.
    This function flattens the images and samples pixels for training.

    Args:
        pre_flood_img (np.ndarray): Pre-flood SAR image (VV, VH).
        post_flood_img (np.ndarray): Post-flood SAR image (VV, VH).
        labels (np.ndarray): Ground truth label mask
        (1 for flood, 0 for non-flood).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The feature matrix X with shape
            (n_samples, n_features).
            - np.ndarray: The label vector y with shape (n_samples,).
    """
    # Calculate change detection features
    # Note: Images should be in dB scale for subtraction to be
    # meaningful as a ratio
    diff_img = post_flood_img - pre_flood_img

    # Stack all features into a single multi-band image
    # Features: pre_VV, pre_VH, post_VV, post_VH, diff_VV, diff_VH
    features = np.dstack((pre_flood_img, post_flood_img, diff_img))

    # Reshape features and labels into 1D arrays
    X = features.reshape(-1, features.shape[2])
    y = labels.reshape(-1)

    # Filter out no-data pixels if any (assuming no-data value
    # is e.g., -9999 or 0 in labels)
    # For this example, we assume all labeled pixels are valid.

    # Subsample the data to make training manageable
    # This is crucial as a full image can have millions of pixels
    X, y = shuffle(X, y, random_state=config.RANDOM_SEED)

    n_samples = min(len(y), config.RF_MAX_SAMPLES)

    return X[:n_samples], y[:n_samples]


# --- Data Preparation for U-Net ---


class UNetDataGenerator(Sequence):
    """
    Custom Keras data generator for loading image chips for U-Net training.
    This is memory-efficient as it loads data in batches from disk.
    Assumes data is structured like Sen1Floods11:
    - training_data/
      - s1_pre/
      - s1_post/
      - labels/
    """

    def __init__(self, image_dir, label_dir, batch_size, dim, n_channels, shuffle=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.image_ids = [f.split(".")[0] for f in os.listdir(image_dir)]
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_ids_temp = [self.image_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)
        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        "Generates data containing batch_size samples"
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, ID in enumerate(list_ids_temp):
            # Load S1 image (assuming it's a 2-channel VV/VH tif)
            img_path = os.path.join(self.image_dir, f"{ID}.tif")
            img, _ = read_geotiff(img_path)

            # Load label mask
            label_path = os.path.join(self.label_dir, f"{ID}.tif")
            label, _ = read_geotiff(label_path)

            # Sen1Floods11 labels: 1 for water, 0 for non-water, -1 for no-data
            # We will treat no-data as non-water (0) for simplicity in
            # binary classification
            label[label == -1] = 0

            X[i,] = img
            y[i,] = np.expand_dims(label, axis=-1)

        return X, y


def get_unet_data_generators(train_dir, val_dir, batch_size):
    """
    Creates training and validation data generators for the U-Net model.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        batch_size (int): The batch size for the generators.

    Returns:
        tuple: A tuple containing (train_generator, validation_generator).
    """
    train_image_dir = os.path.join(train_dir, "s1_post")
    train_label_dir = os.path.join(train_dir, "labels")
    val_image_dir = os.path.join(val_dir, "s1_post")
    val_label_dir = os.path.join(val_dir, "labels")

    train_gen = UNetDataGenerator(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        batch_size=batch_size,
        dim=(config.IMG_HEIGHT, config.IMG_WIDTH),
        n_channels=config.IMG_CHANNELS,
    )

    val_gen = UNetDataGenerator(
        image_dir=val_image_dir,
        label_dir=val_label_dir,
        batch_size=batch_size,
        dim=(config.IMG_HEIGHT, config.IMG_WIDTH),
        n_channels=config.IMG_CHANNELS,
    )

    return train_gen, val_gen

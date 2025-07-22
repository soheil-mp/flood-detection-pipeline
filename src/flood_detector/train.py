"""
train.py

Main script for training a flood detection model.
This script can train either a Random Forest model
(for rapid, baseline results)
or a U-Net model (for high-accuracy, state-of-the-art results).

The choice of model is determined by command-line arguments.

Usage for Random Forest:
    python src/flood_detector/train.py \
        --model_type random_forest \
        --pre_flood_path path/to/pre_flood.tif \
        --post_flood_path path/to/post_flood.tif \
        --label_path path/to/labels.tif \
        --model_output_path results/models/rf_model.joblib

Usage for U-Net:
    python src/flood_detector/train.py \
        --model_type unet \
        --training_data_dir path/to/unet_training_data/ \
        --model_output_path results/models/unet_model.h5
"""

import argparse
import os

from flood_detector import config
from flood_detector.data_handler import (
    read_geotiff,
    create_rf_training_data,
    get_unet_data_generators,
)
from flood_detector.models.random_forest import train_rf_model, save_rf_model
from flood_detector.models.unet import build_unet_model


def train_random_forest(args):
    """
    Manages the training workflow for the Random Forest model.
    """
    print("--- Starting Random Forest Training Workflow ---")

    # 1. Load data
    print("Loading data...")
    pre_flood_img, _ = read_geotiff(args.pre_flood_path)
    post_flood_img, _ = read_geotiff(args.post_flood_path)
    labels, _ = read_geotiff(args.label_path)

    if pre_flood_img is None or post_flood_img is None or labels is None:
        raise ValueError("Failed to load one or more input images.")

    # 2. Prepare training data
    print("Preparing training data...")
    X, y = create_rf_training_data(pre_flood_img, post_flood_img, labels)

    # 3. Train model
    model = train_rf_model(X, y)

    # 4. Save model
    save_rf_model(model, args.model_output_path)
    print(
        f"Random Forest model training complete. \
            Model saved to {args.model_output_path}"
    )


def train_unet(args):
    """
    Manages the training workflow for the U-Net model.
    """
    print("--- Starting U-Net Training Workflow ---")

    # 1. Prepare data generators
    print("Preparing data generators...")
    # Assume the training data dir contains 'train' and 'val' subdirectories
    train_dir = os.path.join(args.training_data_dir, "train")
    val_dir = os.path.join(args.training_data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Training data directory must contain 'train' and \
                'val' subdirectories. Check path: {args.training_data_dir}"
        )

    train_generator, val_generator = get_unet_data_generators(
        train_dir, val_dir, args.batch_size
    )

    # 2. Build model
    model = build_unet_model()

    # 3. Train model
    print("Starting U-Net model training...")
    history = model.fit(
        train_generator, validation_data=val_generator, epochs=args.epochs, verbose=1
    )
    history.history["val_loss"][-1]  # Access last validation loss for logging

    # 4. Save model
    print(f"Saving U-Net model to {args.model_output_path}...")
    os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)
    model.save(args.model_output_path)
    print(
        f"U-Net model training complete. \
        Model saved to {args.model_output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flood detection model.")

    # --- Common Arguments ---
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["random_forest", "unet"],
        help="The type of model to train.",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        required=True,
        help="Path to save the trained model file.",
    )

    # --- Random Forest Specific Arguments ---
    rf_group = parser.add_argument_group("Random Forest Arguments")
    rf_group.add_argument(
        "--pre_flood_path", type=str, help="Path to the pre-flood GeoTIFF image."
    )
    rf_group.add_argument(
        "--post_flood_path", type=str, help="Path to the post-flood GeoTIFF image."
    )
    rf_group.add_argument(
        "--label_path", type=str, help="Path to the ground truth label GeoTIFF."
    )

    # --- U-Net Specific Arguments ---
    unet_group = parser.add_argument_group("U-Net Arguments")
    unet_group.add_argument(
        "--training_data_dir",
        type=str,
        help="Path to the root directory of \
            U-Net training data (e.g., Sen1Floods11).",
    )
    unet_group.add_argument(
        "--epochs",
        type=int,
        default=config.UNET_EPOCHS,
        help="Number of epochs to train the U-Net model.",
    )
    unet_group.add_argument(
        "--batch_size",
        type=int,
        default=config.UNET_BATCH_SIZE,
        help="Batch size for U-Net training.",
    )

    args = parser.parse_args()

    # --- Execute Training Workflow ---
    if args.model_type == "random_forest":
        # Check for required RF args
        if not all([args.pre_flood_path, args.post_flood_path, args.label_path]):
            parser.error(
                "For --model_type random_forest, you must provide \
                    --pre_flood_path, --post_flood_path, and --label_path."
            )
        train_random_forest(args)
    elif args.model_type == "unet":
        # Check for required U-Net args
        if not args.training_data_dir:
            parser.error(
                "For --model_type unet, you must \
                provide --training_data_dir."
            )
        train_unet(args)

"""
predict.py

This script runs inference using a pre-trained flood detection model.
It takes a preprocessed satellite image as input and produces a binary
flood map as a GeoTIFF output.

It supports both Random Forest and U-Net models.

Usage for Random Forest:
    python src/flood_detector/predict.py \
        --model_path results/models/rf_model.joblib \
        --pre_flood_image path/to/pre_flood_processed.tif \
        --post_flood_image path/to/post_flood_processed.tif \
        --output_map results/maps/rf_flood_map.tif

Usage for U-Net:
    python src/flood_detector/predict.py \
        --model_path results/models/unet_model.h5 \
        --post_flood_image path/to/post_flood_processed.tif \
        --output_map results/maps/unet_flood_map.tif
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model

from flood_detector.data_handler import read_geotiff, write_geotiff
from flood_detector.models.random_forest import load_rf_model


def predict_random_forest(args):
    """
    Generates a flood map using a trained Random Forest model.
    """
    print("--- Running Inference with Random Forest Model ---")

    # 1. Load model
    model = load_rf_model(args.model_path)

    # 2. Load images
    print("Loading input images...")
    pre_flood_img, profile = read_geotiff(args.pre_flood_image)
    post_flood_img, _ = read_geotiff(args.post_flood_image)

    if pre_flood_img is None or post_flood_img is None:
        raise ValueError("Failed to load one or more input images.")

    original_shape = pre_flood_img.shape[:2]

    # 3. Prepare feature data for prediction
    print("Preparing feature data...")
    diff_img = post_flood_img - pre_flood_img
    features = np.dstack((pre_flood_img, post_flood_img, diff_img))
    X_predict = features.reshape(-1, features.shape[2])

    # 4. Run prediction
    print("Predicting flood map... (This may take a while for large images)")
    y_pred = model.predict(X_predict)

    # 5. Reshape prediction back to image dimensions
    flood_map = y_pred.reshape(original_shape)

    # 6. Save the output map
    print("Saving output flood map...")
    write_geotiff(flood_map.astype(np.uint8), profile, args.output_map)
    print(f"Flood map saved to {args.output_map}")


def predict_unet(args):
    """
    Generates a flood map using a trained U-Net model.
    """
    print("--- Running Inference with U-Net Model ---")

    # 1. Load model
    # Note: Custom objects might be needed if custom metrics/layers were used.
    # For a standard U-Net, this should work directly.
    print("Loading U-Net model...")
    model = load_model(
        args.model_path, compile=False
    )  # Set compile=False for faster loading

    # 2. Load image
    print("Loading input image...")
    post_flood_img, profile = read_geotiff(args.post_flood_image)
    if post_flood_img is None:
        raise ValueError("Failed to load input image.")

    # 3. Preprocess image for prediction (normalize, add batch dimension)
    # Assuming the model was trained on normalized data (e.g., min-max scaled)
    # Here we just add the batch dimension.
    # A more robust pipeline would apply the exact same scaling as in training.
    img_for_pred = np.expand_dims(post_flood_img, axis=0)

    # 4. Run prediction
    print("Predicting flood map...")
    pred_mask = model.predict(img_for_pred)

    # 5. Post-process the prediction
    # Squeeze the batch dimension and apply a threshold (0.5 for sigmoid)
    pred_mask = (pred_mask[0, :, :, 0] > 0.5).astype(np.uint8)

    # 6. Save the output map
    print("Saving output flood map...")
    write_geotiff(pred_mask, profile, args.output_map)
    print(f"Flood map saved to {args.output_map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a flood map from a trained model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.joblib for RF, .h5 for U-Net).",
    )
    parser.add_argument(
        "--post_flood_image",
        type=str,
        required=True,
        help="Path to the preprocessed post-flood GeoTIFF image.",
    )
    parser.add_argument(
        "--output_map",
        type=str,
        required=True,
        help="Path to save the output flood map GeoTIFF.",
    )

    # RF-specific argument
    parser.add_argument(
        "--pre_flood_image",
        type=str,
        help="Path to the preprocessed pre-flood \
            GeoTIFF image (Required for Random Forest).",
    )

    args = parser.parse_args()

    # --- Determine model type and execute ---
    if args.model_path.endswith(".joblib"):
        if not args.pre_flood_image:
            parser.error(
                "For a Random Forest model (.joblib), \
                    --pre_flood_image is required."
            )
        predict_random_forest(args)
    elif args.model_path.endswith(".h5"):
        predict_unet(args)
    else:
        parser.error(
            "Could not determine model type from file extension. \
                Use .joblib for RF and .h5 for U-Net."
        )

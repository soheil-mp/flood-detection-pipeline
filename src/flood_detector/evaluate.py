"""
evaluate.py

This script evaluates the performance of a generated flood map by
comparing it against a ground truth label mask.

It calculates and reports key metrics for imbalanced segmentation tasks:
- Intersection over Union (IoU)
- F1-Score
- Precision
- Recall

Usage:
    python src/flood_detector/evaluate.py \
        --predicted_map path/to/your/predicted_map.tif \
        --ground_truth_map path/to/ground_truth_labels.tif
"""

import argparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from flood_detector.data_handler import read_geotiff


def calculate_metrics(y_true, y_pred):
    """
    Calculates and prints key performance metrics from flattened arrays.

    Args:
        y_true (np.ndarray): A 1D array of ground truth labels.
        y_pred (np.ndarray): A 1D array of predicted labels.
    """
    # Ensure arrays are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    print("--- Evaluating Model Performance ---")

    # --- Confusion Matrix ---
    # Rows are true labels, columns are predicted labels
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    print("-" * 30)

    # --- Core Metrics ---
    # Precision: Of all positive predictions, how many were correct?
    # High precision -> Low false positive rate
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Recall (Sensitivity): Of all actual positives, how many did we find?
    # High recall -> Low false negative rate (crucial for disaster response)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # F1-Score: Harmonic mean of precision and recall
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Intersection over Union (IoU) / Jaccard Score:
    # The gold standard for segmentation
    # IoU = TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    print("Key Performance Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Intersection over Union (IoU): {iou:.4f}")
    print("-" * 30)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a predicted flood map \
            against a ground truth map."
    )

    parser.add_argument(
        "--predicted_map",
        type=str,
        required=True,
        help="Path to the predicted binary flood \
            map GeoTIFF (1 for flood, 0 for non-flood).",
    )
    parser.add_argument(
        "--ground_truth_map",
        type=str,
        required=True,
        help="Path to the ground truth label GeoTIFF \
            (1 for flood, 0 for non-flood).",
    )

    args = parser.parse_args()

    # 1. Load the maps
    print("Loading maps for evaluation...")
    predicted_data, _ = read_geotiff(args.predicted_map)
    ground_truth_data, _ = read_geotiff(args.ground_truth_map)

    if predicted_data is None or ground_truth_data is None:
        raise ValueError("Could not load one or both of the maps.")

    # Handle multi-band vs single-band images if necessary
    if predicted_data.ndim == 3:
        predicted_data = predicted_data[:, :, 0]
    if ground_truth_data.ndim == 3:
        ground_truth_data = ground_truth_data[:, :, 0]

    # 2. Check shapes
    if predicted_data.shape != ground_truth_data.shape:
        raise ValueError(
            f"Shape mismatch: Predicted map is {predicted_data.shape} \
                but ground truth is {ground_truth_data.shape}."
        )

    # 3. Calculate metrics
    # The function will print the results
    calculate_metrics(ground_truth_data, predicted_data)

    print("Evaluation complete.")

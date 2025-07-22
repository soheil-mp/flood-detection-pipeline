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

from . import config
from .data_handler import (
    read_geotiff,
    create_rf_training_data,
    get_unet_data_generators,
)
from .models.random_forest import train_rf_model, save_rf_model
from .models.unet import build_unet_model


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
    Manages the enhanced training workflow for the U-Net model with
    robust training practices. Includes gradient clipping, learning rate
    scheduling, early stopping, and NaN monitoring.
    """
    print("--- Starting Enhanced U-Net Training Workflow ---")

    # Import required modules for advanced training
    from tensorflow.keras import callbacks
    import numpy as np
    from .training_monitor import (
        TrainingDiagnostics,
        AdvancedTrainingCallbacks,
        create_training_visualization,
    )
    from .inference_callback import create_inference_callback

    # 1. Prepare data generators
    print("Preparing data generators...")
    train_dir = os.path.join(args.training_data_dir, "train")
    val_dir = os.path.join(args.training_data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Training data directory must contain 'train' and 'val' "
            f"subdirectories. Check path: {args.training_data_dir}"
        )

    train_generator, val_generator = get_unet_data_generators(
        train_dir, val_dir, args.batch_size
    )

    # 2. Run comprehensive diagnostics
    print("Running pre-training diagnostics...")

    # Check data quality
    train_diagnostics = TrainingDiagnostics.check_data_quality(train_generator)
    val_diagnostics = TrainingDiagnostics.check_data_quality(val_generator, 3)

    if train_diagnostics["issues"] or val_diagnostics["issues"]:
        print(
            "WARNING: Data quality issues \
            detected. Proceeding with caution."
        )
        for issue in train_diagnostics["issues"] + val_diagnostics["issues"]:
            print(f"  - {issue}")

    # 3. Build enhanced model with better stability
    model = build_unet_model()

    # Test model forward pass
    forward_test = TrainingDiagnostics.test_model_forward_pass(model, train_generator)
    if not forward_test["success"]:
        raise RuntimeError(
            f"Model forward pass failed: "
            f"{forward_test.get('error', 'Unknown error')}"
        )

    # 4. Setup advanced training callbacks for stability and monitoring
    print("Setting up enhanced training callbacks...")

    # Create logs directory
    log_dir = os.path.join(config.RESULTS_DIR, "training_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create TensorBoard subdirectories to prevent directory errors
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(os.path.join(tensorboard_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(tensorboard_dir, "validation"), exist_ok=True)

    # Create CSV file for logger
    csv_log_path = os.path.join(log_dir, "training_metrics.csv")

    # Ensure CSV file can be created by touching it
    with open(csv_log_path, "w") as f:
        f.write("")  # Create empty file

    # Create sample inference directory
    sample_inference_dir = os.path.join(log_dir, "sample_inference")
    os.makedirs(sample_inference_dir, exist_ok=True)

    # Enhanced callbacks
    callback_list = [
        # Early stopping with best weights restoration
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.UNET_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate reduction on plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.UNET_LR_REDUCTION_FACTOR,
            patience=config.UNET_LR_REDUCTION_PATIENCE,
            min_lr=config.UNET_MIN_LEARNING_RATE,
            verbose=1,
        ),
        # Model checkpoint for best model
        callbacks.ModelCheckpoint(
            args.model_output_path.replace(".h5", "_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # CSV logger for detailed metrics
        callbacks.CSVLogger(csv_log_path, append=False),
        # TensorBoard for visualization
        callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
        ),
        # Advanced monitoring callbacks
        AdvancedTrainingCallbacks.get_nan_terminator(),
        AdvancedTrainingCallbacks.get_comprehensive_logger(log_dir),
        # Sample inference callback for visual monitoring
        create_inference_callback(
            validation_generator=val_generator,
            output_dir=sample_inference_dir,
            num_samples=4,
            save_frequency=1,
        ),
    ]

    # 5. Train model with enhanced monitoring
    print(
        "Starting enhanced U-Net model training with \
        stability monitoring..."
    )
    print(
        f"Training for up to {args.epochs} epochs with \
            batch size {args.batch_size}"
    )
    print(f"Logs will be saved to: {log_dir}")

    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=callback_list,
            verbose=1,
        )

        # Check if training completed successfully
        if len(history.history["loss"]) == 0:
            raise ValueError("Training failed - no history recorded")

        # Log training results
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]

        print("\nTraining Summary:")
        print(f"- Final training loss: {final_train_loss:.6f}")
        print(f"- Final validation loss: {final_val_loss:.6f}")
        print(f"- Total epochs completed: {len(history.history['loss'])}")

        # Check for potential issues
        if np.isnan(final_train_loss) or np.isnan(final_val_loss):
            print(
                "WARNING: Final loss values contain NaN. \
                Model may be unstable."
            )

        # Create training visualization
        viz_path = os.path.join(log_dir, "training_history.png")
        create_training_visualization(history, viz_path)

    except Exception as e:
        print(f"ERROR during training: {e}")
        print("Attempting to save current model state...")
        try:
            emergency_path = args.model_output_path.replace(".h5", "_emergency.h5")
            model.save(emergency_path)
            print(f"Emergency model saved to: {emergency_path}")
        except Exception as save_error:
            print(f"Failed to save emergency model state: {save_error}")
        raise

    # 6. Save final model and artifacts
    print(f"Saving final U-Net model to {args.model_output_path}...")
    os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)

    try:
        model.save(args.model_output_path)
        print(
            f"U-Net model training complete. "
            f"Model saved to {args.model_output_path}"
        )

        # Save training history for analysis
        history_path = args.model_output_path.replace(".h5", "_history.npy")
        np.save(history_path, history.history)
        print(f"Training history saved to {history_path}")

        # Save final training report
        report_path = os.path.join(log_dir, "final_report.txt")
        with open(report_path, "w") as f:
            f.write("U-Net Training Final Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model saved to: {args.model_output_path}\n")
            f.write(f"Training epochs: {len(history.history['loss'])}\n")
            f.write(f"Final training loss: {final_train_loss:.6f}\n")
            f.write(f"Final validation loss: {final_val_loss:.6f}\n")
            f.write(
                f"Best validation loss: \
                    "
                f"{min(history.history['val_loss']):.6f}\n"
            )
            training_completed_flag = (
                "Yes"
                if not (np.isnan(final_train_loss) or np.isnan(final_val_loss))
                else "No"
            )
            f.write(
                f"\nTraining completed successfully: " f"{training_completed_flag}\n"
            )

        print(f"Final training report saved to: {report_path}")

    except Exception as e:
        print(f"ERROR saving model: {e}")
        raise


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

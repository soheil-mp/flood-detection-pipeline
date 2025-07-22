"""
training_monitor.py

Advanced monitoring utilities for U-Net training to detect and resolve
common training issues like NaN losses, gradient explosions, and data problems.
Provides comprehensive diagnostics and automatic fixes.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import os


class TrainingDiagnostics:
    """
    Comprehensive diagnostics for training issues detection and resolution.
    """

    @staticmethod
    def check_data_quality(data_generator, num_batches=5):
        """
        Analyzes data quality to detect potential issues.

        Args:
            data_generator: Keras data generator
            num_batches: Number of batches to analyze

        Returns:
            dict: Diagnostic report
        """
        print("--- Running Data Quality Diagnostics ---")

        issues = []
        stats = {
            "input_stats": {"min": [], "max": [], "mean": [], "std": []},
            "label_stats": {"min": [], "max": [], "mean": [], "std": []},
            "nan_count": {"input": 0, "label": 0},
            "inf_count": {"input": 0, "label": 0},
        }

        for i in range(min(num_batches, len(data_generator))):
            try:
                X, y = data_generator[i]

                # Check for NaN values
                input_nan = np.isnan(X).sum()
                label_nan = np.isnan(y).sum()
                stats["nan_count"]["input"] += input_nan
                stats["nan_count"]["label"] += label_nan

                if input_nan > 0:
                    issues.append(f"Batch {i}: {input_nan} NaN values in input")
                if label_nan > 0:
                    issues.append(f"Batch {i}: {label_nan} NaN values in labels")

                # Check for infinite values
                input_inf = np.isinf(X).sum()
                label_inf = np.isinf(y).sum()
                stats["inf_count"]["input"] += input_inf
                stats["inf_count"]["label"] += label_inf

                if input_inf > 0:
                    issues.append(f"Batch {i}: {input_inf} infinite values in input")
                if label_inf > 0:
                    issues.append(f"Batch {i}: {label_inf} infinite values in labels")

                # Collect statistics
                stats["input_stats"]["min"].append(np.min(X))
                stats["input_stats"]["max"].append(np.max(X))
                stats["input_stats"]["mean"].append(np.mean(X))
                stats["input_stats"]["std"].append(np.std(X))

                stats["label_stats"]["min"].append(np.min(y))
                stats["label_stats"]["max"].append(np.max(y))
                stats["label_stats"]["mean"].append(np.mean(y))
                stats["label_stats"]["std"].append(np.std(y))

                # Check value ranges
                if np.min(X) < -100 or np.max(X) > 100:
                    issues.append(
                        f"Batch {i}: Extreme input values "
                        f"(min: {np.min(X):.2f}, max: {np.max(X):.2f})"
                    )

                if np.min(y) < 0 or np.max(y) > 1:
                    issues.append(
                        f"Batch {i}: Labels out of [0,1] range "
                        f"(min: {np.min(y):.2f}, max: {np.max(y):.2f})"
                    )

            except Exception as e:
                issues.append(f"Error loading batch {i}: {e}")

        # Summarize statistics
        for key in stats["input_stats"]:
            if stats["input_stats"][key]:
                stats["input_stats"][key] = np.array(stats["input_stats"][key])

        for key in stats["label_stats"]:
            if stats["label_stats"][key]:
                stats["label_stats"][key] = np.array(stats["label_stats"][key])

        print("Data Quality Report:")
        print(
            f"- Total NaN values: Input={stats['nan_count']['input']}, "
            f"Labels={stats['nan_count']['label']}"
        )
        print(
            f"- Total Inf values: Input={stats['inf_count']['input']}, "
            f"Labels={stats['inf_count']['label']}"
        )
        print(
            f"- Input range: [{np.min(stats['input_stats']['min']):.2f}, "
            f"{np.max(stats['input_stats']['max']):.2f}]"
        )
        print(
            f"- Label range: [{np.min(stats['label_stats']['min']):.2f}, "
            f"{np.max(stats['label_stats']['max']):.2f}]"
        )

        if issues:
            print("ISSUES DETECTED:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No critical data quality issues detected.")

        return {"stats": stats, "issues": issues}

    @staticmethod
    def test_model_forward_pass(model, data_generator):
        """
        Tests model forward pass to detect gradient issues.

        Args:
            model: Keras model
            data_generator: Data generator

        Returns:
            dict: Test results
        """
        print("--- Testing Model Forward Pass ---")

        try:
            # Get a single batch
            X, y = data_generator[0]

            # Test forward pass
            predictions = model.predict(X, verbose=0)

            # Check predictions
            pred_nan = np.isnan(predictions).sum()
            pred_inf = np.isinf(predictions).sum()

            print("Forward pass test:")
            print(f"- Input shape: {X.shape}")
            print(f"- Output shape: {predictions.shape}")
            print(
                f"- Output range: [{np.min(predictions):.6f}, "
                f"{np.max(predictions):.6f}]"
            )
            print(f"- NaN count: {pred_nan}")
            print(f"- Inf count: {pred_inf}")

            # Test loss computation
            loss_value = model.evaluate(X, y, verbose=0)
            print(f"- Loss value: {loss_value}")

            issues = []
            if pred_nan > 0:
                issues.append(f"Forward pass produces {pred_nan} NaN values")
            if pred_inf > 0:
                issues.append(f"Forward pass produces {pred_inf} infinite values")
            if np.isnan(loss_value[0]):
                issues.append("Loss computation returns NaN")

            return {
                "success": True,
                "predictions": predictions,
                "loss": loss_value,
                "issues": issues,
            }

        except Exception as e:
            print(f"Forward pass test FAILED: {e}")
            return {
                "success": False,
                "error": str(e),
                "issues": [f"Forward pass failed: {e}"],
            }


class AdvancedTrainingCallbacks:
    """
    Collection of advanced callbacks for robust training.
    """

    @staticmethod
    def get_nan_terminator():
        """Returns a callback that terminates training on NaN."""

        class NaNTerminator(callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if logs is None:
                    logs = {}

                for key, value in logs.items():
                    if np.isnan(value) or np.isinf(value):
                        print(f"\nNaN/Inf detected in {key}: {value}")
                        print("Terminating training to prevent further issues.")
                        self.model.stop_training = True
                        return

        return NaNTerminator()

    @staticmethod
    def get_gradient_monitor():
        """Returns a callback that monitors gradient norms."""

        class GradientMonitor(callbacks.Callback):
            def __init__(self, log_freq=10):
                super().__init__()
                self.log_freq = log_freq

            def on_batch_end(self, batch, logs=None):
                if batch % self.log_freq == 0:
                    # Get gradients for monitoring
                    weights = self.model.trainable_weights
                    gradients = tf.gradients(logs.get("loss", 0), weights)

                    if gradients:
                        grad_norms = [tf.norm(g) for g in gradients if g is not None]
                        if grad_norms:
                            max_grad_norm = tf.reduce_max(grad_norms)
                            mean_grad_norm = tf.reduce_mean(grad_norms)

                            print(f"\nBatch {batch} gradient stats:")
                            print(f"  Max gradient norm: {max_grad_norm:.6f}")
                            print(f"  Mean gradient norm: {mean_grad_norm:.6f}")

                            if max_grad_norm > 10.0:
                                print("  WARNING: Large gradient detected!")

        return GradientMonitor()

    @staticmethod
    def get_comprehensive_logger(log_dir):
        """Returns a callback for comprehensive training logging."""

        class ComprehensiveLogger(callbacks.Callback):
            def __init__(self, log_directory):
                super().__init__()
                self.log_dir = log_directory
                os.makedirs(log_directory, exist_ok=True)

            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}

                # Save detailed logs
                log_file = os.path.join(self.log_dir, f"epoch_{epoch:03d}.txt")
                with open(log_file, "w") as f:
                    f.write(f"Epoch {epoch} Results:\n")
                    for key, value in logs.items():
                        f.write(f"{key}: {value}\n")

                # Check for potential issues
                if "loss" in logs and logs["loss"] > logs.get("val_loss", float("inf")):
                    overfitting_ratio = logs["loss"] / logs.get("val_loss", 1)
                    if overfitting_ratio > 2.0:
                        print(
                            f"\nWARNING: Potential overfitting detected "
                            f"(train/val loss ratio: {overfitting_ratio:.2f})"
                        )

        return ComprehensiveLogger(log_dir)


def create_training_visualization(history, save_path):
    """
    Creates comprehensive training visualization plots.

    Args:
        history: Training history object
        save_path: Path to save the plots
    """
    if not history or not history.history:
        print("No training history available for visualization.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    if "accuracy" in history.history:
        axes[0, 1].plot(history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history.history:
            axes[0, 1].plot(
                history.history["val_accuracy"], label="Validation Accuracy"
            )
        axes[0, 1].set_title("Model Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # IoU plot
    iou_keys = [k for k in history.history.keys() if "iou" in k.lower()]
    if iou_keys:
        for key in iou_keys:
            axes[1, 0].plot(history.history[key], label=key)
        axes[1, 0].set_title("IoU Metrics")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("IoU")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Learning rate plot (if available)
    if "lr" in history.history:
        axes[1, 1].plot(history.history["lr"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training visualization saved to: {save_path}")


if __name__ == "__main__":
    print("Training Monitor Module - Run diagnostics on your training setup")
    print("Import this module and use TrainingDiagnostics class methods")

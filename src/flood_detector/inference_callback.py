"""
inference_callback.py

Custom Keras callback for running sample inference during training.
Generates flood maps from validation samples after each epoch to monitor
training progress visually.
"""

import numpy as np
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import os
import cv2


class SampleInferenceCallback(callbacks.Callback):
    """
    Custom callback that runs inference on sample validation images
    after each epoch and saves the results for visual monitoring.
    """

    def __init__(
        self,
        validation_generator,
        output_dir,
        num_samples=4,
        save_frequency=1,
        threshold=0.5,
    ):
        """
        Initialize the sample inference callback.

        Args:
            validation_generator: Validation data generator
            output_dir: Directory to save inference results
            num_samples: Number of samples to run inference on
            save_frequency: Run inference every N epochs (1 = every epoch)
            threshold: Threshold for binary classification (default 0.5)
        """
        super().__init__()
        self.validation_generator = validation_generator
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.save_frequency = save_frequency
        self.threshold = threshold

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get sample data once at initialization
        self._prepare_sample_data()

    def _prepare_sample_data(self):
        """Prepare sample data for inference."""
        print(
            f"Preparing {self.num_samples} sample images for "
            f"inference monitoring..."
        )

        # Get the first batch from validation generator
        sample_batch = self.validation_generator[0]
        self.sample_images, self.sample_labels = sample_batch

        # Select specified number of samples
        actual_samples = min(self.num_samples, self.sample_images.shape[0])
        self.sample_images = self.sample_images[:actual_samples]
        self.sample_labels = self.sample_labels[:actual_samples]

        print(f"Selected {actual_samples} samples for inference monitoring")

    def on_epoch_end(self, epoch, logs=None):
        """Run inference and save results after each epoch."""
        if (epoch + 1) % self.save_frequency != 0:
            return

        try:
            print(f"\n--- Running Sample Inference (Epoch {epoch + 1}) ---")

            # Run inference on sample images
            predictions = self.model.predict(self.sample_images, verbose=0)

            # Create epoch-specific directory
            epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Process each sample
            for i in range(len(self.sample_images)):
                self._save_sample_inference(
                    epoch + 1,
                    i,
                    self.sample_images[i],
                    self.sample_labels[i],
                    predictions[i],
                    epoch_dir,
                )

            # Create summary visualization
            self._create_epoch_summary(epoch + 1, epoch_dir, logs)

            print(f"Sample inference results saved to: {epoch_dir}")

        except Exception as e:
            print(f"Error during sample inference: {e}")

    def _save_sample_inference(
        self, epoch, sample_idx, image, label, prediction, save_dir
    ):
        """Save individual sample inference result."""

        # Convert prediction to binary mask
        pred_binary = (prediction[:, :, 0] > self.threshold).astype(np.uint8)

        # Convert label to binary (handle label smoothing)
        label_binary = (label[:, :, 0] > 0.5).astype(np.uint8)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Input image (show first 2 channels if multi-channel)
        if image.shape[2] >= 2:
            # For SAR data, show VV and VH polarizations
            axes[0, 0].imshow(image[:, :, 0], cmap="gray")
            axes[0, 0].set_title("Input Channel 1 (VV)")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(image[:, :, 1], cmap="gray")
            axes[0, 1].set_title("Input Channel 2 (VH)")
            axes[0, 1].axis("off")
        else:
            axes[0, 0].imshow(image[:, :, 0], cmap="gray")
            axes[0, 0].set_title("Input Image")
            axes[0, 0].axis("off")

            axes[0, 1].axis("off")

        # Ground truth
        axes[0, 2].imshow(label_binary, cmap="Blues", vmin=0, vmax=1)
        axes[0, 2].set_title("Ground Truth")
        axes[0, 2].axis("off")

        # Prediction (continuous)
        axes[1, 0].imshow(prediction[:, :, 0], cmap="Reds", vmin=0, vmax=1)
        axes[1, 0].set_title("Prediction (Raw)")
        axes[1, 0].axis("off")

        # Prediction (binary)
        axes[1, 1].imshow(pred_binary, cmap="Reds", vmin=0, vmax=1)
        axes[1, 1].set_title(f"Prediction (Binary, t={self.threshold})")
        axes[1, 1].axis("off")

        # Overlay comparison
        overlay = self._create_overlay(image[:, :, 0], label_binary, pred_binary)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Overlay (GT=Blue, Pred=Red)")
        axes[1, 2].axis("off")

        # Calculate metrics for this sample
        intersection = np.sum(label_binary * pred_binary)
        union = np.sum(label_binary) + np.sum(pred_binary) - intersection
        iou = intersection / (union + 1e-7)

        accuracy = np.mean(label_binary == pred_binary)

        # Add metrics text
        metrics_text = f"IoU: {iou:.3f}\nAccuracy: {accuracy:.3f}"
        fig.text(
            0.02,
            0.02,
            metrics_text,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

        plt.suptitle(f"Epoch {epoch} - Sample {sample_idx + 1}", fontsize=16)
        plt.tight_layout()

        # Save the figure
        sample_path = os.path.join(save_dir, f"sample_{sample_idx+1:02d}.png")
        plt.savefig(sample_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_overlay(self, base_image, ground_truth, prediction):
        """Create an RGB overlay image showing ground truth and prediction."""
        # Normalize base image
        base_norm = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Create RGB image
        overlay = np.stack([base_norm, base_norm, base_norm], axis=2)

        # Add ground truth in blue channel
        overlay[:, :, 2] = np.where(ground_truth == 1, 255, overlay[:, :, 2])

        # Add prediction in red channel
        overlay[:, :, 0] = np.where(prediction == 1, 255, overlay[:, :, 0])

        # Handle overlap (purple for true positives)
        overlap = (ground_truth == 1) & (prediction == 1)
        overlay[overlap] = [255, 0, 255]  # Purple

        return overlay

    def _create_epoch_summary(self, epoch, save_dir, logs):
        """Create summary visualization for the epoch."""

        # Run inference on all samples
        predictions = self.model.predict(self.sample_images, verbose=0)

        # Calculate metrics for all samples
        sample_metrics = []
        for i in range(len(self.sample_images)):
            label_binary = (self.sample_labels[i, :, :, 0] > 0.5).astype(np.uint8)
            pred_binary = (predictions[i, :, :, 0] > self.threshold).astype(np.uint8)

            intersection = np.sum(label_binary * pred_binary)
            union = np.sum(label_binary) + np.sum(pred_binary) - intersection
            iou = intersection / (union + 1e-7)
            accuracy = np.mean(label_binary == pred_binary)

            sample_metrics.append({"iou": iou, "accuracy": accuracy})

        # Create summary plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # IoU distribution
        ious = [m["iou"] for m in sample_metrics]
        axes[0].bar(range(1, len(ious) + 1), ious, color="skyblue", alpha=0.7)
        axes[0].set_title("Sample IoU Scores")
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("IoU")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)

        # Add mean line
        mean_iou = np.mean(ious)
        axes[0].axhline(
            y=mean_iou, color="red", linestyle="--", label=f"Mean IoU: {mean_iou:.3f}"
        )
        axes[0].legend()

        # Accuracy distribution
        accuracies = [m["accuracy"] for m in sample_metrics]
        axes[1].bar(
            range(1, len(accuracies) + 1), accuracies, color="lightgreen", alpha=0.7
        )
        axes[1].set_title("Sample Accuracy Scores")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # Add mean line
        mean_acc = np.mean(accuracies)
        axes[1].axhline(
            y=mean_acc, color="red", linestyle="--", label=f"Mean Acc: {mean_acc:.3f}"
        )
        axes[1].legend()

        # Add training metrics if available
        if logs:
            title = f"Epoch {epoch} Summary\n"
            title += f'Train Loss: {logs.get("loss", "N/A"):.4f}, '
            title += f'Val Loss: {logs.get("val_loss", "N/A"):.4f}'
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        # Save summary
        summary_path = os.path.join(save_dir, "epoch_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Save metrics to text file
        metrics_path = os.path.join(save_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Epoch {epoch} Sample Inference Metrics\n")
            f.write(f'{"="*40}\n')
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            f.write(f"Mean Accuracy: {mean_acc:.4f}\n")
            f.write("\nPer-sample metrics:\n")
            for i, metrics in enumerate(sample_metrics):
                f.write(
                    f'Sample {i+1}: IoU={metrics["iou"]:.4f}, '
                    f'Acc={metrics["accuracy"]:.4f}\n'
                )

            if logs:
                f.write("\nTraining metrics:\n")
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")


def create_inference_callback(
    validation_generator,
    output_dir="results/sample_inference",
    num_samples=4,
    save_frequency=1,
):
    """
    Factory function to create the sample inference callback.

    Args:
        validation_generator: Validation data generator
        output_dir: Directory to save inference results
        num_samples: Number of samples to process
        save_frequency: Save every N epochs

    Returns:
        SampleInferenceCallback: Configured callback
    """
    return SampleInferenceCallback(
        validation_generator=validation_generator,
        output_dir=output_dir,
        num_samples=num_samples,
        save_frequency=save_frequency,
    )


if __name__ == "__main__":
    print("Sample Inference Callback Module")
    print(
        "Use create_inference_callback() to \
        add visual monitoring to your training"
    )

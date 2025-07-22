"""
unet.py

This module defines the U-Net model architecture for semantic segmentation
of floodwater. The U-Net is a state-of-the-art deep learning model for
image segmentation tasks and is particularly effective for this application.

The implementation uses TensorFlow with the Keras API.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
from flood_detector import config


def conv_block(input_tensor, num_filters):
    """
    A single convolutional block used in the U-Net encoder and decoder.
    Consists of two 3x3 convolutions, each followed by a ReLU activation.
    """
    x = layers.Conv2D(
        num_filters, (3, 3), padding="same", kernel_initializer="he_normal"
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        num_filters, (3, 3), padding="same", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(input_tensor, num_filters):
    """
    An encoder block (downsampling path).
    Consists of a convolutional block followed by a max pooling operation.
    """
    conv = conv_block(input_tensor, num_filters)
    pool = layers.MaxPooling2D((2, 2))(conv)
    return conv, pool


def decoder_block(input_tensor, skip_features, num_filters):
    """
    A decoder block (upsampling path).
    Consists of an up-convolution, concatenation with skip features,
    and a convolutional block.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(
        input_tensor
    )
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
):
    """
    Builds the U-Net model architecture.

    Args:
        input_shape (tuple): The shape of the input images.

    Returns:
        tf.keras.Model: The compiled U-Net model.
    """
    print("--- Building U-Net Model ---")
    inputs = layers.Input(shape=input_shape)

    # --- Encoder Path ---
    # Block 1
    s1, p1 = encoder_block(inputs, 64)
    # Block 2
    s2, p2 = encoder_block(p1, 128)
    # Block 3
    s3, p3 = encoder_block(p2, 256)
    # Block 4
    s4, p4 = encoder_block(p3, 512)

    # --- Bridge ---
    b1 = conv_block(p4, 1024)

    # --- Decoder Path ---
    # Block 1
    d1 = decoder_block(b1, s4, 512)
    # Block 2
    d2 = decoder_block(d1, s3, 256)
    # Block 3
    d3 = decoder_block(d2, s2, 128)
    # Block 4
    d4 = decoder_block(d3, s1, 64)

    # --- Output Layer ---
    # A 1x1 convolution with a sigmoid activation
    # for binary classification (flood/non-flood)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d4)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    # --- Compile the Model ---
    # We use Adam optimizer and binary cross-entropy for the loss function.
    # Metrics can include accuracy, IoU, etc.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.UNET_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)],
    )

    print("--- U-Net Model Built and Compiled Successfully ---")
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage: build the model and print its summary
    print("--- Running U-Net Module Example ---")

    # Build the model with default parameters from config
    unet_model = build_unet_model()

    # The summary is printed inside the build function.
    # We can also save a plot of the model architecture.
    model_plot_path = os.path.join(config.RESULTS_DIR, "unet_model_architecture.png")
    os.makedirs(os.path.dirname(model_plot_path), exist_ok=True)
    tf.keras.utils.plot_model(unet_model, to_file=model_plot_path, show_shapes=True)
    print(f"Model architecture plot saved to {model_plot_path}")

    print("Example completed successfully.")

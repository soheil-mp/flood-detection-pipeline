"""
random_forest.py

This module defines the Random Forest model for flood detection.
It includes functions to create, train, and save the model using
scikit-learn. This model serves as a robust baseline and is suitable
for rapid response scenarios where deep learning may be too slow or
data-hungry.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import numpy as np

from flood_detector import config
from flood_detector.data_handler import create_rf_training_data


def build_rf_model():
    """
    Builds and returns a scikit-learn RandomForestClassifier with
    parameters defined in the config file.

    Returns:
        RandomForestClassifier: An untrained scikit-learn RF model.
    """
    rf_model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=config.RANDOM_SEED,
        n_jobs=config.RF_N_JOBS,
        verbose=2,  # Print progress during training
    )
    return rf_model


def train_rf_model(X, y):
    """
    Trains the Random Forest model on the provided data.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The label vector.

    Returns:
        RandomForestClassifier: The trained scikit-learn model.
    """
    print("--- Training Random Forest Model ---")

    # Split data into training and testing sets for internal validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_SEED
    )

    print(
        f"Training on {len(y_train)} samples, \
        validating on {len(y_test)} samples."
    )

    model = build_rf_model()
    model.fit(X_train, y_train)

    print("\n--- Model Training Complete ---")
    print("Internal Validation Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Non-Flood", "Flood"]))

    return model


def save_rf_model(model, file_path):
    """
    Saves the trained Random Forest model to a file using joblib.

    Args:
        model (RandomForestClassifier): The trained model to save.
        file_path (str): The path to save the model file.
    """
    print(f"Saving model to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print("Model saved successfully.")


def load_rf_model(file_path):
    """
    Loads a trained Random Forest model from a file.

    Args:
        file_path (str): Path to the saved model file.

    Returns:
        RandomForestClassifier: The loaded scikit-learn model.
    """
    print(f"Loading model from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")
    model = joblib.load(file_path)
    print("Model loaded successfully.")
    return model


if __name__ == "__main__":
    # Example usage: create and train a model on dummy data
    print("--- Running Random Forest Module Example ---")

    # This is a dummy example. In a real scenario, you would load actual data.
    # We create dummy pre, post, and label images.
    # In a real run, these would be loaded via `read_geotiff`.
    print("Generating dummy data for demonstration...")
    dummy_pre_img = np.random.rand(100, 100, 2)
    dummy_post_img = np.random.rand(100, 100, 2)
    dummy_labels = np.random.randint(0, 2, size=(100, 100))

    print("Creating training data from dummy images...")
    X_data, y_data = create_rf_training_data(
        dummy_pre_img, dummy_post_img, dummy_labels
    )

    # Train the model
    trained_model = train_rf_model(X_data, y_data)

    # Save the model
    model_path = os.path.join(config.MODELS_DIR, "rf_model_example.joblib")
    save_rf_model(trained_model, model_path)

    # Load the model back
    loaded_model = load_rf_model(model_path)
    print("Example completed successfully.")

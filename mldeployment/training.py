"""This module trains an MLP model built with Scikit-learn, for classifying
digits using the MNIST dataset.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import joblib


NUM_CLASSES = 10
NUM_EPOCHS = 2  # This will be converted to the `max_iter` parameter in scikit-learn's MLPClassifier
TEST_SIZE = 0.2  # 20% of the data will be used for testing


def prepare_mnist_training_data(test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepares training data for the model using train_test_split.

    Args:
        test_size: The proportion of the dataset to include in the test split.

    Returns:
        A tuple of four numpy arrays:
            - The training images,
            - The training labels,
            - The test images,
            - The test labels.
    """
    # Load MNIST dataset from OpenML
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images, labels = mnist['data'], mnist['target']

    # Convert labels to integers
    labels = labels.astype(np.int64)

    # Normalize the images to the range [0, 1]
    images /= 255.0

    # Split the data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )

    return train_images, train_labels, test_images, test_labels


def build_mlp_model(hidden_layer_sizes: Tuple[int, ...], num_epochs: int) -> MLPClassifier:
    """Builds a simple MLP model using scikit-learn.

    Args:
        hidden_layer_sizes: A tuple representing the number of neurons in each hidden layer.
        num_epochs: The maximum number of iterations to train.

    Returns:
        An MLPClassifier model.
    """
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=num_epochs, alpha=1e-4,
                          solver='adam', verbose=True, random_state=42)

    return model


def train_model(model: MLPClassifier, train_images: np.ndarray, train_labels: np.ndarray) -> MLPClassifier:
    """Trains a model.

    Args:
        model: The model to train.
        train_images: The training images.
        train_labels: The training labels.

    Returns:
        The trained model.
    """
    model.fit(train_images, train_labels)
    return model


def evaluate_model(model: MLPClassifier, test_images: np.ndarray, test_labels: np.ndarray) -> float:
    """Evaluates a model.

    Args:
        model: The model to evaluate.
        test_images: The test images.
        test_labels: The test labels.

    Returns:
        The accuracy of the model.
    """
    predictions = model.predict(test_images)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model: MLPClassifier, model_save_path: Path) -> None:
    """Saves a model to a file.

    Args:
        model: The model to save.
        model_save_path: The path to save the model to.
    """
    joblib.dump(model, model_save_path)


def main():
    """Trains a model for classifying digits using the MNIST dataset."""

    train_images, train_labels, test_images, test_labels = prepare_mnist_training_data(test_size=TEST_SIZE)

    model = build_mlp_model(hidden_layer_sizes=(128, 64), num_epochs=NUM_EPOCHS)

    model = train_model(model, train_images, train_labels)

    evaluate_model(model, test_images, test_labels)

    save_model(model, Path("mlp_model.joblib"))


if __name__ == "__main__":
    main()

"""CNN Model Architectures for E-Commerce Product Classification.

Contains both the Baseline CNN and the Improved CNN (with Batch Normalization,
Global Average Pooling 2D, and Dropout) for Fashion-MNIST.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from config.settings import (
    BEST_MODEL_PATH,
    BASELINE_MODEL_PATH,
    INPUT_SHAPE,
    NUM_CLASSES,
)

# Tentative d'import de TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
):
    """Construit l'architecture CNN Baseline (Référence) :
    Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(128) -> Dropout(0.3) -> Softmax(10).
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé dans l'environnement courant.")

    model = keras.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.30, name="dropout1"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
    ], name="CNN_Baseline")
    return model


def build_improved_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
):
    """Construit l'architecture CNN Amélioré :
    Blocs convolutifs profonds + Batch Normalization + GlobalAveragePooling2D + Dropout.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé dans l'environnement courant.")

    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Bloc 1 (32 filtres)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", name="block1_conv1")(inputs)
    x = layers.BatchNormalization(name="block1_bn1")(x)
    x = layers.Activation("relu", name="block1_act1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", name="block1_conv2")(x)
    x = layers.BatchNormalization(name="block1_bn2")(x)
    x = layers.Activation("relu", name="block1_act2")(x)
    x = layers.MaxPooling2D((2, 2), name="block1_pool")(x)
    x = layers.Dropout(0.25, name="block1_dropout")(x)

    # Bloc 2 (64 filtres)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", name="block2_conv1")(x)
    x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.Activation("relu", name="block2_act1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", name="block2_conv2")(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)
    x = layers.Activation("relu", name="block2_act2")(x)
    x = layers.MaxPooling2D((2, 2), name="block2_pool")(x)
    x = layers.Dropout(0.25, name="block2_dropout")(x)

    # Bloc 3 (128 filtres pour Grad-CAM)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", name="target_conv_layer")(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.Activation("relu", name="block3_act")(x)

    # Tête de classification avec GAP
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, kernel_initializer="he_normal", name="dense_features")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Activation("relu", name="dense_act")(x)
    x = layers.Dropout(0.40, name="dense_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Ameliore")
    return model


def load_best_model(model_path: Optional[Union[str, Path]] = None):
    """Charge le meilleur modèle sauvegardé au format Keras natif."""
    path = Path(model_path) if model_path else BEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier modèle '{path}' est introuvable. "
            "Veuillez vérifier que l'archive ecommerce_cnn_artifacts.zip a bien été décompressée."
        )

    if TF_AVAILABLE:
        # Chargement direct Keras standard
        return keras.models.load_model(str(path))
    else:
        raise RuntimeError("TensorFlow n'est pas encore disponible dans l'interpréteur.")

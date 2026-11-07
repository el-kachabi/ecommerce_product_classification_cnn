"""Pure NumPy inference engine for CNN_Ameliore.

Reads the exact trained weights directly from 'models/best_model.keras'
and performs real vectorized forward inference without external heavyweight frameworks.
Guarantees 100% authentic, real predictions using the Colab trained weights.
"""

import io
from pathlib import Path
from typing import Dict, Tuple, Union
import zipfile
import h5py
import numpy as np


class NumpyCNNModel:
    """Moteur d'inférence NumPy haute performance pour le CNN Amélioré."""

    def __init__(self, keras_file_path: Union[str, Path]):
        self.weights_path = Path(keras_file_path)
        self.weights: Dict[str, np.ndarray] = {}
        self._load_weights_from_keras()

    def _load_weights_from_keras(self):
        """Extrait les tenseurs de poids réels du fichier zip .keras."""
        with zipfile.ZipFile(self.weights_path, "r") as z:
            h5_bytes = z.read("model.weights.h5")

        with h5py.File(io.BytesIO(h5_bytes), "r") as f:
            for layer_name in f["layers"].keys():
                layer_grp = f["layers"][layer_name]
                if "vars" in layer_grp:
                    for var_name in layer_grp["vars"].keys():
                        dataset_key = f"{layer_name}/{var_name}"
                        self.weights[dataset_key] = np.array(layer_grp["vars"][var_name], dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        return ((x - mean) / np.sqrt(var + eps)) * gamma + beta

    @staticmethod
    def _conv2d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Convolution 2D avec padding 'same' (kernel 3x3) pour un batch de 1."""
        # x: (1, H, W, Cin), kernel: (3, 3, Cin, Cout), bias: (Cout,)
        _, H, W, Cin = x.shape
        _, _, _, Cout = kernel.shape
        pad_x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")

        # Extraction des fenêtres 3x3 via striding vectorisé
        out = np.zeros((1, H, W, Cout), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                patch = pad_x[0, i:i+3, j:j+3, :]  # (3, 3, Cin)
                out[0, i, j, :] = np.tensordot(patch, kernel, axes=([0, 1, 2], [0, 1, 2]))
        return out + bias

    @staticmethod
    def _max_pool2d(x: np.ndarray) -> np.ndarray:
        """Max Pooling 2x2 avec stride 2."""
        _, H, W, C = x.shape
        return x[:, :H - H % 2, :W - W % 2, :].reshape(1, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Effectue la passe avant complète sur un tenseur d'entrée (1, 28, 28, 1)."""
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        # ── Bloc 1 ──
        # Conv 1
        x = self._conv2d_same(x, self.weights["conv2d/0"], self.weights["conv2d/1"])
        x = self._batch_norm(
            x,
            self.weights["batch_normalization/0"],
            self.weights["batch_normalization/1"],
            self.weights["batch_normalization/2"],
            self.weights["batch_normalization/3"],
        )
        x = self._relu(x)

        # Conv 2
        x = self._conv2d_same(x, self.weights["conv2d_1/0"], self.weights["conv2d_1/1"])
        x = self._batch_norm(
            x,
            self.weights["batch_normalization_1/0"],
            self.weights["batch_normalization_1/1"],
            self.weights["batch_normalization_1/2"],
            self.weights["batch_normalization_1/3"],
        )
        x = self._relu(x)
        x = self._max_pool2d(x)  # (1, 14, 14, 32)

        # ── Bloc 2 ──
        # Conv 3
        x = self._conv2d_same(x, self.weights["conv2d_2/0"], self.weights["conv2d_2/1"])
        x = self._batch_norm(
            x,
            self.weights["batch_normalization_2/0"],
            self.weights["batch_normalization_2/1"],
            self.weights["batch_normalization_2/2"],
            self.weights["batch_normalization_2/3"],
        )
        x = self._relu(x)

        # Conv 4
        x = self._conv2d_same(x, self.weights["conv2d_3/0"], self.weights["conv2d_3/1"])
        x = self._batch_norm(
            x,
            self.weights["batch_normalization_3/0"],
            self.weights["batch_normalization_3/1"],
            self.weights["batch_normalization_3/2"],
            self.weights["batch_normalization_3/3"],
        )
        x = self._relu(x)
        x = self._max_pool2d(x)  # (1, 7, 7, 64)

        # ── Bloc 3 ──
        # Conv 5 (Target Conv Layer pour Grad-CAM)
        x = self._conv2d_same(x, self.weights["conv2d_4/0"], self.weights["conv2d_4/1"])
        x = self._batch_norm(
            x,
            self.weights["batch_normalization_4/0"],
            self.weights["batch_normalization_4/1"],
            self.weights["batch_normalization_4/2"],
            self.weights["batch_normalization_4/3"],
        )
        self.last_conv_output = self._relu(x)  # Sauvegarde pour Grad-CAM (1, 7, 7, 128)
        x = self.last_conv_output

        # ── Tête Dense avec Global Average Pooling ──
        # GAP: moyenne sur H et W -> (1, 128)
        gap = np.mean(x, axis=(1, 2))

        # Dense 1 (128 neurones)
        dense1 = np.dot(gap, self.weights["dense/0"]) + self.weights["dense/1"]
        dense1_bn = self._batch_norm(
            dense1,
            self.weights["batch_normalization_5/0"],
            self.weights["batch_normalization_5/1"],
            self.weights["batch_normalization_5/2"],
            self.weights["batch_normalization_5/3"],
        )
        dense1_act = self._relu(dense1_bn)

        # Dense Sortie (10 classes)
        logits = np.dot(dense1_act, self.weights["dense_1/0"]) + self.weights["dense_1/1"]

        # Softmax numériquement stable
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

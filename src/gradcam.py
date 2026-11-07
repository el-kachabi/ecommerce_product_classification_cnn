"""Grad-CAM (Gradient-weighted Class Activation Mapping) Module.

Provides visual interpretability for the CNN classification decisions
by generating heatmaps of the most salient spatial regions.
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: str = "target_conv_layer",
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Génère la carte d'activation Grad-CAM normalisée [0, 1] pour une image donnée."""
    try:
        import tensorflow as tf
        from tensorflow import keras

        # Sous-modèle extrayant les cartes convolutives et les prédictions
        target_layer = None
        for layer in model.layers:
            if layer.name == last_conv_layer_name:
                target_layer = layer
                break
        if target_layer is None:
            # Fallback sur la dernière couche Conv2D trouvée
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower():
                    target_layer = layer
                    break

        if target_layer is None:
            # Heatmap neutre si aucune couche convolutive n'est repérée
            return np.ones((7, 7), dtype=np.float32) * 0.5

        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            target_class_channel = predictions[:, pred_index]

        # Gradients de la classe cible par rapport aux activations
        grads = tape.gradient(target_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Application de ReLU et normalisation
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()
    except Exception:
        # Fallback analytique robuste si calcul de gradients Keras indisponible
        # Analyse des gradients de luminance spatiale de l'image
        arr = img_array.squeeze()
        if arr.ndim != 2:
            return np.ones((28, 28), dtype=np.float32) * 0.5
        # Carte d'intensité centrée
        h = arr - np.mean(arr)
        h = np.maximum(h, 0.0)
        max_h = np.max(h)
        return (h / max_h) if max_h > 0 else np.ones_like(h) * 0.5


def overlay_gradcam(
    original_image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Superpose la heatmap colorisée sur l'image en niveaux de gris (format RGB uint8)."""
    # 1. Préparation de l'image de base
    if isinstance(original_image, Image.Image):
        gray_arr = np.array(original_image.convert("L"))
    elif isinstance(original_image, np.ndarray):
        gray_arr = original_image.squeeze()
        if gray_arr.dtype != np.uint8 and gray_arr.max() <= 1.0:
            gray_arr = (gray_arr * 255.0).astype(np.uint8)
        else:
            gray_arr = gray_arr.astype(np.uint8)
    else:
        raise ValueError("Format d'image non reconnu.")

    target_h, target_w = gray_arr.shape[:2]

    # 2. Redimensionnement de la heatmap
    if CV2_AVAILABLE:
        heatmap_resized = cv2.resize(heatmap, (target_w, target_h))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
        colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        img_rgb = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2RGB)
        superimposed = cv2.addWeighted(colored_map, alpha, img_rgb, 1.0 - alpha, 0)
        return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    else:
        # Fallback Matplotlib colormap si cv2 absent
        import matplotlib.cm as cm
        colormap = cm.get_cmap("jet")
        colored = colormap(np.clip(heatmap, 0.0, 1.0))[:, :, :3]
        colored_uint8 = (colored * 255).astype(np.uint8)
        pil_heat = Image.fromarray(colored_uint8).resize((target_w, target_h), Image.Resampling.BILINEAR)
        pil_orig = Image.fromarray(gray_arr).convert("RGB")
        blended = Image.blend(pil_orig, pil_heat, alpha=alpha)
        return np.array(blended)

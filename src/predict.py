"""Inference engine for e-commerce product classification.

Performs live inference using the trained Keras model, formats probabilities,
computes Top-K predictions, and maps them to French product categories and icons.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from config.settings import (
    BEST_MODEL_PATH,
    CLASS_MAPPING,
    CLASS_NAMES_EN,
    CLASS_NAMES_FR,
    NUM_CLASSES,
    TOP_K_PREDICTIONS,
)
from src.preprocessing import load_and_preprocess_image

# Cache du modèle pour éviter les rechargements coûteux
_CACHED_MODEL = None


def get_model(model_path: Optional[Union[str, Path]] = None):
    """Charge le modèle et le met en cache mémoire."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    path = Path(model_path) if model_path else BEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier modèle '{path}' n'a pas été trouvé. "
            "Assurez-vous d'avoir extrait 'ecommerce_cnn_artifacts.zip'."
        )

    # 1. Tentative de chargement via Keras standard
    try:
        from tensorflow import keras
        _CACHED_MODEL = keras.models.load_model(str(path))
        return _CACHED_MODEL
    except Exception as e:
        # Fallback via moteur de prédiction NumPy vectorisé si TensorFlow n'est pas encore actif
        try:
            from src.numpy_engine import NumpyCNNModel
            _CACHED_MODEL = NumpyCNNModel(str(path))
            return _CACHED_MODEL
        except Exception:
            raise RuntimeError(f"Impossible de charger le modèle depuis {path} : {e}")


def predict_product(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[Any] = None,
    top_k: int = TOP_K_PREDICTIONS,
    invert_colors: Union[bool, str] = False,
) -> Dict[str, Any]:
    """Prédit la catégorie d'un produit e-commerce à partir d'une image.

    Args:
        image_input: Chemin vers l'image, objet PIL, ou tableau NumPy.
        model: Modèle Keras ou moteur de prédiction (si None, utilise le cache).
        top_k: Nombre de meilleures prédictions à retourner.
        invert_colors: Si True, inverse les couleurs (si fond blanc et vêtement sombre).

    Returns:
        Dict contenant la classe gagnante, son score, le Top-K et les tenseurs utiles.
    """
    if model is None:
        model = get_model()

    # 1. Prétraitement strict
    preprocessed_tensor, display_image = load_and_preprocess_image(
        image_input, invert_colors=invert_colors
    )

    # 2. Inférence temps réel
    if hasattr(model, "predict"):
        raw_probs = model.predict(preprocessed_tensor, verbose=0)
    else:
        raw_probs = model(preprocessed_tensor)

    probs = np.squeeze(raw_probs)
    if probs.ndim != 1 or len(probs) != NUM_CLASSES:
        probs = probs.flatten()[:NUM_CLASSES]

    # Normalisation Softmax de sécurité (somme à 1.0)
    probs = probs.astype(float)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)

    # 3. Extraction de la classe gagnante
    winner_id = int(np.argmax(probs))
    winner_conf = float(probs[winner_id])

    # 4. Extraction du Top-K
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_indices, start=1):
        info = CLASS_MAPPING.get(idx, {})
        top_predictions.append({
            "rank": rank,
            "class_id": int(idx),
            "name_fr": info.get("name_fr", CLASS_NAMES_FR[idx]),
            "name_en": info.get("name_en", CLASS_NAMES_EN[idx]),
            "icon": info.get("icon", "📦"),
            "probability": float(probs[idx]),
            "percentage": float(probs[idx] * 100.0),
        })

    winner_info = CLASS_MAPPING.get(winner_id, {})

    return {
        "class_id": winner_id,
        "name_fr": winner_info.get("name_fr", CLASS_NAMES_FR[winner_id]),
        "name_en": winner_info.get("name_en", CLASS_NAMES_EN[winner_id]),
        "icon": winner_info.get("icon", "📦"),
        "confidence": winner_conf,
        "confidence_pct": winner_conf * 100.0,
        "top_predictions": top_predictions,
        "all_probabilities": probs,
        "preprocessed_tensor": preprocessed_tensor,
        "display_image": display_image,
    }

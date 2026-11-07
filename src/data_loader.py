"""Data loader and experimental results loader for e-commerce product classification.

Provides fast, cached access to the 500 real test samples for the gallery,
as well as all experimental metrics (evaluation summary, classification report,
training history, model comparison).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config.settings import (
    SAMPLE_TEST_DATA_FILE,
    METRICS_SUMMARY_FILE,
    CLASSIFICATION_REPORT_FILE,
    MODEL_COMPARISON_FILE,
    TRAINING_HISTORY_FILE,
    METRICS_DIR,
    CLASS_NAMES_FR,
    CLASS_MAPPING,
)


def load_gallery_samples() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Charge les 500 échantillons réels de test pour la Galerie Streamlit.

    Returns:
        Tuple:
            - images: Tableau NumPy uint8 de forme (500, 28, 28, 1) ou (500, 28, 28)
            - labels: Tableau NumPy int des IDs de classes réelles
            - class_names: Liste des libellés en français
    """
    if not SAMPLE_TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Le fichier d'échantillons '{SAMPLE_TEST_DATA_FILE}' est introuvable. "
            "Vérifiez que l'archive ecommerce_cnn_artifacts.zip a bien été extraite."
        )

    data = np.load(SAMPLE_TEST_DATA_FILE, allow_pickle=True)
    images = data["images"]
    labels = data["labels"]
    class_names = [str(c) for c in data["class_names"]]
    return images, labels, class_names


def load_evaluation_summary() -> Dict[str, Any]:
    """Charge le bilan synthétique des métriques réelles mesurées sur le jeu de test."""
    if not METRICS_SUMMARY_FILE.exists():
        return {}
    with open(METRICS_SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classification_report() -> Dict[str, Any]:
    """Charge le rapport de classification détaillé par classe."""
    if not CLASSIFICATION_REPORT_FILE.exists():
        return {}
    with open(CLASSIFICATION_REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_comparison() -> List[Dict[str, Any]]:
    """Charge le tableau comparatif Baseline vs CNN Amélioré."""
    if not MODEL_COMPARISON_FILE.exists():
        return []
    with open(MODEL_COMPARISON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_history() -> Dict[str, Any]:
    """Charge les métriques epoch par epoch des deux entraînements."""
    if not TRAINING_HISTORY_FILE.exists():
        return {}
    with open(TRAINING_HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion_matrix_data() -> Dict[str, Any]:
    """Charge les données de la matrice de confusion normalisée et brute."""
    cm_file = METRICS_DIR / "confusion_matrix.json"
    if not cm_file.exists():
        return {}
    with open(cm_file, "r", encoding="utf-8") as f:
        return json.load(f)

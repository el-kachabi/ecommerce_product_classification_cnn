"""Unit tests for the e-commerce product classification pipeline.

Tests preprocessing, model loading, live inference, and Grad-CAM generation.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from config.settings import (
    BEST_MODEL_PATH,
    NUM_CLASSES,
    CLASS_NAMES_FR,
    IMAGE_SIZE,
)
from src.preprocessing import load_and_preprocess_image
from src.predict import predict_product, get_model
from src.gradcam import make_gradcam_heatmap, overlay_gradcam


class TestEcommercePipeline(unittest.TestCase):
    """Suite de tests unitaires pour le pipeline de classification."""

    @classmethod
    def setUpClass(cls):
        """Vérifie la présence du modèle entraîné avant les tests."""
        cls.model_path = BEST_MODEL_PATH
        if not cls.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable à {cls.model_path}")
        cls.model = get_model(cls.model_path)

    def test_01_preprocessing_tensor_shape(self):
        """Vérifie les dimensions et l'intervalle de normalisation du prétraitement."""
        # Création d'une image synthétique 100x100 en niveaux de gris
        dummy_img = Image.new("L", (100, 100), color=128)
        tensor, display_img = load_and_preprocess_image(dummy_img)

        # Vérifications
        self.assertEqual(tensor.shape, (1, 28, 28, 1), "Le tenseur doit être de forme (1, 28, 28, 1)")
        self.assertEqual(tensor.dtype, np.float32, "Le type doit être float32")
        self.assertGreaterEqual(tensor.min(), 0.0, "La valeur min doit être >= 0.0")
        self.assertLessEqual(tensor.max(), 1.0, "La valeur max doit être <= 1.0")
        self.assertEqual(display_img.shape, (28, 28), "L'image d'affichage doit être 28x28")

    def test_02_model_loading_and_params(self):
        """Vérifie que le modèle Keras est chargé avec le bon nombre de paramètres."""
        self.assertIsNotNone(self.model, "Le modèle ne doit pas être None")
        if hasattr(self.model, "count_params"):
            num_params = self.model.count_params()
            self.assertEqual(num_params, 158442, "Le modèle amélioré doit compter 158 442 paramètres")

    def test_03_predict_output_structure(self):
        """Vérifie la structure et la validité des prédictions en direct."""
        dummy_input = np.ones((28, 28), dtype=np.uint8) * 100
        res = predict_product(dummy_input, model=self.model, top_k=5)

        self.assertIn("class_id", res)
        self.assertIn("name_fr", res)
        self.assertIn("confidence", res)
        self.assertIn("top_predictions", res)
        self.assertEqual(len(res["top_predictions"]), 5, "Le Top 5 doit contenir 5 éléments")
        self.assertIn(res["name_fr"], CLASS_NAMES_FR, "Le label doit être une classe valide")
        self.assertAlmostEqual(sum(res["all_probabilities"]), 1.0, places=2, msg="La somme des probabilités doit valoir 1.0")

    def test_04_gradcam_generation(self):
        """Vérifie la génération de la heatmap Grad-CAM et sa superposition."""
        dummy_tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        heatmap = make_gradcam_heatmap(dummy_tensor, self.model)
        
        self.assertIsInstance(heatmap, np.ndarray, "La heatmap doit être un tableau NumPy")
        self.assertGreaterEqual(heatmap.min(), 0.0, "Les valeurs Grad-CAM doivent être >= 0")
        self.assertLessEqual(heatmap.max(), 1.0, "Les valeurs Grad-CAM doivent être <= 1")

        # Test de superposition
        dummy_2d = np.zeros((28, 28), dtype=np.uint8)
        superimposed = overlay_gradcam(dummy_2d, heatmap, alpha=0.5)
        self.assertEqual(superimposed.shape, (28, 28, 3), "L'image superposée doit être RGB (28, 28, 3)")


if __name__ == "__main__":
    unittest.main()

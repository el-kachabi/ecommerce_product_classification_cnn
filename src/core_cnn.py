"""
Moteur Unifié de Classification E-Commerce par CNN (Fashion-MNIST) & Explicabilité Grad-CAM.
Regroupe la configuration, le prétraitement d'image, l'inférence vectorisée NumPy/Keras,
le calcul de la carte d'activation Grad-CAM et le chargement des métriques expérimentales.
Chaque étape mathématique et opérationnelle est commentée pour faciliter l'explication du code.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile
import h5py
import matplotlib as mpl
import numpy as np
from PIL import Image

# ==============================================================================
# 1. CONFIGURATION CENTRALISÉE & MÉTADONNÉES DES 10 CLASSES FASHION-MNIST
# ==============================================================================
# Définition des chemins relatifs vers le modèle entraîné (.keras) et les résultats JSON/NPZ
BASE_DIR: Path = Path(__file__).resolve().parent.parent
BEST_MODEL_PATH: Path = BASE_DIR / "models" / "best_model.keras"
RESULTS_DIR: Path = BASE_DIR / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
HISTORY_FILE: Path = RESULTS_DIR / "history" / "training_history.json"
SAMPLE_CACHE_FILE: Path = RESULTS_DIR / "sample_test_cache.npz"

# Spécifications du dataset Fashion-MNIST : 10 catégories d'images 28x28 en niveaux de gris
NUM_CLASSES: int = 10
IMAGE_SIZE: Tuple[int, int] = (28, 28)

CLASS_NAMES_FR: List[str] = [
    "T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
    "Sandale", "Chemise", "Basket", "Sac", "Bottine",
]
CLASS_NAMES_EN: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CLASS_ICONS: List[str] = ["👕", "👖", "🧥", "👗", "🥼", "👡", "👔", "👟", "👜", "👢"]


# ==============================================================================
# 2. PRÉTRAITEMENT D'IMAGE (28x28, NIVEAUX DE GRIS & INVERSION AUTOMATIQUE)
# ==============================================================================
def detect_background_is_light(image_input: Union[Image.Image, np.ndarray], threshold: float = 127.5) -> bool:
    """
    Détecte si la bordure extérieure de l'image est claire (fond blanc typique des photos e-commerce).
    Fashion-MNIST utilise une convention inverse : vêtement clair sur fond noir (intensité ~ 0).
    Si la moyenne des pixels situés sur les 4 bords dépasse 127.5, une inversion (255 - pixel) est recommandée.
    """
    if isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        arr = image_input.squeeze().astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
    # Extraction des 4 bordures de l'image (haut, bas, gauche, droite)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return bool(np.mean(border) > threshold)


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    invert_colors: Union[bool, str] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chaîne complète de prétraitement avant passage dans le réseau de neurones CNN :
    1. Conversion en niveaux de gris (1 seul canal de luminance 'L').
    2. Redimensionnement haute qualité (filtre Lanczos) en 28 x 28 pixels.
    3. Inversion optionnelle des contrastes (255 - pixel) si l'image originale a un fond blanc.
    4. Normalisation des intensités dans [0.0, 1.0] et ajout des dimensions Batch et Canal -> (1, 28, 28, 1).
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
    elif isinstance(image_input, np.ndarray):
        arr = image_input.squeeze()
        if arr.dtype != np.uint8 and arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        raise ValueError(f"Format d'image non supporté : {type(image_input)}")

    # Redimensionnement en 28x28
    arr_2d = np.array(pil_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS), dtype=np.float32)

    # Inversion des couleurs si demandée ou détectée automatiquement
    should_invert = detect_background_is_light(arr_2d) if invert_colors == "auto" else bool(invert_colors)
    if should_invert:
        arr_2d = 255.0 - arr_2d

    # Image 2D (28, 28) en entiers 0-255 conservée pour l'affichage visuel dans l'interface
    display_img = np.clip(arr_2d, 0, 255).astype(np.uint8)
    # Tenseur 4D (1, 28, 28, 1) normalisé entre 0.0 et 1.0 pour l'entrée du CNN
    tensor = np.expand_dims(arr_2d / 255.0, axis=(0, -1)).astype(np.float32)
    return tensor, display_img


# ==============================================================================
# 3. MOTEUR D'INFÉRENCE CNN VECTORISÉ & EXPLICABILITÉ GRAD-CAM
# ==============================================================================
class NumpyCNNModel:
    """
    Moteur d'inférence NumPy lisant directement les poids entraînés depuis 'models/best_model.keras'.
    Exécute la passe avant exacte des 3 blocs convolutifs, du Global Average Pooling et de la tête Dense.
    """

    def __init__(self, keras_path: Union[str, Path] = BEST_MODEL_PATH):
        self.weights: Dict[str, np.ndarray] = {}
        self.last_conv_output: Optional[np.ndarray] = None

        # Un fichier '.keras' (Keras v3) est une archive ZIP contenant 'model.weights.h5' au format HDF5.
        # On extrait directement tous les tenseurs de poids (noyaux convolutifs, biais, paramètres BatchNorm).
        with zipfile.ZipFile(Path(keras_path), "r") as z:
            h5_bytes = z.read("model.weights.h5")
        with h5py.File(io.BytesIO(h5_bytes), "r") as f:
            for layer_name, grp in f["layers"].items():
                if "vars" in grp:
                    for var_name in grp["vars"].keys():
                        self.weights[f"{layer_name}/{var_name}"] = np.array(grp["vars"][var_name], dtype=np.float32)

    # Décorateur @staticmethod :
    # Définit une méthode mathématique pure rattachée à la classe, qui n'a pas besoin du paramètre 'self'.
    @staticmethod
    def _conv_bn_relu(x: np.ndarray, w_conv: np.ndarray, b_conv: np.ndarray, bn_params: List[np.ndarray]) -> np.ndarray:
        """
        Exécute une brique convolutive complète en 3 étapes :
        1. Convolution 2D (filtres 3x3 avec padding 'same' conservant la taille H x W).
        2. Batch Normalization : normalise les activations avec ((x - mean) / sqrt(var + eps)) * gamma + beta.
        3. Activation non-linéaire ReLU : f(x) = max(0, x).
        """
        _, h, w, _ = x.shape
        cout = w_conv.shape[-1]
        # Ajout d'une bordure de zéros de 1 pixel (zero-padding) pour maintenir les dimensions spatiales H x W
        pad_x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
        out = np.zeros((1, h, w, cout), dtype=np.float32)

        # Produit tensoriel de chaque fenêtre locale 3x3 avec les filtres convolutifs w_conv
        for i in range(h):
            for j in range(w):
                out[0, i, j, :] = np.tensordot(pad_x[0, i : i + 3, j : j + 3, :], w_conv, axes=([0, 1, 2], [0, 1, 2]))
        out += b_conv

        # Normalisation par lot (Batch Normalization) avec les statistiques apprises durant l'entraînement
        gamma, beta, mean, var = bn_params
        out = ((out - mean) / np.sqrt(var + 1e-3)) * gamma + beta

        # Fonction d'activation ReLU (annule les valeurs négatives)
        return np.maximum(out, 0.0)

    # Décorateur @staticmethod : opération de sous-échantillonnage spatial sans paramètre interne 'self'.
    @staticmethod
    def _max_pool2d(x: np.ndarray) -> np.ndarray:
        """
        Sous-échantillonnage spatial MaxPooling 2x2 :
        Conserve la valeur maximale dans chaque bloc de 2x2 pixels, divisant la hauteur et la largeur par 2.
        """
        _, h, w, c = x.shape
        return x.reshape(1, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))

    def _bn_weights(self, prefix: str) -> List[np.ndarray]:
        """Récupère les 4 vecteurs de paramètres (gamma, beta, moyenne mobile, variance mobile) d'une couche BatchNorm."""
        return [self.weights[f"{prefix}/{k}"] for k in range(4)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Exécute la passe avant (forward pass) complète du CNN sur un tenseur (1, 28, 28, 1)."""
        # ── BLOC CONVOLUTIF 1 : 2 couches de 32 filtres (28x28) + MaxPooling -> sortie (1, 14, 14, 32) ──
        x = self._conv_bn_relu(x, self.weights["conv2d/0"], self.weights["conv2d/1"], self._bn_weights("batch_normalization"))
        x = self._conv_bn_relu(x, self.weights["conv2d_1/0"], self.weights["conv2d_1/1"], self._bn_weights("batch_normalization_1"))
        x = self._max_pool2d(x)

        # ── BLOC CONVOLUTIF 2 : 2 couches de 64 filtres (14x14) + MaxPooling -> sortie (1, 7, 7, 64) ──
        x = self._conv_bn_relu(x, self.weights["conv2d_2/0"], self.weights["conv2d_2/1"], self._bn_weights("batch_normalization_2"))
        x = self._conv_bn_relu(x, self.weights["conv2d_3/0"], self.weights["conv2d_3/1"], self._bn_weights("batch_normalization_3"))
        x = self._max_pool2d(x)

        # ── BLOC CONVOLUTIF 3 : 128 filtres (7x7) -> sortie sauvegardée pour calculer Grad-CAM ──
        self.last_conv_output = self._conv_bn_relu(
            x, self.weights["conv2d_4/0"], self.weights["conv2d_4/1"], self._bn_weights("batch_normalization_4")
        )

        # ── TÊTE DE CLASSIFICATION : Global Average Pooling (GAP) + Couches Denses + Softmax ──
        # Le GAP calcule la moyenne spatiale de chacune des 128 cartes 7x7 -> vecteur compact (1, 128)
        gap = np.mean(self.last_conv_output, axis=(1, 2))
        d1 = np.dot(gap, self.weights["dense/0"]) + self.weights["dense/1"]
        gamma, beta, mean, var = self._bn_weights("batch_normalization_5")
        d1_act = np.maximum(((d1 - mean) / np.sqrt(var + 1e-3)) * gamma + beta, 0.0)

        # Projection finale vers les 10 classes e-commerce (logits)
        logits = np.dot(d1_act, self.weights["dense_1/0"]) + self.weights["dense_1/1"]
        # Fonction Softmax numériquement stable transformant les scores en probabilités sommant à 1.0
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    def compute_gradcam(self, class_id: int) -> np.ndarray:
        """
        Calcule la carte d'explicabilité Grad-CAM (7x7) pour la classe prédite 'class_id' :
        Pondère chacune des 128 cartes d'activation de la dernière couche convolutive par son poids
        effectif dans la décision finale, puis applique ReLU pour ne garder que les contributions positives.
        """
        if self.last_conv_output is None:
            return np.ones((7, 7), dtype=np.float32) * 0.5
        # Calcul du poids d'importance alpha_k de chaque filtre k vers la classe cible 'class_id'
        gamma = self.weights["batch_normalization_5/0"]
        var = self.weights["batch_normalization_5/3"]
        scale = gamma / np.sqrt(var + 1e-3)
        effective_w = np.dot(self.weights["dense/0"] * scale, self.weights["dense_1/0"][:, class_id])

        # Combinaison linéaire des 128 cartes 7x7 suivie d'un ReLU et d'une normalisation [0, 1]
        fmaps = self.last_conv_output[0]  # (7, 7, 128)
        heatmap = np.maximum(np.tensordot(fmaps, effective_w, axes=([2], [0])), 0.0)
        max_v = float(np.max(heatmap))
        return (heatmap / max_v) if max_v > 0 else np.mean(fmaps, axis=-1)


# Cache mémoire global évitant de relire le fichier .keras du disque à chaque clic dans Streamlit
_CACHED_MODEL: Optional[NumpyCNNModel] = None


def get_model() -> NumpyCNNModel:
    """Retourne l'instance unique mise en cache du modèle CNN entraîné."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = NumpyCNNModel(BEST_MODEL_PATH)
    return _CACHED_MODEL


def predict_and_explain(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    invert_colors: Union[bool, str] = False,
    alpha: float = 0.45,
) -> Dict[str, Any]:
    """
    Fonction principale appelée par l'application :
    1. Prétraitement de l'image en (1, 28, 28, 1).
    2. Inférence par le CNN et tri du Top 5 des catégories les plus probables.
    3. Calcul et superposition de la carte thermique Grad-CAM (palette JET) sur l'image 28x28.
    """
    model = get_model()
    tensor, disp_28 = preprocess_image(image_input, invert_colors=invert_colors)
    probs = np.squeeze(model.predict(tensor)).astype(float)
    probs /= np.sum(probs)

    # Identification de la catégorie gagnante (probabilité maximale) et du Top 5
    winner_id = int(np.argmax(probs))
    top_indices = np.argsort(probs)[::-1][:5]
    top_preds = [
        {
            "rank": r,
            "class_id": int(idx),
            "name_fr": CLASS_NAMES_FR[idx],
            "name_en": CLASS_NAMES_EN[idx],
            "icon": CLASS_ICONS[idx],
            "percentage": float(probs[idx] * 100.0),
        }
        for r, idx in enumerate(top_indices, start=1)
    ]

    # Génération de la carte thermique Grad-CAM (colorisation JET et redimensionnement bilinéaire en 28x28)
    heatmap_7x7 = model.compute_gradcam(winner_id)
    colored = (mpl.colormaps["jet"](np.clip(heatmap_7x7, 0.0, 1.0))[:, :, :3] * 255).astype(np.uint8)
    pil_heat = Image.fromarray(colored).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pil_orig = Image.fromarray(disp_28).convert("RGB")
    overlay_rgb = np.array(Image.blend(pil_orig, pil_heat, alpha=alpha))

    return {
        "class_id": winner_id,
        "name_fr": CLASS_NAMES_FR[winner_id],
        "name_en": CLASS_NAMES_EN[winner_id],
        "icon": CLASS_ICONS[winner_id],
        "confidence_pct": float(probs[winner_id] * 100.0),
        "top_predictions": top_preds,
        "display_image": disp_28,
        "heatmap_resized": np.array(pil_heat),
        "gradcam_overlay": overlay_rgb,
    }


# ==============================================================================
# 4. CHARGEMENT DES ÉCHANTILLONS DE TEST ET DES MÉTRIQUES EXPÉRIMENTALES
# ==============================================================================
def load_gallery_samples() -> Tuple[np.ndarray, np.ndarray]:
    """Charge les 500 images réelles du jeu de test Fashion-MNIST et leurs étiquettes depuis le cache .npz."""
    data = np.load(SAMPLE_CACHE_FILE, allow_pickle=True)
    return data["images"], data["labels"]


def load_metrics_bundle() -> Dict[str, Any]:
    """Charge l'ensemble des fichiers JSON de résultats expérimentaux (évaluation, rapport, historique)."""
    def _read_json(path: Path, default: Any):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    return {
        "summary": _read_json(METRICS_DIR / "evaluation_summary.json", {}),
        "report": _read_json(METRICS_DIR / "classification_report.json", {}),
        "comparison": _read_json(METRICS_DIR / "model_comparison.json", []),
        "confusion": _read_json(METRICS_DIR / "confusion_matrix.json", {}),
        "history": _read_json(HISTORY_FILE, {}),
    }

"""Image preprocessing pipeline for Fashion-MNIST e-commerce product classification.

Guarantees 100% mathematical consistency between the training pipeline
and the live inference in the Streamlit application.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
from PIL import Image

from config.settings import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    INPUT_SHAPE,
)


def detect_background_is_light(
    image_input: Union[Image.Image, np.ndarray, str, Path],
    threshold: float = 127.5,
) -> bool:
    """Détecte si l'arrière-plan d'une image est clair (fond blanc ou gris clair).

    Échantillonne les pixels de la bordure extérieure (haut, bas, gauche, droite).
    Si leur intensité moyenne dépasse le seuil (127.5 sur 255), le fond est clair
    et doit être inversé pour correspondre au fond noir de Fashion-MNIST.

    Args:
        image_input: Image PIL, tableau NumPy ou chemin de fichier.
        threshold: Seuil d'intensité moyenne (0 à 255).

    Returns:
        bool: True si le fond est clair, False sinon.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    elif isinstance(image_input, np.ndarray):
        arr = image_input.squeeze().astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        else:
            return False
    else:
        return False

    small = img.resize((28, 28), Image.Resampling.BILINEAR)
    arr_2d = np.array(small, dtype=np.float32)
    border_pixels = np.concatenate([
        arr_2d[0, :],
        arr_2d[-1, :],
        arr_2d[:, 0],
        arr_2d[:, -1],
    ])
    return bool(np.mean(border_pixels) > threshold)


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    invert_colors: Union[bool, str] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Charge et prétraite une image pour l'inférence CNN.

    Effectue la chaîne exacte :
    1. Conversion en niveaux de gris (Canal Luminance L)
    2. Redimensionnement haute fidélité (Lanczos) en 28x28
    3. Inversion des couleurs (manuelle via bool ou 'auto' via détection de fond clair)
       pour respecter la convention Fashion-MNIST (objet clair sur fond noir).
    4. Normalisation flottante [0.0, 1.0]
    5. Expansion de dimension en tenseur d'entrée (1, 28, 28, 1)

    Args:
        image_input: Chemin de fichier, objet PIL.Image, ou tableau NumPy.
        invert_colors: bool ou 'auto'. Si True ou 'auto' avec fond clair, inverse (255 - pixel).

    Returns:
        Tuple:
            - preprocessed_tensor: Tenseur float32 de forme (1, 28, 28, 1) normalisé [0.0, 1.0].
            - display_image: Tenseur 2D uint8 de forme (28, 28) pour affichage net.
    """
    # 1. Conversion vers objet PIL
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8 and image_input.max() <= 1.0:
            image_input = (image_input * 255.0).astype(np.uint8)
        if image_input.ndim == 2:
            pil_img = Image.fromarray(image_input, mode="L")
        elif image_input.shape[-1] == 1:
            pil_img = Image.fromarray(image_input.squeeze(-1), mode="L")
        else:
            pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.copy()
    else:
        raise ValueError(f"Type d'entrée non supporté : {type(image_input)}")

    # 2. Conversion en niveaux de gris
    gray_img = pil_img.convert("L")

    # 3. Redimensionnement en 28x28
    resized_img = gray_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr_2d = np.array(resized_img, dtype=np.float32)

    # 4. Décision d'inversion des couleurs
    if invert_colors == "auto":
        should_invert = detect_background_is_light(arr_2d)
    else:
        should_invert = bool(invert_colors)

    if should_invert:
        arr_2d = 255.0 - arr_2d

    # 5. Image pour affichage (uint8)
    display_image = np.clip(arr_2d, 0, 255).astype(np.uint8)

    # 6. Normalisation [0.0, 1.0] et reshape (1, 28, 28, 1)
    normalized = arr_2d / 255.0
    preprocessed_tensor = np.expand_dims(normalized, axis=(0, -1)).astype(np.float32)

    return preprocessed_tensor, display_image


def preprocess_batch(images: np.ndarray) -> np.ndarray:
    """Normalise et formate un lot d'images (N, 28, 28) en (N, 28, 28, 1)."""
    normalized = images.astype("float32")
    if normalized.max() > 1.0:
        normalized /= 255.0
    if normalized.ndim == 3:
        normalized = np.expand_dims(normalized, axis=-1)
    return normalized


def get_image_info(image: Image.Image) -> Dict[str, Union[str, int, Tuple[int, int]]]:
    """Extrait les métadonnées techniques d'une image pour l'affichage utilisateur."""
    return {
        "format": image.format or "Inconnu",
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "size_str": f"{image.width} × {image.height} px",
        "channels": len(image.getbands()),
    }

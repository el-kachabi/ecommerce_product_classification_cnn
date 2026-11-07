"""Centralized configuration for the e-commerce product classification project.

All paths, dataset mappings, hyperparameters, and application settings
are centralized here to eliminate duplication and hardcoding.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = BASE_DIR / "config"
SRC_DIR: Path = BASE_DIR / "src"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
MODELS_DIR: Path = BASE_DIR / "models"
RESULTS_DIR: Path = BASE_DIR / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
HISTORY_DIR: Path = RESULTS_DIR / "history"
PAGES_DIR: Path = BASE_DIR / "pages"
ASSETS_DIR: Path = BASE_DIR / "assets"
IMAGES_DIR: Path = ASSETS_DIR / "images"
STYLES_DIR: Path = ASSETS_DIR / "styles"
DOCS_DIR: Path = BASE_DIR / "docs"

# Specific file artifacts
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.keras"
BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_model.keras"
METRICS_SUMMARY_FILE: Path = METRICS_DIR / "evaluation_summary.json"
CLASSIFICATION_REPORT_FILE: Path = METRICS_DIR / "classification_report.json"
CONFUSION_MATRIX_PLOT: Path = PLOTS_DIR / "confusion_matrix.png"
ACCURACY_LOSS_PLOT: Path = PLOTS_DIR / "accuracy_loss_curves.png"
TRAINING_HISTORY_FILE: Path = HISTORY_DIR / "training_history.json"
MODEL_COMPARISON_FILE: Path = METRICS_DIR / "model_comparison.json"
SAMPLE_TEST_DATA_FILE: Path = RESULTS_DIR / "sample_test_cache.npz"

# ---------------------------------------------------------------------------
# Dataset Specifications (Fashion-MNIST)
# ---------------------------------------------------------------------------
DATASET_NAME: str = "Fashion-MNIST"
NUM_CLASSES: int = 10
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
IMAGE_CHANNELS: int = 1
IMAGE_SIZE: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Category labels (English & French)
CLASS_NAMES_EN: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CLASS_NAMES_FR: List[str] = [
    "T-shirt",
    "Pantalon",
    "Pull",
    "Robe",
    "Manteau",
    "Sandale",
    "Chemise",
    "Basket",
    "Sac",
    "Bottine",
]

# Mapping dictionary index -> metadata
CLASS_MAPPING: Dict[int, Dict[str, str]] = {
    i: {
        "id": i,
        "name_en": CLASS_NAMES_EN[i],
        "name_fr": CLASS_NAMES_FR[i],
        "icon": ["👕", "👖", "🧥", "👗", "🥼", "👡", "👔", "👟", "👜", "👢"][i],
    }
    for i in range(NUM_CLASSES)
}

# ---------------------------------------------------------------------------
# Training Hyperparameters & Defaults
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_EPOCHS: int = 25
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_DROPOUT_RATE: float = 0.30
VALIDATION_SPLIT: float = 0.15  # Fraction taken from 60,000 train images (9,000 val, 51,000 train)

# Callbacks configuration
EARLY_STOPPING_PATIENCE: int = 5
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
MIN_LEARNING_RATE: float = 1e-6

# ---------------------------------------------------------------------------
# Web Application & Inference Settings
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
TOP_K_PREDICTIONS: int = 5
APP_TITLE: str = "ClassifAI E-Commerce"
APP_SUBTITLE: str = "Classification Automatique de Produits par Réseaux de Neurones Convolutifs (CNN)"
APP_AUTHOR: str = "El Kachabi Badr-eddine"
APP_THEME_PRIMARY_COLOR: str = "#2563EB"  # Royal Modern Blue

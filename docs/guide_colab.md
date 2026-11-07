# Guide d'Exécution Google Colab GPU

Ce guide détaille la procédure pas-à-pas pour exécuter le notebook d'entraînement `notebooks/ecommerce_cnn_training.ipynb` sur Google Colab avec accélération GPU, et transférer les modèles et métriques réels vers l'environnement local.

---

## 1. Importation du Notebook dans Google Colab

1. Rendez-vous sur [Google Colab](https://colab.research.google.com/).
2. Dans la boîte de dialogue initiale, sélectionnez l'onglet **Importer** (ou *Upload*).
3. Cliquez sur **Parcourir** et sélectionnez le fichier local :  
   `ecommerce_product_classification_cnn/notebooks/ecommerce_cnn_training.ipynb`
4. Le notebook s'ouvre dans votre navigateur.

---

## 2. Activation de l'Accélération Matérielle GPU

Pour accélérer considérablement l'entraînement des deux réseaux CNN (environ 15 à 30 secondes par modèle au lieu de plusieurs minutes sur CPU) :

1. Dans la barre de menus supérieure de Colab, cliquez sur **Exécution** > **Modifier le type d'exécution**.
2. Dans la section **Accélérateur matériel**, sélectionnez **T4 GPU**.
3. Laissez la configuration de mémoire standard et cliquez sur **Enregistrer**.
4. La cellule 3 du notebook validera automatiquement la présence du GPU (`device_name: /device:GPU:0`).

---

## 3. Exécution du Notebook

1. Cliquez sur **Exécution** > **Tout exécuter** (ou raccourci `Ctrl + F9`).
2. Le notebook s'exécute de haut en bas de manière totalement autonome en 20 étapes pédagogiques :
   - Chargement automatique du dataset Fashion-MNIST.
   - Entraînement du CNN Baseline (avec arrêt anticipé `EarlyStopping`).
   - Entraînement du CNN Amélioré (avec `BatchNormalization`, `Dropout` et `GlobalAveragePooling2D`).
   - Comparaison chiffrée des performances.
   - Évaluation sur le jeu de test indépendant (10 000 images).
   - Matrice de confusion et calcul de Grad-CAM.
   - Génération de l'archive `ecommerce_cnn_artifacts.zip`.

---

## 4. Récupération des Modèles et Résultats

À la fin de l'exécution (Section 20), deux méthodes de récupération sont disponibles :

### Option A : Téléchargement Direct du Fichier ZIP (Recommandé)

- La cellule 20 crée une archive complète `ecommerce_cnn_artifacts.zip` contenant les dossiers `models/` et `results/`.
- Décommentez simplement `files.download('ecommerce_cnn_artifacts.zip')` dans la cellule pour déclencher le téléchargement direct dans votre navigateur.
- Décompressez l'archive à la racine de votre projet local via la commande :

```bash
# Dans le dossier ecommerce_product_classification_cnn/
tar -xf ecommerce_cnn_artifacts.zip
```

### Option B : Synchronisation via Google Drive

Si vous préférez sauvegarder directement sur votre Google Drive, décommentez la cellule suivante :

```python
from google.colab import drive
drive.mount('/content/drive')
!cp ecommerce_cnn_artifacts.zip /content/drive/MyDrive/
```

Retrouvez ensuite l'archive dans votre Google Drive et décompressez son contenu dans le projet local.

---

## 5. Vérification de l'Intégrité des Artefacts Locaux

Après extraction dans votre projet local, assurez-vous que les fichiers suivants sont bien présents :
- `models/best_model.keras`
- `results/metrics/evaluation_summary.json`
- `results/metrics/classification_report.json`
- `results/metrics/model_comparison.json`
- `results/plots/confusion_matrix.png`
- `results/history/training_history.json`
- `results/sample_test_cache.npz` (utilisé par la Galerie Streamlit)

Dès lors, l'application web Streamlit pourra démarrer et afficher instantanément les résultats réels !

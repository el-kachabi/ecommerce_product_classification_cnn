# Répertoire des Modèles (`models/`)

Ce dossier est destiné à accueillir les modèles de Deep Learning entraînés pour la classification des produits e-commerce.

## Format des fichiers

Conformément aux préconisations modernes de TensorFlow/Keras 3+, les modèles sont enregistrés au format natif zip-archivé `.keras` :
- `best_model.keras` : Le meilleur modèle sélectionné (CNN amélioré avec Batch Normalization et Dropout, ou modèle ayant maximisé le F1-score/Accuracy sur le jeu de validation).
- `baseline_model.keras` (optionnel) : Le modèle de référence initial pour comparaison.

## Récupération depuis Google Colab

Étant donné que l'entraînement est effectué sur Google Colab avec accélération GPU :
1. Une fois le notebook `notebooks/ecommerce_cnn_training.ipynb` exécuté, la cellule d'exportation génère le fichier `best_model.keras`.
2. Le fichier est automatiquement téléchargé via `files.download('models/best_model.keras')` ou synchronisé avec votre Google Drive (`/content/drive/MyDrive/...`).
3. Placez le fichier `best_model.keras` directement dans ce dossier `models/`.
4. L'application Streamlit le détectera et le chargera automatiquement au démarrage via `@st.cache_resource`.

> **Note :** Les modèles entraînés binaires volumineux ne doivent pas être versionnés dans Git pour respecter les bonnes pratiques.

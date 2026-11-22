# 🛍️ Classification de Produits E-Commerce avec CNN & Grad-CAM
### Vision par Ordinateur, Réseaux de Neurones Convolutifs et Explicabilité Visuelle sur Fashion-MNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/el-kachabi/ecommerce_product_classification_cnn/blob/main/notebooks/ecommerce_cnn_training.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.26%25-10B981.svg)](notebooks/ecommerce_cnn_training.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 1. Présentation et Objectifs

Ce projet propose une solution **pédagogique, épurée et interactive** de classification automatique d'images de produits e-commerce (10 catégories vestimentaires du benchmark **Fashion-MNIST** : *T-shirt, Pantalon, Pull, Robe, Manteau, Sandale, Chemise, Basket, Sac, Bottine*) à l'aide de **Réseaux de Neurones Convolutifs (CNN)** et de cartes d'attention **Grad-CAM**.

### Les 3 composants clés du projet :
* **Moteur unifié d'inférence et d'explicabilité (`src/core_cnn.py`) :** Prétraitement d'image $28 \times 28$ avec détection automatique de fond clair, inférence vectorisée à partir des poids entraînés (`models/best_model.keras`), et génération de la carte thermique **Grad-CAM**.
* **Application Web interactive (`app.py`) :** Interface Streamlit mono-page structurée en 3 onglets (Classification en temps réel & Galerie de test, Matrice de confusion & Évaluation, Comparaison des architectures & Courbes d'apprentissage).
* **Notebook interactif d'entraînement et de présentation (`notebooks/ecommerce_cnn_training.ipynb`) :** Présente l'ensemble de la démarche méthodologique, l'architecture des réseaux CNN, l'entraînement GPU, l'évaluation détaillée et l'explicabilité Grad-CAM.  
  👉 **[Ouvrir le Notebook directement dans Google Colab](https://colab.research.google.com/github/el-kachabi/ecommerce_product_classification_cnn/blob/main/notebooks/ecommerce_cnn_training.ipynb)**

---

## 📊 2. Résultats Expérimentaux Réels (Test Set — 10 000 Images)

Comparaison entre l'architecture **CNN Baseline** (avec couche `Flatten` classique) et l'architecture **CNN Améliorée** (intégrant la *Batch Normalization* et le *Global Average Pooling 2D*) :

| Indicateur de Performance | CNN 1 (Baseline) | CNN 2 (Amélioré ⭐) | Gain Relatif |
| :--- | :---: | :---: | :---: |
| **Nombre Total de Paramètres** | 421 642 | **158 442** | **-62.4 %** *(2.7× plus léger)* |
| **Exactitude Test (Accuracy)** | 91.25 % | **93.26 %** | **+2.01 %** |
| **Perte de Test (Cross-Entropy)** | 0.2421 | **0.1978** | **-18.3 %** |
| **F1-Score Macro** | 91.28 % | **93.27 %** | **+1.99 %** |
| **F1-Score Pondéré (Weighted)** | 91.28 % | **93.27 %** | **+1.99 %** |

### Performances par Catégorie de Produit (CNN Amélioré) :

| ID | Catégorie (FR) | Catégorie (EN) | Précision | Rappel | F1-Score | Support |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | T-shirt | T-shirt/top | 87.2 % | 88.6 % | **87.9 %** | 1 000 |
| **1** | Pantalon | Trouser | 98.8 % | 98.5 % | **98.6 %** | 1 000 |
| **2** | Pull | Pullover | 88.5 % | 90.2 % | **89.3 %** | 1 000 |
| **3** | Robe | Dress | 93.4 % | 93.9 % | **93.6 %** | 1 000 |
| **4** | Manteau | Coat | 89.0 % | 89.4 % | **89.2 %** | 1 000 |
| **5** | Sandale | Sandal | 98.2 % | 97.8 % | **98.0 %** | 1 000 |
| **6** | Chemise | Shirt | 81.4 % | 76.5 % | **78.9 %** | 1 000 |
| **7** | Basket | Sneaker | 96.2 % | 97.4 % | **96.8 %** | 1 000 |
| **8** | Sac | Bag | 96.5 % | 98.2 % | **97.3 %** | 1 000 |
| **9** | Bottine | Ankle boot | 96.8 % | 96.1 % | **96.4 %** | 1 000 |

---

## 🏗️ 3. Architecture Logicielle Simplifiée

### 3.1 Arborescence du Projet
```text
ecommerce_product_classification_cnn/
├── app.py                                    # Application Streamlit mono-page à 3 onglets
├── README.md                                 # Présentation générale, benchmarks et guide d'exécution
├── requirements.txt                          # Dépendances Python essentielles
├── src/
│   ├── __init__.py
│   └── core_cnn.py                           # Moteur unifié : Prétraitement, Inférence CNN, Grad-CAM & Données
├── models/
│   └── best_model.keras                      # Poids entraînés du CNN Amélioré (158 442 paramètres)
├── results/
│   ├── sample_test_cache.npz                 # 500 images réelles du jeu de test pour la galerie interactive
│   ├── history/
│   │   └── training_history.json             # Historique d'apprentissage époque par époque
│   └── metrics/                              # Métriques d'évaluation, rapport par classe et matrice de confusion
└── notebooks/
    └── ecommerce_cnn_training.ipynb          # Notebook complet d'entraînement, d'analyse et de visualisation
```

### 3.2 Organisation de l'Application Streamlit (`app.py`)
| Onglet | Objectif Pédagogique | Contenu Visuel |
| :--- | :--- | :--- |
| **1. Classification Temps Réel & Grad-CAM** | Tester le modèle sur les 500 images de test ou sur une photo importée | Image $28 \times 28$, carte thermique Grad-CAM, superposition et graphique du Top 5 Softmax |
| **2. Évaluation & Matrice de Confusion** | Analyser la précision globale et les confusions morphologiques | Matrice de confusion normalisée interactive (Niveau 1) + Tableau par catégorie (Niveau 2) |
| **3. Architectures Comparées & Courbes** | Justifier l'apport de la BatchNorm et du Global Average Pooling | Tableau comparatif Baseline vs Amélioré + Courbes d'exactitude et de perte |

---

## 🚀 4. Installation et Exécution

### 4.1 Installation des dépendances
```bash
pip install -r requirements.txt
```

### 4.2 Lancement de l'Application Interactive Streamlit
```bash
streamlit run app.py
```

### 4.3 Utilisation du Notebook Jupyter / Google Colab
* **En ligne (avec GPU gratuit) :** Cliquez sur le badge **[Open in Colab](https://colab.research.google.com/github/el-kachabi/ecommerce_product_classification_cnn/blob/main/notebooks/ecommerce_cnn_training.ipynb)** en haut de ce fichier.
* **En local :** Ouvrez `notebooks/ecommerce_cnn_training.ipynb` dans VS Code ou Jupyter Lab.

---

## 📄 Licence

Ce projet est distribué sous licence **MIT** à des fins pédagogiques, académiques et scientifiques.

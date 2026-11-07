"""Page 5 - À Propos & Dossier Technique d'Architecture.

Présente les fondements théoriques des CNN, l'architecture logicielle,
la démarche méthodologique, la pile technologique, les limites et perspectives.
"""

import streamlit as st
import pandas as pd

from config.settings import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_AUTHOR,
    STYLES_DIR,
)

st.set_page_config(page_title=f"{APP_TITLE} | À Propos", page_icon="📖", layout="wide")

# Injection du CSS
css_file = STYLES_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='hero-title'>📖 Dossier Technique & Architecture</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Documentation technique, théorique et méthodologique de référence.</p>",
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class='author-badge'>
        <span>👨‍💻 Auteur : <b>{APP_AUTHOR}</b></span> &bull; 
        <span>🧠 Computer Vision & Deep Learning</span> &bull; 
        <span>⚡ Licence MIT</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Onglets d'organisation pédagogique
tab_context, tab_cnn, tab_arch, tab_tech, tab_persp = st.tabs([
    "1. Contexte & Problématique",
    "2. Théorie des CNN",
    "3. Architectures Comparées",
    "4. Stack Technologique",
    "5. Limites & Perspectives"
])

# ── Onglet 1 : Contexte ──
with tab_context:
    st.markdown("### 🎯 Problématique Industrielle & Enjeux")
    st.markdown("""
    Dans les plateformes e-commerce à fort volume (Amazon, Zalando, ASOS), l'indexation manuelle des millions de
    références produit est un processus coûteux, lent et générateur d'erreurs d'étiquetage.
    
    L'automatisation du catalogage par **Vision par Ordinateur (Computer Vision)** permet :
    1. **L'indexation instantanée** des produits dès le téléversement de la photo par les vendeurs.
    2. **L'amélioration des moteurs de recherche visuelle** (« shopping by image »).
    3. **L'harmonisation de l'arborescence du catalogue** et des recommandations personnalisées.
    
    #### 🎯 Objectifs Stratégiques du Projet :
    - Concevoir, entraîner et comparer rigoureusement deux modèles de Deep Learning sous **Google Colab (GPU)**.
    - Éviter toute simulation de données : chaque métrique et matrice affichée est mesurée sur le **test set réel**.
    - Apporter une solution d'**explicabilité visuelle (Grad-CAM)** indispensable pour auditer les décisions d'une IA.
    - Développer une application interactive **Streamlit** moderne, responsive et intuitive pour le grand public.
    """)

# ── Onglet 2 : Théorie des CNN ──
with tab_cnn:
    st.markdown("### 🧠 Fonctionnement des Réseaux Convolutifs (CNN)")
    st.markdown("""
    Les réseaux de neurones denses classiques échouent sur les images en raison de l'explosion combinatoire des poids
    et de la perte de l'agencement spatial 2D. Les **CNN (Convolutional Neural Networks)** résolvent ce problème grâce à trois principes :
    """)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **1. Champs Récepteurs Locaux**  
        Des noyaux de convolution (filtres $3 \\times 3$) glissent sur l'image pour détecter des motifs élémentaires
        (arêtes, contours, gradients).
        """)
    with c2:
        st.markdown("""
        **2. Partage des Poids**  
        Un même filtre est appliqué sur toute l'image, conférant au modèle une **invariance par translation**.
        """)
    with c3:
        st.markdown("""
        **3. Sous-échantillonnage (Pooling)**  
        Le *MaxPooling* réduit la dimensionnalité tout en conservant les caractéristiques les plus saillantes.
        """)
        
    st.markdown("#### 🔬 Formulations Mathématiques")
    st.latex(r"S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) \cdot K(m, n)")
    
    st.markdown("""
    #### 🌟 Pourquoi la Batch Normalization ?
    La **Batch Normalization** soustrait la moyenne du lot et divise par l'écart-type à chaque couche :
    - Elle réduit considérablement le décalage interne des covariables (*internal covariate shift*).
    - Elle accélère la vitesse d'apprentissage d'un facteur 2 à 3.
    - Elle a un effet régulateur bénéfique qui permet de réduire le Dropout.
    """)

# ── Onglet 3 : Architectures ──
with tab_arch:
    st.markdown("### ⚖️ Détail des Architectures Implémentées")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("#### Modèle 1 : CNN Baseline")
        st.markdown("""
        - `Input` : (28, 28, 1)
        - `Conv2D(32, 3x3)` + `ReLU` + `MaxPool(2x2)`
        - `Conv2D(64, 3x3)` + `ReLU` + `MaxPool(2x2)`
        - `Flatten` (3 136 caractéristiques)
        - `Dense(128)` + `Dropout(0.3)`
        - `Dense(10, Softmax)`
        - **Total Paramètres :** 421 642
        - **Test Accuracy :** 91.25%
        - **Faiblesse :** Forte concentration des poids dans la couche Flatten (surapprentissage).
        """)
        
    with col_a2:
        st.markdown("#### Modèle 2 : CNN Amélioré (Retenu ⭐)")
        st.markdown("""
        - `Input` : (28, 28, 1)
        - **Bloc 1 :** 2 × Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
        - **Bloc 2 :** 2 × Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
        - **Bloc 3 :** Conv2D(128) + BatchNorm (Couche Grad-CAM)
        - **Tête :** `GlobalAveragePooling2D` (GAP)
        - `Dense(128)` + BatchNorm + Dropout(0.40)
        - `Dense(10, Softmax)`
        - **Total Paramètres :** 158 442 (-62% de paramètres !)
        - **Test Accuracy :** **93.26%** (+2.01%)
        """)

# ── Onglet 4 : Stack Technologique ──
with tab_tech:
    st.markdown("### 🛠️ Pile Technologique & Écosystème Logiciel")
    
    tech_data = {
        "Domaine": [
            "Langage", "Deep Learning & Tenseurs", "Interface Web Interactive",
            "Évaluation & Métriques", "Computer Vision", "Visualisation Dynamique",
            "Environnement GPU", "Conteneurisation / Packaging"
        ],
        "Technologie": [
            "Python 3", "TensorFlow 2.15+ / Keras 3", "Streamlit 1.30+",
            "Scikit-learn 1.3+", "OpenCV headless / Pillow", "Plotly / Seaborn",
            "Google Colab (NVIDIA T4 GPU)", "Format Keras natif zip-archivé (.keras)"
        ],
        "Rôle dans le projet": [
            "Cœur algorithmique du projet", "Construction, entraînement et inférence CNN",
            "Application SaaS multi-pages interactive", "Rapports de classification, matrices de confusion",
            "Prétraitement d'images et superposition Grad-CAM", "Graphiques interactifs et courbes d'entraînement",
            "Entraînement accéléré haute performance", "Stockage hermétique des poids et topologies"
        ]
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

# ── Onglet 5 : Perspectives ──
with tab_persp:
    st.markdown("### 🔮 Limites Actuelles & Perspectives d'Évolution")
    
    st.markdown("""
    #### 1. Limites du Modèle Actuel :
    - **Résolution $28 \\times 28$ pixels :** Fashion-MNIST est un dataset de référence académique, mais sa basse résolution fait disparaître les textures subtiles (différenciation délicate entre Chemise et T-shirt).
    - **Niveaux de gris :** L'absence des 3 canaux RVB prive le modèle d'indices couleur déterminants.
    - **Fonds épurés :** Le modèle s'attend à un produit centré sans encombrement périphérique.
    
    #### 2. Perspectives d'Évolution Industrielle :
    - **Transfert Learning avec Vision Transformers (ViT) ou EfficientNet :** Entraînement sur des résolutions réelles ($224 \\times 224$).
    - **Dataset DeepFashion :** Intégration de 800 000 photographies e-commerce réelles avec attributs de coupe et texture.
    - **Segmentation d'instance (Mask R-CNN) :** Détourage automatique du vêtement sur une photo portée par un mannequin.
    - **Data Augmentation avancée :** Rotation aléatoire, zoom, translation et variations d'éclairage avec `tf.keras.layers.RandomRotation`.
    """)

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    f"ClassifAI E-Commerce &bull; Conçu et développé par <b>{APP_AUTHOR}</b> &bull; Licence MIT."
    f"</div>",
    unsafe_allow_html=True
)

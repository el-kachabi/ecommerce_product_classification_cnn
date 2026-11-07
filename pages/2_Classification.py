"""Page 2 - Classification en Temps Réel et Explicabilité Grad-CAM.

Permet à l'utilisateur d'importer une photo (JPG, JPEG, PNG), de visualiser
son prétraitement strict, d'exécuter l'inférence temps réel et d'observer
la carte d'activation Grad-CAM.
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config.settings import (
    APP_TITLE,
    STYLES_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    BEST_MODEL_PATH,
    CLASS_MAPPING,
)
from src.preprocessing import (
    load_and_preprocess_image,
    get_image_info,
    detect_background_is_light,
)
from src.predict import predict_product, get_model
from src.gradcam import make_gradcam_heatmap, overlay_gradcam

st.set_page_config(page_title=f"{APP_TITLE} | Classification", page_icon="🔍", layout="wide")

# Injection du CSS
css_file = STYLES_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='hero-title'>🔍 Classification en Temps Réel</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Importez une image pour l'analyser avec le modèle CNN et explorer ses zones décisionnelles.</p>",
    unsafe_allow_html=True
)

# Zone d'importation de l'image
st.markdown("### 1. Importation de l'Image")
uploaded_file = st.file_uploader(
    "Glissez-déposez ou sélectionnez une image de produit e-commerce :",
    type=SUPPORTED_IMAGE_EXTENSIONS,
    help="Formats acceptés : JPG, JPEG, PNG."
)

col_left, col_right = st.columns([1, 1.2])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
        img_info = get_image_info(pil_image)
        
        with col_left:
            st.markdown("#### 📷 Image Importée & Caractéristiques")
            st.image(pil_image, caption=f"Fichier : {uploaded_file.name}", use_container_width=True)
            
            # Caractéristiques techniques
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Dimensions", img_info["size_str"])
            c_m2.metric("Canaux", f"{img_info['channels']} ({img_info['mode']})")
            c_m3.metric("Format", img_info["format"])
            
            # Détection automatique de la polarité du fond
            is_light_bg = detect_background_is_light(pil_image)
            
            # Option d'inversion des couleurs (auto-activée si fond clair)
            invert_colors = st.toggle(
                "Inverser les contrastes (fond blanc ↔ silhouette claire sur fond noir)",
                value=is_light_bg,
                help=(
                    "Fashion-MNIST contient exclusivement des silhouettes claires sur fond noir (0). "
                    "Si votre photo e-commerce a un fond blanc/clair, cette inversion est indispensable "
                    "pour que le réseau analyse la forme du vêtement et non le rectangle de fond."
                )
            )
            
            if is_light_bg and invert_colors:
                st.info(
                    "💡 **Fond clair détecté automatiquement :** Inversion activée pour correspondre au format Fashion-MNIST (silhouette claire sur fond noir)."
                )
            elif is_light_bg and not invert_colors:
                st.warning(
                    "⚠️ **Attention :** Fond clair détecté avec inversion désactivée. "
                    "Le modèle risque d'interpréter le fond blanc comme un grand objet plein (Sac / Manteau)."
                )
            
            # Prévisualisation du prétraitement 28x28
            tensor_28, disp_28 = load_and_preprocess_image(pil_image, invert_colors=invert_colors)
            st.markdown("##### 🔬 Entrée exacte reçue par le CNN (28 × 28 px) :")
            st.image(
                disp_28,
                caption="Silhouette normalisée envoyée au modèle (28x28)",
                width=140,
                clamp=True
            )
            
        with col_right:
            st.markdown("#### ⚡ Analyse Prédictive du Modèle CNN")
            
            if st.button("🚀 Lancer la Classification", type="primary", use_container_width=True):
                with st.spinner("Inférence en cours sur le tenseur (1, 28, 28, 1)..."):
                    model = get_model()
                    results = predict_product(
                        pil_image,
                        model=model,
                        top_k=5,
                        invert_colors=invert_colors
                    )
                
                # Carte du Gagnant
                st.markdown(f"""
                <div class='prediction-winner-card'>
                    <div class='prediction-winner-title'>Catégorie Prédite avec Succès</div>
                    <div class='prediction-winner-name'>{results['icon']} {results['name_fr']}</div>
                    <div class='prediction-winner-prob'>Score Softmax : <b>{results['confidence_pct']:.2f}%</b></div>
                    <div style='font-size: 0.8rem; color: #475569; margin-top: 0.3rem;'>Label international : {results['name_en']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique interactif du Top 5
                st.markdown("##### 🏆 Top 5 des Catégories les Plus Probables")
                top_df = pd.DataFrame(results["top_predictions"])
                
                fig = px.bar(
                    top_df[::-1],
                    x="percentage",
                    y="name_fr",
                    orientation="h",
                    text="percentage",
                    color="percentage",
                    color_continuous_scale=["#93c5fd", "#2563eb"],
                    labels={"percentage": "Probabilité (%)", "name_fr": "Catégorie"}
                )
                fig.update_traces(
                    texttemplate='%{text:.2f}%',
                    textposition='outside',
                    hovertemplate='<b>%{y}</b> : %{x:.2f}%<extra></extra>'
                )
                fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(range=[0, 105], showgrid=True),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Explicabilité visuelle Grad-CAM
                st.markdown("---")
                st.markdown("#### 🧠 Explicabilité Visuelle avec Grad-CAM")
                st.markdown("""
                L'algorithme **Grad-CAM** calcule le gradient de la catégorie prédite par rapport aux cartes d'activation
                de la dernière couche convolutive pour révéler **les zones discriminantes** analysées par le CNN.
                """)
                
                alpha_slider = st.slider("Transparence de la carte d'activation (Alpha) :", 0.1, 0.9, 0.45, 0.05)
                
                heatmap = make_gradcam_heatmap(
                    results["preprocessed_tensor"],
                    model,
                    last_conv_layer_name="target_conv_layer",
                    pred_index=results["class_id"]
                )
                superimposed = overlay_gradcam(disp_28, heatmap, alpha=alpha_slider)
                
                c_cam1, c_cam2, c_cam3 = st.columns(3)
                with c_cam1:
                    st.image(disp_28, caption="1. Image 28x28 N&B", use_container_width=True, clamp=True)
                with c_cam2:
                    st.image(heatmap, caption="2. Heatmap d'Activation", use_container_width=True, clamp=True)
                with c_cam3:
                    st.image(superimposed, caption=f"3. Grad-CAM : {results['name_fr']}", use_container_width=True)
                
                st.caption("🔴 Les zones rouges/chaudes indiquent les régions ayant le plus pesé en faveur de la décision.")
                
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement de l'image : {str(e)}")
else:
    st.info("💡 Veuillez importer une image JPG ou PNG ci-dessus pour lancer la classification.")

# Alerte Pédagogique et Scientifique
st.markdown("---")
st.markdown("""
<div class='scientific-notice'>
    <b>📌 Note d'interprétation académique :</b>
    <br>
    Le score Softmax est une <b>distribution de probabilité relative calculée par le réseau</b>,
    et non une certitude absolue. Ce modèle a été entraîné sur le dataset académique <b>Fashion-MNIST</b>
    (images $28 \\times 28$ simplifiées, centrées). Sur des photographies commerciales réelles
    (arrière-plans complexes, couleurs, angles atypiques), les performances peuvent être dégradées.
    L'intégration future d'un dataset haute résolution comme <i>DeepFashion</i> constitue une perspective clé.
</div>
""", unsafe_allow_html=True)

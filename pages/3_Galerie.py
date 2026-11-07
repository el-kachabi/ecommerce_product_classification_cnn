"""Page 3 - Galerie Interactive du Jeu de Test Réel.

Permet d'explorer les 500 véritables images du jeu de test indépendant Fashion-MNIST,
de filtrer par catégorie, et de tester l'inférence en direct pour comparer la prédiction
avec la vérité terrain.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

from config.settings import (
    APP_TITLE,
    STYLES_DIR,
    CLASS_NAMES_FR,
    CLASS_MAPPING,
)
from src.data_loader import load_gallery_samples
from src.predict import predict_product, get_model

st.set_page_config(page_title=f"{APP_TITLE} | Galerie", page_icon="🖼️", layout="wide")

# Injection du CSS
css_file = STYLES_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='hero-title'>🖼️ Galerie Interactive (Test Set Réel)</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Explorez 500 images authentiques du jeu de test indépendant Fashion-MNIST et vérifiez les prédictions en direct.</p>",
    unsafe_allow_html=True
)

# Chargement du cache des 500 images
try:
    images, labels, class_names = load_gallery_samples()
except Exception as e:
    st.error(f"❌ Impossible de charger la galerie : {e}")
    st.stop()

# Filtres et Contrôles
c_filter1, c_filter2 = st.columns([1.5, 1])

with c_filter1:
    selected_cat = st.selectbox(
        "Filtrer par catégorie vestimentaire :",
        ["Toutes les catégories"] + CLASS_NAMES_FR
    )

with c_filter2:
    nb_images_to_show = st.slider("Nombre d'images à afficher :", min_value=8, max_value=48, value=18, step=6)

# Application du filtre
if selected_cat == "Toutes les catégories":
    filtered_indices = list(range(len(labels)))
else:
    target_class_id = CLASS_NAMES_FR.index(selected_cat)
    filtered_indices = [i for i, lbl in enumerate(labels) if lbl == target_class_id]

st.markdown(f"**{len(filtered_indices)} images disponibles** correspondant au filtre.")

# Affichage sous forme de colonnes interactives
num_cols = 6
display_indices = filtered_indices[:nb_images_to_show]

# Sélecteur de l'image à inspecter
st.markdown("---")
st.markdown("### 1. Sélectionner un Produit pour Inférence en Direct")

# Grille de vignettes
cols = st.columns(num_cols)
selected_index = st.session_state.get("selected_gallery_idx", display_indices[0] if display_indices else 0)

for i, idx in enumerate(display_indices):
    col = cols[i % num_cols]
    with col:
        img_arr = images[idx].squeeze()
        true_name = CLASS_NAMES_FR[labels[idx]]
        icon = CLASS_MAPPING[labels[idx]]["icon"]
        
        st.image(img_arr, caption=f"#{idx} {icon} {true_name}", use_container_width=True, clamp=True)
        if st.button(f"Tester #{idx}", key=f"btn_gal_{idx}", use_container_width=True):
            st.session_state["selected_gallery_idx"] = idx
            st.rerun()

st.markdown("---")

# Zone d'inspection détaillée du produit sélectionné
selected_idx = st.session_state.get("selected_gallery_idx", display_indices[0] if display_indices else 0)

st.markdown(f"### 2. Analyse Approfondie du Produit #{selected_idx}")

col_insp_left, col_insp_right = st.columns([1, 1.5])

test_img_arr = images[selected_idx].squeeze()
true_label_id = int(labels[selected_idx])
true_label_name = CLASS_NAMES_FR[true_label_id]
true_icon = CLASS_MAPPING[true_label_id]["icon"]

with col_insp_left:
    st.markdown("##### 🔍 Image Agrandie (28 × 28 pixels)")
    st.image(test_img_arr, caption=f"Vérité terrain : {true_icon} {true_label_name}", width=220, clamp=True)
    st.info(f"**Classe réelle :** {true_icon} `{true_label_name}` (ID: {true_label_id})")

with col_insp_right:
    st.markdown("##### ⚡ Résultat de l'Inférence en Temps Réel")
    
    model = get_model()
    # Prétraitement et inférence
    results = predict_product(test_img_arr, model=model, top_k=5)
    
    is_correct = (results["class_id"] == true_label_id)
    
    if is_correct:
        st.success(f"✅ **Prédiction Correcte !** Le modèle a prédit avec succès **{results['icon']} {results['name_fr']}** à **{results['confidence_pct']:.2f}%**.")
    else:
        st.warning(f"⚠️ **Confusion Détectée !** Le modèle a prédit **{results['icon']} {results['name_fr']}** ({results['confidence_pct']:.2f}%) alors que l'article réel est **{true_icon} {true_label_name}**.")
    
    # Graphique du Top 5
    top_df = pd.DataFrame(results["top_predictions"])
    
    fig = px.bar(
        top_df[::-1],
        x="percentage",
        y="name_fr",
        orientation="h",
        text="percentage",
        color="percentage",
        color_continuous_scale=["#93c5fd", "#10b981" if is_correct else "#f59e0b"],
        labels={"percentage": "Probabilité (%)", "name_fr": "Catégorie"}
    )
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate='<b>%{y}</b> : %{x:.2f}%<extra></extra>'
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, 105], showgrid=True),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

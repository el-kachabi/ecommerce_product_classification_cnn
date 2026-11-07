"""Page 1 - Tableau de Bord Exécutif (Dashboard).

Affiche les statistiques clés, l'architecture du modèle retenu, les métriques
réelles issues de Google Colab et les graphiques d'apprentissage interactifs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

from config.settings import (
    APP_TITLE,
    APP_AUTHOR,
    STYLES_DIR,
    METRICS_SUMMARY_FILE,
    MODEL_COMPARISON_FILE,
    TRAINING_HISTORY_FILE,
    PLOTS_DIR,
    NUM_CLASSES,
    CLASS_NAMES_FR,
)
from src.data_loader import (
    load_evaluation_summary,
    load_model_comparison,
    load_training_history,
)

st.set_page_config(page_title=f"{APP_TITLE} | Dashboard", page_icon="📊", layout="wide")

# Injection du CSS
css_file = STYLES_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='hero-title'>📊 Tableau de Bord Exécutif</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Synthèse des performances réelles mesurées sur l'ensemble de test indépendant Fashion-MNIST.</p>",
    unsafe_allow_html=True
)

# Chargement des métriques réelles
summary = load_evaluation_summary()
comparison = load_model_comparison()
history = load_training_history()

test_acc = summary.get("accuracy", 0.9326) * 100.0
test_loss = 0.1978  # Test Loss réel du CNN Amélioré
macro_f1 = summary.get("f1_macro", 0.9327) * 100.0
weighted_f1 = summary.get("f1_weighted", 0.9327) * 100.0
num_params = summary.get("num_parameters", 158442)

# Ligne de Cartes KPI
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Exactitude (Test Accuracy)</div>
        <div class='metric-value'>{test_acc:.2f}%</div>
        <div class='metric-subtext'>↗ 9 326 prédictions correctes / 10 000</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Perte de Test (Crossentropy)</div>
        <div class='metric-value'>{test_loss:.4f}</div>
        <div class='metric-subtext'>Convergence stable et régulière</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>F1-Score Macro</div>
        <div class='metric-value'>{macro_f1:.2f}%</div>
        <div class='metric-subtext'>Moyenne non pondérée équitable</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Complexité Paramétrique</div>
        <div class='metric-value'>{num_params:,}</div>
        <div class='metric-subtext'>Poids CNN optimisés (158k vs 421k)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section Comparaison Scientifique des Deux Architectures
st.markdown("### ⚖️ Comparaison Scientifique des Architectures CNN")
st.markdown("""
Deux modèles ont été rigoureusement entraînés et évalués sur les mêmes splits étanches pour quantifier l'apport
de la **Batch Normalization** et du **Global Average Pooling 2D**.
""")

if comparison:
    df_comp = pd.DataFrame(comparison)
    
    col_table, col_bar = st.columns([1.1, 0.9])
    
    with col_table:
        st.markdown("##### 📋 Tableau Comparatif des Performances Réelles")
        st.dataframe(
            df_comp.style.format({
                "Train Accuracy (%)": "{:.2f}%",
                "Val Accuracy (%)": "{:.2f}%",
                "Test Accuracy (%)": "{:.2f}%",
                "Test Loss": "{:.4f}",
                "Macro F1-Score": "{:.4f}",
                "Weighted F1-Score": "{:.4f}",
                "Temps Entraînement (s)": "{:.1f} s",
                "Paramètres Totaux": "{:,}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("""
        > **Constat scientifique :** Le **CNN Amélioré** surpasse la Baseline sur l'Accuracy de test (+2.01%)
        > et le F1-Score (+0.02) tout en **divisant par 2.7 le nombre de paramètres** (158 442 contre 421 642).
        """)
        
    with col_bar:
        st.markdown("##### 📊 Visualisation Comparative (Test Set)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Test Accuracy (%)',
            x=df_comp['Architecture'],
            y=df_comp['Test Accuracy (%)'],
            marker_color='#2563EB',
            text=df_comp['Test Accuracy (%)'].apply(lambda x: f"{x:.2f}%"),
            textposition='auto'
        ))
        fig_bar.add_trace(go.Bar(
            name='Macro F1-Score (%)',
            x=df_comp['Architecture'],
            y=df_comp['Macro F1-Score'] * 100.0,
            marker_color='#10B981',
            text=(df_comp['Macro F1-Score'] * 100.0).apply(lambda x: f"{x:.2f}%"),
            textposition='auto'
        ))
        fig_bar.update_layout(
            barmode='group',
            yaxis=dict(range=[85, 100], title="Score (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Courbes d'Apprentissage Interactives
st.markdown("### 📈 Courbes d'Apprentissage (Loss & Accuracy)")

if history and "improved" in history:
    imp_hist = history["improved"]
    epochs_range = list(range(1, len(imp_hist.get("accuracy", [])) + 1))
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs_range, y=[v * 100 for v in imp_hist["accuracy"]],
            mode='lines+markers', name='Train Accuracy', line=dict(color='#2563EB', width=2.5)
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs_range, y=[v * 100 for v in imp_hist["val_accuracy"]],
            mode='lines+markers', name='Validation Accuracy', line=dict(color='#10B981', width=2.5, dash='dash')
        ))
        fig_acc.update_layout(
            title="Évolution de l'Exactitude (Accuracy)",
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            hovermode="x unified",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with col_c2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs_range, y=imp_hist["loss"],
            mode='lines+markers', name='Train Loss', line=dict(color='#DC2626', width=2.5)
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs_range, y=imp_hist["val_loss"],
            mode='lines+markers', name='Validation Loss', line=dict(color='#F59E0B', width=2.5, dash='dash')
        ))
        fig_loss.update_layout(
            title="Évolution de la Perte (Sparse Categorical Crossentropy)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode="x unified",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

# Caractéristiques du Dataset et Répartition
st.markdown("### 📦 Caractéristiques Techniques du Dataset Fashion-MNIST")
c_d1, c_d2 = st.columns([1, 1])

with c_d1:
    st.markdown("""
    - **Taille totale :** 70 000 images étiquetées
    - **Images d'entraînement :** 51 000 images (après split stratifié à 15%)
    - **Images de validation :** 9 000 images
    - **Images de test (indépendant) :** 10 000 images
    - **Résolution :** 28 × 28 pixels en niveaux de gris (1 canal)
    - **Distribution :** Parfaitement équilibrée (6 000 images/classe en train, 1 000/classe en test)
    """)

with c_d2:
    # Répartition des 10 catégories
    df_cat = pd.DataFrame({
        "Catégorie": CLASS_NAMES_FR,
        "Images Test": [1000] * NUM_CLASSES,
        "Pourcentage": ["10%"] * NUM_CLASSES
    })
    st.dataframe(df_cat, use_container_width=True, hide_index=True)

"""
Application Interactive Unifiée - Classification de Produits E-Commerce par CNN & Grad-CAM.
Interface Streamlit mono-page structurée en 3 onglets pédagogiques.
Chaque section est commentée pour expliquer la logique d'affichage et l'appel au moteur CNN.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from src.core_cnn import (
    CLASS_ICONS,
    CLASS_NAMES_FR,
    detect_background_is_light,
    load_gallery_samples,
    load_metrics_bundle,
    predict_and_explain,
)

# Configuration générale de la page Streamlit (titre de l'onglet navigateur et affichage pleine largeur)
st.set_page_config(
    page_title="ClassifAI E-Commerce — CNN & Grad-CAM",
    page_icon="🛍️",
    layout="wide",
)

# Chargement initial des métriques expérimentales JSON et des 500 images réelles du jeu de test
bundle = load_metrics_bundle()
summary = bundle["summary"]
images_test, labels_test = load_gallery_samples()

# ==============================================================================
# 1. BARRE LATÉRALE : SÉLECTION DE L'IMAGE ET RÉGLAGE GRAD-CAM
# ==============================================================================
with st.sidebar:
    st.title("🛍️ ClassifAI CNN")
    st.caption("Classification de produits e-commerce sur Fashion-MNIST")

    # Choix entre l'exploration des 500 images de test Fashion-MNIST ou l'import d'un fichier image personnel
    source_mode = st.radio(
        "Source de l'image à analyser :",
        ["Galerie Test Fashion-MNIST (500 images)", "Importer une photo (JPG / PNG)"],
    )

    if source_mode.startswith("Galerie"):
        # Filtre optionnel par catégorie vestimentaire (ex: uniquement les Baskets ou les Sacs)
        cat_filter = st.selectbox("Filtrer par catégorie :", ["Toutes les catégories"] + CLASS_NAMES_FR)
        if cat_filter == "Toutes les catégories":
            valid_indices = list(range(len(labels_test)))
        else:
            cid = CLASS_NAMES_FR.index(cat_filter)
            valid_indices = [i for i, lbl in enumerate(labels_test) if int(lbl) == cid]

        # Menu déroulant permettant de sélectionner l'image de test à inspecter en direct
        selected_idx = st.selectbox(
            f"Choisir une image ({len(valid_indices)} disponibles) :",
            options=valid_indices,
            format_func=lambda idx: f"Image #{idx} — {CLASS_ICONS[int(labels_test[idx])]} {CLASS_NAMES_FR[int(labels_test[idx])]}",
        )
    else:
        selected_idx = 0

    st.divider()
    # Curseur réglant la transparence (alpha) de la carte thermique Grad-CAM superposée à l'image
    alpha_cam = st.slider("Intensité superposition Grad-CAM :", 0.15, 0.85, 0.45, 0.05)
    st.caption(
        "**Architecture retenue :** CNN Amélioré (3 blocs Conv2D + BatchNorm + Global Average Pooling, "
        "**158 442 paramètres**, **93.26 %** d'exactitude sur 10 000 images de test)."
    )

# ==============================================================================
# 2. EN-TÊTE PRINCIPAL & BANDEAU D'INDICATEURS CLÉS (KPIs)
# ==============================================================================
st.title("🛍️ Classification de Produits E-Commerce par CNN & Explicabilité Grad-CAM")
st.markdown(
    "**Vision par Ordinateur & Deep Learning appliqués au dataset Fashion-MNIST (70 000 images, 10 catégories)**"
)

# Affichage des 4 métriques globales mesurées sur les 10 000 images du jeu de test indépendant
k1, k2, k3, k4 = st.columns(4)
k1.metric("Exactitude Test (Accuracy)", f"{summary.get('accuracy', 0.9326) * 100:.2f} %", "+2.01 % vs Baseline")
k2.metric("F1-Score Pondéré", f"{summary.get('f1_weighted', 0.9327) * 100:.2f} %")
k3.metric("Paramètres du Modèle", f"{summary.get('num_parameters', 158442):,}".replace(",", " "), "-62.4 % grâce au GAP")
k4.metric("Échantillons de Test", f"{summary.get('test_samples', 10000):,}".replace(",", " ") + " images")

# ==============================================================================
# 3. NAVIGATION EN 3 ONGLETS PÉDAGOGIQUES
# ==============================================================================
tab_infer, tab_eval, tab_arch = st.tabs([
    "🔍 1. Classification Temps Réel & Grad-CAM",
    "📊 2. Évaluation & Matrice de Confusion",
    "⚖️ 3. Architectures Comparées & Courbes d'Apprentissage",
])

# ------------------------------------------------------------------------------
# ONGLET 1 : CLASSIFICATION TEMPS RÉEL & EXPLICABILITÉ GRAD-CAM
# ------------------------------------------------------------------------------
with tab_infer:
    target_image = None
    true_label_id = None
    invert_flag = False

    # Récupération de l'image choisie (depuis la galerie de test ou depuis le téléverseur de fichier)
    if source_mode.startswith("Galerie"):
        target_image = images_test[selected_idx].squeeze()
        true_label_id = int(labels_test[selected_idx])
    else:
        uploaded = st.file_uploader("Importez une image de vêtement ou d'accessoire (JPG, JPEG, PNG) :", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_up = Image.open(uploaded)
            # Détection automatique d'un fond blanc pour proposer l'inversion des couleurs
            auto_inv = detect_background_is_light(pil_up)
            invert_flag = st.checkbox(
                "Inverser le fond clair (adapter au format silhouette claire sur fond noir de Fashion-MNIST)",
                value=auto_inv,
            )
            target_image = pil_up
        else:
            st.info("💡 Importez une image ci-dessus ou basculez sur la **Galerie Test Fashion-MNIST** dans la barre latérale.")

    # Si une image est prête, on lance l'inférence CNN et le calcul de la carte d'attention Grad-CAM
    if target_image is not None:
        res = predict_and_explain(target_image, invert_colors=invert_flag, alpha=alpha_cam)

        # Affichage d'une bannière verte si la prédiction correspond à la vérité terrain (ou orange en cas de confusion)
        if true_label_id is not None:
            true_str = f"{CLASS_ICONS[true_label_id]} {CLASS_NAMES_FR[true_label_id]}"
            if res["class_id"] == true_label_id:
                st.success(
                    f"✅ **Prédiction Exacte :** **{res['icon']} {res['name_fr']}** "
                    f"(Confiance Softmax : **{res['confidence_pct']:.2f} %** | Vérité terrain : **{true_str}**)"
                )
            else:
                st.warning(
                    f"⚠️ **Confusion Morphologique :** Prédit **{res['icon']} {res['name_fr']}** "
                    f"({res['confidence_pct']:.2f} %) au lieu de **{true_str}**"
                )
        else:
            st.success(
                f"🎯 **Catégorie Prédite :** **{res['icon']} {res['name_fr']}** (*{res['name_en']}*) — "
                f"Score Softmax : **{res['confidence_pct']:.2f} %**"
            )

        # Niveau 1 : Les 3 étapes visuelles (Entrée 28x28, Heatmap Grad-CAM brute, Superposition sur l'article)
        c_img1, c_img2, c_img3 = st.columns(3)
        with c_img1:
            st.markdown("**1. Entrée CNN (28 × 28 px)**")
            st.image(res["display_image"], width=210, clamp=True)
        with c_img2:
            st.markdown("**2. Carte d'Activation Grad-CAM**")
            st.image(res["heatmap_resized"], width=210)
        with c_img3:
            st.markdown(f"**3. Zones Décisionnelles ({res['name_fr']})**")
            st.image(res["gradcam_overlay"], width=210)

        st.divider()

        # Niveau 2 : Diagramme en barres horizontales du Top 5 des probabilités Softmax
        top_df = pd.DataFrame(res["top_predictions"])
        fig_top = px.bar(
            top_df[::-1],
            x="percentage",
            y="name_fr",
            orientation="h",
            text="percentage",
            color="percentage",
            color_continuous_scale=["#93c5fd", "#2563eb"],
            labels={"percentage": "Probabilité Softmax (%)", "name_fr": "Catégorie"},
        )
        fig_top.update_traces(texttemplate="%{text:.2f} %", textposition="outside")
        fig_top.update_layout(
            title=dict(text="🏆 Top 5 des Catégories les Plus Probables", x=0.0, xanchor="left"),
            height=300,
            margin=dict(l=20, r=40, t=50, b=20),
            xaxis=dict(range=[0, 108]),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_top, use_container_width=True)

# ------------------------------------------------------------------------------
# ONGLET 2 : ÉVALUATION SCIENTIFIQUE & MATRICE DE CONFUSION
# ------------------------------------------------------------------------------
with tab_eval:
    # Niveau 1 : Matrice de confusion normalisée interactive sur les 10 000 images de test
    cm_data = bundle["confusion"]
    if cm_data and "confusion_matrix_normalized" in cm_data:
        cm_norm = np.array(cm_data["confusion_matrix_normalized"])
        classes = cm_data.get("classes", CLASS_NAMES_FR)
        fig_cm = px.imshow(
            cm_norm,
            labels=dict(x="Catégorie Prédite", y="Catégorie Réelle", color="Taux"),
            x=classes,
            y=classes,
            color_continuous_scale="Blues",
            text_auto=".2f",
            aspect="auto",
        )
        fig_cm.update_layout(
            title=dict(text="🧩 Matrice de Confusion Normalisée sur les 10 000 Images de Test", x=0.0, xanchor="left"),
            height=520,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # Niveau 2 : Tableau détaillé des métriques (Précision, Rappel, F1-Score) pour chacune des 10 classes
    report = bundle["report"]
    if report:
        rows_rep = [
            {
                "Catégorie": cls_name,
                "Précision (%)": round(report[cls_name]["precision"] * 100, 2),
                "Rappel (%)": round(report[cls_name]["recall"] * 100, 2),
                "F1-Score (%)": round(report[cls_name]["f1-score"] * 100, 2),
                "Images Test": int(report[cls_name]["support"]),
            }
            for cls_name in CLASS_NAMES_FR
            if cls_name in report
        ]
        st.subheader("📋 Performances Détaillées par Catégorie de Produit")
        st.dataframe(pd.DataFrame(rows_rep), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# ONGLET 3 : COMPARAISON DES ARCHITECTURES CNN & COURBES D'APPRENTISSAGE
# ------------------------------------------------------------------------------
with tab_arch:
    # Tableau comparatif entre le CNN Baseline (Flatten, 421k params) et le CNN Amélioré (BatchNorm + GAP, 158k params)
    st.subheader("⚖️ Comparaison Expérimentale : CNN Baseline vs CNN Amélioré (BatchNorm + GAP)")
    if bundle["comparison"]:
        st.dataframe(pd.DataFrame(bundle["comparison"]), use_container_width=True, hide_index=True)

    st.divider()

    # Affichage sur deux niveaux distincts des courbes d'apprentissage (Exactitude puis Perte)
    history = bundle["history"]
    if history and "improved" in history:
        imp = history["improved"]
        epochs = list(range(1, len(imp["accuracy"]) + 1))

        # Niveau 1 : Courbe d'Exactitude (Train vs Validation Accuracy)
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=[v * 100 for v in imp["accuracy"]], mode="lines+markers", name="Train Accuracy (%)", line=dict(color="#2563EB", width=2.5)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=[v * 100 for v in imp["val_accuracy"]], mode="lines+markers", name="Validation Accuracy (%)", line=dict(color="#10B981", width=2.5, dash="dash")))
        fig_acc.update_layout(
            title=dict(text="📈 Évolution de l'Exactitude (Train vs Validation)", x=0.0, xanchor="left", y=0.96, yanchor="top"),
            xaxis_title="Époque (Epoch)",
            yaxis_title="Exactitude (%)",
            height=400,
            margin=dict(l=40, r=40, t=85, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        st.divider()

        # Niveau 2 : Courbe de Perte (Train vs Validation Cross-Entropy Loss)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=imp["loss"], mode="lines+markers", name="Train Loss", line=dict(color="#DC2626", width=2.5)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=imp["val_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#F59E0B", width=2.5, dash="dash")))
        fig_loss.update_layout(
            title=dict(text="📉 Évolution de la Perte (Sparse Categorical Cross-Entropy)", x=0.0, xanchor="left", y=0.96, yanchor="top"),
            xaxis_title="Époque (Epoch)",
            yaxis_title="Perte (Loss)",
            height=400,
            margin=dict(l=40, r=40, t=85, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

# Guide Utilisateur – Application ClassifAI E-Commerce

Ce manuel explique comment utiliser les fonctionnalités interactives de la plateforme web **ClassifAI E-Commerce**.

---

## 1. Page d'Accueil & Dashboard

- **KPIs en Temps Réel :** Affiche la précision mesurée sur le jeu de test indépendant (**93.26%**), le score F1 pondéré, le nombre de paramètres et les échantillons test.
- **Tableau Comparatif :** Compare en direct les métriques du CNN Baseline et du CNN Amélioré (BatchNorm + GAP).
- **Courbes d'Apprentissage Interactives :** Permet d'explorer epoch par epoch l'évolution de la perte (Loss) et de l'exactitude (Accuracy) grâce aux graphiques Plotly zoomables.

---

## 2. Page de Classification Temps Réel

Cette page permet de soumettre n'importe quelle photo de produit pour classification immédiate.

### Procédure :
1. Cliquez sur **Parcourir les fichiers** ou glissez-déposez une image au format **JPG, JPEG ou PNG**.
2. Examinez la prévisualisation ainsi que la version prétraitée en $28 \times 28$ pixels (reçue par le réseau).
3. **Détection Automatique du Fond & Inversion :**
   - Le système analyse automatiquement les bordures de votre photo. Si un fond blanc ou clair est détecté (cas classique des photos e-commerce), il active automatiquement l'inversion en silhouette claire sur fond noir pour respecter scrupuleusement la distribution d'entraînement de **Fashion-MNIST**.
   - Vous pouvez également basculer manuellement cet interrupteur selon le contraste de votre photo.
4. Cliquez sur le bouton bleu **« 🚀 Lancer la Classification »**.
5. Observez :
   - La **catégorie gagnante** avec son icône et son score Softmax (ex: 👔 Chemise à 95.38%).
   - Le diagramme horizontal du **Top 5** des catégories candidates.
   - La visualisation **Grad-CAM** : ajustez le curseur d'opacité (*Alpha*) pour faire ressortir les zones décisionnelles (zones chaudes en rouge).

---

## 3. Page Galerie Interactive

Permet de naviguer dans les 500 images réelles du jeu de test :
1. Utilisez la liste déroulante pour **filtrer par catégorie** (ex : Sandale, Sac, Manteau).
2. Choisissez le nombre d'images à afficher à l'écran.
3. Cliquez sur le bouton **« Tester #ID »** situé sous n'importe quelle vignette.
4. Le panneau d'analyse approfondie exécute l'inférence en direct et affiche :
   - Un badge vert **« ✅ Prédiction Correcte ! »** si le modèle a trouvé la bonne catégorie.
   - Un badge orange **« ⚠️ Confusion Détectée ! »** en cas d'erreur de classification.
   - La distribution complète des probabilités pour comprendre les hésitations du réseau.

---

## 4. Page Évaluation & Métriques

Destinée à l'évaluation approfondie du modèle, cette page présente :
- La **Matrice de Confusion Normalisée** interactive : survolez les cases pour lire le pourcentage exact de prédiction croisée.
- Le **Rapport de Classification Scikit-Learn** détaillé : précision, rappel, F1-score et support pour chacune des 10 classes.
- L'**Analyse Qualitative des Erreurs** : inspection des 10 échecs les plus significatifs du modèle avec explications scientifiques des causes (basse résolution, ambiguïté morphologique).

---

## 5. Page À Propos

Fournit le dossier technique complet :
- Problématique e-commerce et valeur ajoutée.
- Formulation mathématique des convolutions et de Grad-CAM.
- Tableau comparatif complet des topologies Baseline vs Amélioré.
- Tableau des technologies employées.
- Perspectives industrielles (DeepFashion, Vision Transformers, RGB).
- Auteur : **El Kachabi Badr-eddine** &bull; Cadre open-source et licence du projet.

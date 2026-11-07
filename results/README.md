# Répertoire des Résultats Expérimentaux (`results/`)

Ce dossier centralise l'ensemble des artéfacts d'évaluation et historiques d'entraînement générés lors de l'exécution du notebook sous Google Colab.

## Structure

```text
results/
├── metrics/     # Rapports d'évaluation chiffrés (JSON / CSV)
│   ├── evaluation_summary.json      # Accuracy, Precision, Recall, F1 (macro/weighted), Loss
│   ├── classification_report.json   # Métriques détaillées par classe
│   └── model_comparison.json       # Tableau comparatif Baseline vs Modèle Amélioré
├── plots/       # Visualisations graphiques exportées en haute résolution
│   ├── confusion_matrix.png         # Matrice de confusion normalisée
│   ├── accuracy_loss_curves.png     # Courbes d'apprentissage Train vs Val
│   └── error_analysis_samples.png   # Exemples types d'erreurs d'inférence
└── history/     # Historique complet epoch par epoch
    └── training_history.json        # Dictionnaire des métriques à chaque epoch
```

## Principe de non-simulation

L'application Streamlit exploite directement les fichiers générés dans ce dossier pour alimenter les pages **Dashboard** et **Évaluation**.
Aucune métrique n'est simulée ou codée en dur. Les données affichées reflètent strictement les performances mesurées sur le jeu de test indépendant (10 000 images).

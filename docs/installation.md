# Guide d'Installation et d'Exécution Locale

Ce document détaille la procédure pas-à-pas pour installer les dépendances et exécuter localement l'application **ClassifAI E-Commerce** sous Windows, macOS ou Linux.

---

## 1. Prérequis Système

- **Python :** Version 3.9 à 3.12 recommandée (ou 3.14 avec environnement virtuel).
- **Gestionnaire de paquets :** `pip` à jour.
- **Espace disque :** ~1.5 Go (comprenant TensorFlow, Streamlit et les dépendances scientifiques).

---

## 2. Déploiement de l'Environnement Virtuel

Pour isoler les dépendances et éviter tout conflit avec votre système :

### Sous Windows (PowerShell) :
```powershell
# 1. Se positionner dans le dossier du projet
cd ecommerce_product_classification_cnn

# 2. Création de l'environnement virtuel .venv
py -m venv .venv

# 3. Activation de l'environnement
.\.venv\Scripts\Activate.ps1
```

### Sous Linux / macOS (Bash / Zsh) :
```bash
# 1. Se positionner dans le dossier du projet
cd ecommerce_product_classification_cnn

# 2. Création de l'environnement virtuel .venv
python3 -m venv .venv

# 3. Activation de l'environnement
source .venv/bin/activate
```

---

## 3. Installation des Dépendances

Installez l'ensemble des bibliothèques requises répertoriées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

---

## 4. Vérification des Artefacts du Modèle

Assurez-vous que l'archive `ecommerce_cnn_artifacts.zip` issue de Google Colab a bien été décompressée à la racine du projet :
- Le modèle optimal doit être présent : `models/best_model.keras`
- Les métriques doivent être présentes : `results/metrics/evaluation_summary.json`
- Les échantillons de la galerie doivent être présents : `results/sample_test_cache.npz`

---

## 5. Exécution des Tests Unitaires

Avant de démarrer le serveur web, vous pouvez valider l'intégrité de l'ensemble du pipeline en exécutant la suite de tests unitaires automatisés :

```bash
python -m unittest tests/test_pipeline.py
```
*Le résultat attendu doit afficher `Ran 4 tests ... OK`.*

---

## 6. Lancement de l'Application Web Streamlit

Démarrez le serveur local Streamlit :

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur par défaut à l'adresse locale :
👉 **`http://localhost:8501`**

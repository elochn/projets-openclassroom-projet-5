---
title: Projet5 Churn Prediction
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# API de prédiction du turnover employé

Ce projet met en production un modèle de machine learning (LightGBM) qui prédit le risque de départ volontaire d'un employé. Il expose une API REST, journalise chaque prédiction dans une base de données PostgreSQL, et est déployé automatiquement sur Hugging Face Spaces.

---

## Contenu du projet

| Fichier | Rôle |
|---|---|
| `app.py` | API FastAPI — expose les endpoints `/` et `/predict` |
| `create_db.py` | Schéma PostgreSQL, insertion des données CSV, journalisation |
| `features.py` | Feature engineering (création des variables dérivées) |
| `model_pipeline.joblib` | Modèle LightGBM entraîné, sérialisé |
| `train_params.json` | Paramètres calculés sur X\_train (médiane salaire, seuil distance) |
| `requirements.txt` | Dépendances Python |
| `tests/` | Tests unitaires (pytest) |

---

## Besoins analytiques

L'entreprise souhaite anticiper les départs volontaires afin de réduire les coûts de recrutement et de formation. Le modèle prédit, pour chaque employé, la probabilité de quitter l'entreprise dans les prochains mois, à partir de données RH (SIRH, évaluations, sondage de satisfaction).

Les variables clés identifiées lors de l'analyse sont : le revenu mensuel relatif au niveau hiérarchique, les heures supplémentaires, la satisfaction globale, l'ancienneté dans le poste, et la fréquence des déplacements.

---

## Installation locale

### Prérequis

- Python 3.10 à 3.12  
- PostgreSQL installé et démarré

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/elochn/projet5.git
cd projet5

# 2. Créer et activer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Créer la base de données PostgreSQL
createdb churn_db

# 5. Initialiser les tables et insérer les données
python3 create_db.py
```

---

## Utilisation

### Démarrer l'API

```bash
uvicorn app:app --reload
```

L'API est accessible sur `http://localhost:8000`.

### Endpoints

| Méthode | URL | Description |
|---|---|---|
| GET | `/` | Vérification que l'API est opérationnelle |
| POST | `/predict` | Prédiction du risque de churn |

### Exemple d'appel `/predict`

Interface interactive disponible sur `http://localhost:8000/docs` (Swagger UI).

Exemple de corps de requête JSON :

```json
{
  "genre": "Homme",
  "statut_marital": "Célibataire",
  "departement": "Ventes",
  "poste": "Représentant commercial",
  "domaine_etude": "Marketing",
  "frequence_deplacement": "Frequent",
  "augementation_salaire_precedente": "11 %",
  "heure_supplementaires": "Oui",
  "revenu_mensuel": 2500,
  "satisfaction_employee_environnement": 2,
  "satisfaction_employee_nature_travail": 2,
  "satisfaction_employee_equipe": 3,
  "satisfaction_employee_equilibre_pro_perso": 1,
  "note_evaluation_precedente": 3,
  "note_evaluation_actuelle": 4,
  "niveau_hierarchique_poste": 1,
  "niveau_education": 2,
  "nb_formations_suivies": 2,
  "nombre_participation_pee": 0,
  "nombre_experiences_precedentes": 5,
  "annees_dans_l_entreprise": 2,
  "annees_dans_le_poste_actuel": 1,
  "annees_depuis_la_derniere_promotion": 1,
  "annes_sous_responsable_actuel": 1,
  "age": 28.0,
  "annee_experience_totale": 6.0,
  "distance_domicile_travail": 20.0
}
```

Réponse :

```json
{
  "prediction": 1,
  "probabilite_churn": 0.847,
  "interpretation": "Risque de churn"
}
```

---

## Base de données

### Schéma

Le script `create_db.py` crée deux tables dans PostgreSQL :

**`employees`** — contient les 1 470 employés du jeu de données RH (importés depuis les fichiers CSV).

**`prediction_logs`** — journalise chaque appel à `/predict` avec :
- les 27 variables d'entrée fournies à l'API
- la prédiction (`0` ou `1`)
- la probabilité de churn
- l'interprétation textuelle
- l'horodatage automatique (`created_at`)

### Connexion

La base de données est configurée via une variable d'environnement :

```bash
export DATABASE_URL=postgresql://localhost/churn_db
```

La valeur par défaut (développement local) est `postgresql://localhost/churn_db`.  
En production sur Hugging Face Spaces, `DATABASE_URL` est stockée comme secret dans les paramètres du Space.

### Vérifier les données

```bash
# Nombre d'employés importés
psql churn_db -c "SELECT COUNT(*) FROM employees;"

# Dernières prédictions enregistrées
psql churn_db -c "SELECT id, created_at, prediction, probabilite_churn FROM prediction_logs ORDER BY created_at DESC LIMIT 5;"
```

---

## Sécurité

- **Aucun secret hardcodé** : `DATABASE_URL` est toujours lue depuis les variables d'environnement, jamais écrite en dur dans le code.
- **Secrets en production** : le token Hugging Face (`HF_TOKEN`) et `DATABASE_URL` sont stockés comme secrets chiffrés dans les paramètres du Space Hugging Face — ils ne sont jamais exposés dans le code ni dans les logs.
- **Variables sensibles hors dépôt** : `.env`, `.venv/`, `.coverage`, `htmlcov/` sont listés dans `.gitignore` et ne sont jamais commités.

---

## Tests

```bash
# Lancer tous les tests
pytest

# Avec rapport de couverture
pytest --cov=. --cov-report=html
```

Le rapport HTML est généré dans `htmlcov/index.html`.

Les tests couvrent :
- La création des tables PostgreSQL (via SQLite en mémoire)
- L'insertion et la lecture d'un employé
- La contrainte d'unicité sur `id_employee`
- L'insertion d'un log de prédiction complet
- Le remplissage automatique de `created_at`
- Le pipeline de prédiction du modèle

---

## Déploiement (Hugging Face Spaces)

Le déploiement est automatisé via GitHub Actions (`.github/workflows/`).

À chaque push sur `main` :
1. Les tests sont exécutés
2. Si les tests passent, le code est déployé automatiquement sur Hugging Face Spaces

### Configuration requise

Dans les secrets GitHub du dépôt (`Settings > Secrets and variables > Actions`) :

| Secret | Valeur |
|---|---|
| `HF_TOKEN` | Token d'accès Hugging Face (write) |

Dans les secrets du Space Hugging Face (`Settings > Variables and secrets`) :

| Secret | Valeur |
|---|---|
| `DATABASE_URL` | URL de connexion PostgreSQL en production |

### URL de l'application déployée

`https://huggingface.co/spaces/elch99/projet5`

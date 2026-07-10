import os
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Literal
from joblib import load
import pandas as pd
import json
from features import feature_engineering
from sqlalchemy.orm import Session

API_KEY = os.getenv("API_KEY", "dev-key-local")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")


try: 
    from create_db import engine, PredictionLog
    DB_AVAILABLE = True
except Exception as e:
    print(f"Database non disponible: {e}")
    DB_AVAILABLE = False

# Chargement du modèle et des paramètres
pipeline = load("model_pipeline.joblib")

with open("train_params.json", "r") as f:
    params = json.load(f)

median_salary_by_level = {int(k): v for k, v in params["median_salary_by_level"].items()}
distance_threshold = params["distance_threshold"]

app = FastAPI(title="Churn Prediction API")

class EmployeeData(BaseModel):
    # Catégorielles — valeurs exactes acceptées par le modèle
    genre: Literal["F", "M"]
    statut_marital: Literal["Célibataire", "Divorcé(e)", "Marié(e)"]
    departement: Literal["Commercial", "Consulting", "Ressources Humaines"]
    poste: Literal["Assistant de Direction", "Cadre Commercial", "Consultant",
                   "Directeur Technique", "Manager", "Représentant Commercial",
                   "Ressources Humaines", "Senior Manager", "Tech Lead"]
    domaine_etude: Literal["Autre", "Entrepreunariat", "Infra & Cloud",
                           "Marketing", "Ressources Humaines", "Transformation Digitale"]
    frequence_deplacement: Literal["Aucun", "Frequent", "Occasionnel"]
    augementation_salaire_precedente: Literal["11 %", "12 %", "13 %", "14 %", "15 %",
                                              "16 %", "17 %", "18 %", "19 %", "20 %",
                                              "21 %", "22 %", "23 %", "24 %", "25 %"]
    heure_supplementaires: Literal["Oui", "Non"]
    # Entiers
    revenu_mensuel: int
    satisfaction_employee_environnement: int
    satisfaction_employee_nature_travail: int
    satisfaction_employee_equipe: int
    satisfaction_employee_equilibre_pro_perso: int
    note_evaluation_precedente: int
    note_evaluation_actuelle: int
    niveau_hierarchique_poste: int
    niveau_education: int
    nb_formations_suivies: int
    nombre_participation_pee: int = 0
    nombre_experiences_precedentes: int
    annees_dans_l_entreprise: int
    annees_dans_le_poste_actuel: int
    annees_depuis_la_derniere_promotion: int
    annes_sous_responsable_actuel: int
    # Flottants
    age: float
    annee_experience_totale: float
    distance_domicile_travail: float

@app.get("/")
def root():
    return {"message": "API opérationnelle ✅"}

@app.post("/predict",
    responses={
        401: {"description": "Not authenticated", "content": {"application/json": {"example": {"detail": "Not authenticated"}}}},
        403: {"description": "Clé API invalide", "content": {"application/json": {"example": {"detail": "Clé API invalide"}}}},
        500: {"description": "Internal Server Error", "content": {"application/json": {"example": {"detail": "Internal Server Error"}}}},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "Risque élevé": {
                            "summary": "Profil 1 — Jeune commercial surchargé, mal payé",
                            "value": {
                                "genre": "M", "statut_marital": "Célibataire", "departement": "Commercial",
                                "poste": "Représentant Commercial", "domaine_etude": "Marketing",
                                "frequence_deplacement": "Frequent", "augementation_salaire_precedente": "11 %",
                                "heure_supplementaires": "Oui", "revenu_mensuel": 1200,
                                "satisfaction_employee_environnement": 1, "satisfaction_employee_nature_travail": 1,
                                "satisfaction_employee_equipe": 2, "satisfaction_employee_equilibre_pro_perso": 1,
                                "note_evaluation_precedente": 3, "note_evaluation_actuelle": 3,
                                "niveau_hierarchique_poste": 1, "niveau_education": 2, "nb_formations_suivies": 1,
                                "nombre_participation_pee": 0, "nombre_experiences_precedentes": 6,
                                "annees_dans_l_entreprise": 2, "annees_dans_le_poste_actuel": 1,
                                "annees_depuis_la_derniere_promotion": 2, "annes_sous_responsable_actuel": 1,
                                "age": 26.0, "annee_experience_totale": 4.0, "distance_domicile_travail": 25.0
                            }
                        },
                        "Risque faible": {
                            "summary": "Profil 2 — Manager senior épanoui",
                            "value": {
                                "genre": "F", "statut_marital": "Marié(e)", "departement": "Consulting",
                                "poste": "Senior Manager", "domaine_etude": "Transformation Digitale",
                                "frequence_deplacement": "Aucun", "augementation_salaire_precedente": "20 %",
                                "heure_supplementaires": "Non", "revenu_mensuel": 15000,
                                "satisfaction_employee_environnement": 4, "satisfaction_employee_nature_travail": 4,
                                "satisfaction_employee_equipe": 4, "satisfaction_employee_equilibre_pro_perso": 4,
                                "note_evaluation_precedente": 4, "note_evaluation_actuelle": 4,
                                "niveau_hierarchique_poste": 4, "niveau_education": 4, "nb_formations_suivies": 3,
                                "nombre_participation_pee": 3, "nombre_experiences_precedentes": 2,
                                "annees_dans_l_entreprise": 15, "annees_dans_le_poste_actuel": 6,
                                "annees_depuis_la_derniere_promotion": 2, "annes_sous_responsable_actuel": 8,
                                "age": 42.0, "annee_experience_totale": 20.0, "distance_domicile_travail": 5.0
                            }
                        },
                        "Risque moyen": {
                            "summary": "Profil 3 — Consultant mid-level, quelques signaux",
                            "value": {
                                "genre": "M", "statut_marital": "Divorcé(e)", "departement": "Consulting",
                                "poste": "Consultant", "domaine_etude": "Infra & Cloud",
                                "frequence_deplacement": "Occasionnel", "augementation_salaire_precedente": "14 %",
                                "heure_supplementaires": "Oui", "revenu_mensuel": 5000,
                                "satisfaction_employee_environnement": 2, "satisfaction_employee_nature_travail": 3,
                                "satisfaction_employee_equipe": 3, "satisfaction_employee_equilibre_pro_perso": 2,
                                "note_evaluation_precedente": 3, "note_evaluation_actuelle": 3,
                                "niveau_hierarchique_poste": 2, "niveau_education": 3, "nb_formations_suivies": 2,
                                "nombre_participation_pee": 1, "nombre_experiences_precedentes": 4,
                                "annees_dans_l_entreprise": 5, "annees_dans_le_poste_actuel": 3,
                                "annees_depuis_la_derniere_promotion": 4, "annes_sous_responsable_actuel": 3,
                                "age": 34.0, "annee_experience_totale": 10.0, "distance_domicile_travail": 15.0
                            }
                        }
                    }
                }
            }
        }
    }
)

def predict(data: EmployeeData, _ = Security(verify_api_key)):
    # 1. Conversion en DataFrame
    df = pd.DataFrame([data.model_dump()]) # transformation de l'objet EmployeeData reçu par l'API en dictionnaire Python (27 champs input)

    # 2. Feature engineering (crée toutes les colonnes dérivées)
    df = feature_engineering(df, median_salary_by_level, distance_threshold)

    # 3. Suppression des colonnes brutes remplacées par leurs dérivées
    cols_supp = ['revenu_mensuel', 'augementation_salaire_precedente', 
                 'heure_supplementaires', 'haut_performer']
    df = df.drop(columns=cols_supp)

    # 4. Application de la fonction Feature_Engineering (typage)
    from features import Feature_Engineering
    df = Feature_Engineering(df)

    # 5. Prédiction
    probabilite = float(pipeline.predict_proba(df)[0][1])
    prediction = int(probabilite >= 0.48)


    if DB_AVAILABLE:
        with Session(engine) as session:
            log = PredictionLog(
                **data.model_dump(), # dictionnaire python "déplié" pour passer chaque champ comme argument séparé à PredictionLog(...)
                prediction=prediction,
                probabilite_churn=probabilite,
                interpretation="Risque de churn" if prediction == 1
                else "Pas de risque de churn"
            )
            session.add(log)
            session.commit()

    return {
        "prediction": prediction,
        "probabilite_churn": round(probabilite, 3),
        "interpretation": "Risque de churn" if prediction == 1 else "Pas de risque de churn"
    }


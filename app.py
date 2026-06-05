from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import json
from features import feature_engineering
from sqlalchemy.orm import Session

try: 
    from create_db import engine, PredictionLog
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# Chargement du modèle et des paramètres
pipeline = load("model_pipeline.joblib")

with open("train_params.json", "r") as f:
    params = json.load(f)

median_salary_by_level = {int(k): v for k, v in params["median_salary_by_level"].items()}
distance_threshold = params["distance_threshold"]

app = FastAPI(title="Churn Prediction API")

class EmployeeData(BaseModel):
    # Catégorielles
    genre: str
    statut_marital: str
    departement: str
    poste: str
    domaine_etude: str
    frequence_deplacement: str
    augementation_salaire_precedente: str  # ex: "15 %"
    heure_supplementaires: str             # "Oui" ou "Non"
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
    nombre_participation_pee: int
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

@app.post("/predict")
def predict(data: EmployeeData):
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
    prediction = int(pipeline.predict(df)[0])
    probabilite = float(pipeline.predict_proba(df)[0][1])

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
from joblib import load
import numpy as np
import pandas as pd
import pytest

# Chargement du modèle une seule fois pour tous les tests
pipeline = load("model_pipeline.joblib")

# --- Données de test minimales (une seule ligne fictive) ---
# Ces colonnes correspondent à X_test après Feature_Engineering
sample = pd.DataFrame([{
    "note_evaluation_precedente": 3,
    "note_evaluation_actuelle": 3,
    "satisfaction_employee_environnement": 3,
    "satisfaction_employee_nature_travail": 3,
    "satisfaction_employee_equipe": 3,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "annees_dans_l_entreprise": 5,
    "annees_dans_le_poste_actuel": 2,
    "annees_depuis_la_derniere_promotion": 1,
    "annes_sous_responsable_actuel": 2,
    "nb_formations_suivies": 2,
    "niveau_hierarchique_poste": 2,
    "distance_domicile_travail": 10,
    "nombre_participation_pee": 3,
    "age": 32,
    "annee_experience_totale": 8,
    "poste": "Manager",
    "departement": "Ventes",
    "domaine_etude": "Informatique",
    "niveau_education": 3,
    "genre": "Homme",
    "statut_marital": "Célibataire",
    "nombre_experiences_precedentes": 2,
    "frequence_deplacement": "Rarement",
    "satisfaction_globale": 3.0,
    "gap_perf_satisfaction": 0.0,
    "evolution_performance": 0,
    "pee_par_an": 0.5,
    "ratio_poste_anciennete": 0.33,
    "ratio_manager": 0.33,
    "formation_par_an": 0.33,
    "revenu_relatif": 1.0,
    "sous_paye": False,
    "augmentation_pct": 15,
    "faible_augmentation": False,
    "jeune": False,
    "heures_supp": False,
    "distance_penible": False,
    "mobilite_forte": False,
}])


def test_model_loads():
    """Vérifie que le modèle se charge correctement"""
    assert pipeline is not None


def test_model_has_predict():
    """Vérifie que le pipeline a bien une méthode predict"""
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_output_predict_proba():
    """Vérifie que le nombre de prédictions = nombre de lignes en entrée"""
    proba = pipeline.predict_proba(sample)[:, 1]
    assert sample.shape[0] == len(proba)


def test_proba_between_0_and_1():
    """Vérifie que les probabilités sont bien entre 0 et 1"""
    proba = pipeline.predict_proba(sample)[:, 1]
    assert all(0 <= p <= 1 for p in proba)


def test_predict_output_is_binary():
    """Vérifie que les prédictions sont bien 0 ou 1"""
    predictions = pipeline.predict(sample)
    assert all(p in [0, 1] for p in predictions)
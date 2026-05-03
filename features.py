import pandas as pd
import numpy as np

# Ces valeurs sont calculées sur X_train et doivent être fixées ici
# pour éviter le data leakage en production
MEDIAN_SALARY_BY_LEVEL = None  # sera chargé depuis un fichier séparé
DISTANCE_THRESHOLD = None       # sera chargé depuis un fichier séparé


def create_satisfaction_globale(df):
    df = df.copy()
    df["satisfaction_globale"] = df[[
        "satisfaction_employee_environnement",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso"
    ]].mean(axis=1)
    return df

def create_gap_perf_satisfaction(df):
    df = df.copy()
    df["gap_perf_satisfaction"] = df["note_evaluation_actuelle"] - df["satisfaction_globale"]
    return df

def create_evolution_performance(df):
    df = df.copy()
    df["evolution_performance"] = df["note_evaluation_actuelle"] - df["note_evaluation_precedente"]
    return df

def create_haut_performer(df):
    df = df.copy()
    df["haut_performer"] = (df["note_evaluation_actuelle"] >= 4).astype(int)
    return df

def create_pee_par_an(df):
    df = df.copy()
    df["pee_par_an"] = df["nombre_participation_pee"] / (df["annees_dans_l_entreprise"] + 1)
    return df

def create_ratio_poste_anciennete(df):
    df = df.copy()
    df["ratio_poste_anciennete"] = df["annees_dans_le_poste_actuel"] / (df["annees_dans_l_entreprise"] + 1)
    return df

def create_blocage_promotion(df):
    df = df.copy()
    df["blocage_promotion"] = (df["annees_depuis_la_derniere_promotion"] > 3).astype(int)
    return df

def create_ratio_manager(df):
    df = df.copy()
    df["ratio_manager"] = df["annes_sous_responsable_actuel"] / (df["annees_dans_l_entreprise"] + 1)
    return df

def create_formation_par_an(df):
    df = df.copy()
    df["formation_par_an"] = df["nb_formations_suivies"] / (df["annees_dans_l_entreprise"] + 1)
    return df

def create_revenu_relatif(df, median_salary_by_level):
    df = df.copy()
    df["revenu_relatif"] = df["revenu_mensuel"] / df["niveau_hierarchique_poste"].map(median_salary_by_level)
    return df

def create_sous_paye(df):
    df = df.copy()
    df["sous_paye"] = (df["revenu_relatif"] < 0.9).astype(int)
    return df

def create_augmentation_pct(df):
    df = df.copy()
    df["augmentation_pct"] = df["augementation_salaire_precedente"].str.replace(" %", "", regex=False).astype(int)
    return df

def create_faible_augmentation(df):
    df = df.copy()
    df["faible_augmentation"] = (df["augmentation_pct"] <= 12).astype(int)
    return df

def create_jeune(df):
    df = df.copy()
    df["jeune"] = (df["age"] < 30).astype(int)
    return df

def create_heures_supp(df):
    df = df.copy()
    df["heures_supp"] = (df["heure_supplementaires"] == "Oui").astype(int)
    return df

def create_distance_penible(df, distance_threshold):
    df = df.copy()
    df["distance_penible"] = (df["distance_domicile_travail"] > distance_threshold).astype(int)
    return df

def create_mobilite_forte(df):
    df = df.copy()
    df["mobilite_forte"] = (df["frequence_deplacement"] == "Frequent").astype(int)
    return df


def feature_engineering(df, median_salary_by_level, distance_threshold):
    """
    Applique tout le feature engineering sur un DataFrame brut.
    
    median_salary_by_level : dict {niveau: mediane_salaire} calculé sur X_train
    distance_threshold : float, quantile 0.75 de distance calculé sur X_train
    """
    df = create_satisfaction_globale(df)
    df = create_gap_perf_satisfaction(df)
    df = create_evolution_performance(df)
    df = create_haut_performer(df)
    df = create_pee_par_an(df)

    df = create_ratio_poste_anciennete(df)
    df = create_blocage_promotion(df)
    df = create_ratio_manager(df)
    df = create_formation_par_an(df)

    df = create_revenu_relatif(df, median_salary_by_level)
    df = create_sous_paye(df)
    df = create_augmentation_pct(df)
    df = create_faible_augmentation(df)

    df = create_jeune(df)
    df = create_heures_supp(df)
    df = create_distance_penible(df, distance_threshold)
    df = create_mobilite_forte(df)

    return df
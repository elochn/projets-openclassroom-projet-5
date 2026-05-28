import os
import json
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import datetime, timezone


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/churn_db")
engine = create_engine(DATABASE_URL.replace("postgresql://","postgresql+psycopg://"))

class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_employee = Column(Integer, unique=True, nullable=False)

    # Données SIRH
    age = Column(Float)
    genre = Column(String(10))
    revenu_mensuel = Column(Integer)
    statut_marital = Column(String(30))
    departement = Column(String(50))
    poste = Column(String(50))
    nombre_experiences_precedentes = Column(Integer)
    annee_experience_totale = Column(Float)
    annees_dans_l_entreprise = Column(Integer)
    annees_dans_le_poste_actuel = Column(Integer)

    # Données Évaluation
    satisfaction_employee_environnement = Column(Integer)
    note_evaluation_precedente = Column(Integer)
    niveau_hierarchique_poste = Column(Integer)
    satisfaction_employee_nature_travail = Column(Integer)
    satisfaction_employee_equipe = Column(Integer)
    satisfaction_employee_equilibre_pro_perso = Column(Integer)
    note_evaluation_actuelle = Column(Integer)
    heure_supplementaires = Column(String(5))
    augementation_salaire_precedente = Column(String(10))

    # Données Sondage
    a_quitte_l_entreprise = Column(String(5))
    nombre_participation_pee = Column(Integer)
    nb_formations_suivies = Column(Integer)
    distance_domicile_travail = Column(Float)
    niveau_education = Column(Integer)
    domaine_etude = Column(String(50))
    frequence_deplacement = Column(String(20))
    annees_depuis_la_derniere_promotion = Column(Integer)
    annes_sous_responsable_actuel = Column(Integer)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Inputs
    genre = Column(String(10))
    statut_marital = Column(String(30))
    departement = Column(String(50))
    poste = Column(String(50))
    domaine_etude = Column(String(50))
    frequence_deplacement = Column(String(20))
    augementation_salaire_precedente = Column(String(10))
    heure_supplementaires = Column(String(5))
    revenu_mensuel = Column(Integer)
    satisfaction_employee_environnement = Column(Integer)
    satisfaction_employee_nature_travail = Column(Integer)
    satisfaction_employee_equipe = Column(Integer)
    satisfaction_employee_equilibre_pro_perso = Column(Integer)
    note_evaluation_precedente = Column(Integer)
    note_evaluation_actuelle = Column(Integer)
    niveau_hierarchique_poste = Column(Integer)
    niveau_education = Column(Integer)
    nb_formations_suivies = Column(Integer)
    nombre_participation_pee = Column(Integer)
    nombre_experiences_precedentes = Column(Integer)
    annees_dans_l_entreprise = Column(Integer)
    annees_dans_le_poste_actuel = Column(Integer)
    annees_depuis_la_derniere_promotion = Column(Integer)
    annes_sous_responsable_actuel = Column(Integer)
    age = Column(Float)
    annee_experience_totale = Column(Float)
    distance_domicile_travail = Column(Float)

    # Outputs
    prediction = Column(Integer)
    probabilite_churn = Column(Float)
    interpretation = Column(String(50))


def create_tables():
    Base.metadata.create_all(engine)
    print("Tables créées.")


def compute_and_save_train_params():
    sirh = pd.read_csv("extrait_sirh.csv")
    sondage = pd.read_csv("extrait_sondage.csv")
    eval = pd.read_csv("extrait_eval.csv")
    df = pd.concat([sirh, sondage, eval], axis=1)

    median_salary_by_level = (
        df.groupby("niveau_hierarchique_poste")["revenu_mensuel"]
        .median()
        .astype(int)
        .to_dict()
    )
    distance_threshold = float(df["distance_domicile_travail"].quantile(0.75))

    params = {
        "median_salary_by_level": median_salary_by_level,
        "distance_threshold": distance_threshold
    }

    with open("train_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print("train_params.json créé.")


def load_csv_to_db():
    sirh = pd.read_csv("extrait_sirh.csv")
    eval_df = pd.read_csv("extrait_eval.csv")
    sondage = pd.read_csv("extrait_sondage.csv")
    df = pd.concat([sirh, eval_df, sondage], axis=1)

    with Session(engine) as session:
        for _, row in df.iterrows():
            emp = Employee(
                id_employee=row["id_employee"],
                age=row["age"],
                genre=row["genre"],
                revenu_mensuel=row["revenu_mensuel"],
                statut_marital=row["statut_marital"],
                departement=row["departement"],
                poste=row["poste"],

                nombre_experiences_precedentes=row["nombre_experiences_precedentes"],
                annee_experience_totale=row["annee_experience_totale"],
                annees_dans_l_entreprise=row["annees_dans_l_entreprise"],
                annees_dans_le_poste_actuel=row["annees_dans_le_poste_actuel"],
                satisfaction_employee_environnement=row["satisfaction_employee_environnement"],
                note_evaluation_precedente=row["note_evaluation_precedente"],
                niveau_hierarchique_poste=row["niveau_hierarchique_poste"],
                satisfaction_employee_nature_travail=row["satisfaction_employee_nature_travail"],
                satisfaction_employee_equipe=row["satisfaction_employee_equipe"],
                satisfaction_employee_equilibre_pro_perso=row["satisfaction_employee_equilibre_pro_perso"],
                note_evaluation_actuelle=row["note_evaluation_actuelle"],
                heure_supplementaires=row["heure_supplementaires"],
                augementation_salaire_precedente=row["augementation_salaire_precedente"],
                a_quitte_l_entreprise=row["a_quitte_l_entreprise"],
                nombre_participation_pee=row["nombre_participation_pee"],
                nb_formations_suivies=row["nb_formations_suivies"],
                distance_domicile_travail=row["distance_domicile_travail"],
                niveau_education=row["niveau_education"],
                domaine_etude=row["domaine_etude"],
                frequence_deplacement=row["frequence_deplacement"],
                annees_depuis_la_derniere_promotion=row["annees_depuis_la_derniere_promotion"],
                annes_sous_responsable_actuel=row["annes_sous_responsable_actuel"],
            )
            session.add(emp)
        session.commit()
        
    print(f"{len(df)} employés insérés dans la base.")
    

if __name__ == "__main__":
    create_tables()
    compute_and_save_train_params()
    load_csv_to_db()
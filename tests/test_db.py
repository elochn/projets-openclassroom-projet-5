import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session
from create_db import Base, Employee, PredictionLog

# La fixture test_engine
@pytest.fixture
def test_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

# La fixture session
@pytest.fixture
def session(test_engine):
    with Session(test_engine) as s:
        yield s

# Premier test: Création de tables
def test_tables_created(test_engine):
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()
    assert "employees" in tables
    assert "prediction_logs" in tables

# Test d'insertion et d'enregistrement d'un employé
def test_insert_employee(session):
    emp = Employee(
        id_employee=999,
        age=30.0,
        genre="F",
        revenu_mensuel=4000,
        statut_marital="Célibataire",
        departement="Commercial",
        poste="Consultant",
        nombre_experiences_precedentes=3,
        annee_experience_totale=5.0,
        annees_dans_l_entreprise=2,
        annees_dans_le_poste_actuel=1,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=3,
        niveau_hierarchique_poste=2,
        satisfaction_employee_nature_travail=3,
        satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=2,
        note_evaluation_actuelle=4,
        heure_supplementaires="Non",
        augementation_salaire_precedente="15 %",
        a_quitte_l_entreprise="Non",
        nombre_participation_pee=1,
        nb_formations_suivies=2,
        distance_domicile_travail=10.0,
        niveau_education=3,
        domaine_etude="Marketing",
        frequence_deplacement="Occasionnel",
        annees_depuis_la_derniere_promotion=1,
        annes_sous_responsable_actuel=2,
    )
    session.add(emp)
    session.commit()

    resultat = session.get(Employee, 1)
    assert resultat is not None
    assert resultat.genre == "F"
    assert resultat.id_employee == 999

# Test de contrainte unique
def test_employee_id_unique(session):
    emp1 = Employee(id_employee=1, genre="F") # on essaie d'insérer deux employés avec le même identifiant id_employee=1
    emp2 = Employee(id_employee=1, genre="M")
    session.add(emp1)
    session.commit()
    session.add(emp2)
    with pytest.raises(Exception): # on s'attend à ce que cette ligne provoque une erreur, auquel cas, test réussi
        session.commit()

# Test d'insertion et d'enregistrement PredictionLog
def test_insert_prediction_log(session):
    log = PredictionLog(
        genre="M",
        statut_marital="Marié(e)",
        departement="Consulting",
        poste="Manager",
        domaine_etude="Autre",
        frequence_deplacement="Frequent",
        augementation_salaire_precedente="20 %",
        heure_supplementaires="Oui",
        revenu_mensuel=6000,
        satisfaction_employee_environnement=2,
        satisfaction_employee_nature_travail=2,
        satisfaction_employee_equipe=1,
        satisfaction_employee_equilibre_pro_perso=1,
        note_evaluation_precedente=3,
        note_evaluation_actuelle=4,
        niveau_hierarchique_poste=3,
        niveau_education=4,
        nb_formations_suivies=1,
        nombre_participation_pee=0,
        nombre_experiences_precedentes=7,
        annees_dans_l_entreprise=8,
        annees_dans_le_poste_actuel=3,
        annees_depuis_la_derniere_promotion=4,
        annes_sous_responsable_actuel=5,
        age=40.0,
        annee_experience_totale=12.0,
        distance_domicile_travail=30.0,
        prediction=1,
        probabilite_churn=0.78,
        interpretation="Risque de churn",
    )
    session.add(log)
    session.commit()

    resultat = session.query(PredictionLog).first()
    assert resultat is not None
    assert resultat.prediction == 1
    assert resultat.probabilite_churn == 0.78
    assert resultat.interpretation == "Risque de churn"

# Test d'affichage automatique de l'horodatage sans mention de "created_at"
def test_prediction_log_created_at_auto(session):
    log = PredictionLog(
        genre="F",
        prediction=0,
        probabilite_churn=0.3,
        interpretation="Pas de risque de churn",
    )
    session.add(log)
    session.commit()

    resultat = session.query(PredictionLog).first()
    assert resultat.created_at is not None

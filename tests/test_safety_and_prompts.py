"""Tests des contrôles de sécurité et du formatage des prompts."""

from triage.eval.safety import (
    has_dangerous_advice,
    is_dangerous_undertriage,
    safety_report,
)
from triage.prompts import IM_END, IM_START, render_chatml
from triage.serve.triage_agent import extract_triage_level


# --- Sécurité ---
def test_sous_triage_dangereux():
    """Classer une urgence vitale en différé est un sous-triage dangereux."""
    assert is_dangerous_undertriage("CONSULTATION_DIFFEREE", "URGENCE_VITALE") is True
    assert is_dangerous_undertriage("URGENCE_VITALE", "URGENCE_VITALE") is False
    # Sur-triage (plus prudent) : non dangereux.
    assert is_dangerous_undertriage("URGENCE_VITALE", "URGENCE_MODEREE") is False


def test_conseil_dangereux():
    """« Rentrez chez vous » pour un cas grave est un conseil dangereux."""
    assert has_dangerous_advice("Rentrez chez vous, rien de grave.", "URGENCE_VITALE")
    # Pour un cas bénin, le même conseil est acceptable.
    assert not has_dangerous_advice("Reposez-vous.", "CONSULTATION_DIFFEREE")


def test_safety_report_agrege():
    preds = [
        {"predicted_level": "CONSULTATION_DIFFEREE", "expected_level": "URGENCE_VITALE", "text": "rentrez chez vous"},
        {"predicted_level": "URGENCE_VITALE", "expected_level": "URGENCE_VITALE", "text": "urgence"},
    ]
    rep = safety_report(preds)
    assert rep["dangerous_undertriage"] == 1
    assert rep["vital_sensitivity"] == 0.5
    # Le cas 1 est un sous-triage CRITIQUE (vital manqué).
    assert rep["critical_undertriage"] == 1


def test_sous_triage_modere_non_critique():
    """Un modéré classé différé est un sous-triage, mais PAS critique."""
    rep = safety_report([
        {"predicted_level": "CONSULTATION_DIFFEREE",
         "expected_level": "URGENCE_MODEREE", "text": ""},
    ])
    assert rep["dangerous_undertriage"] == 1
    assert rep["critical_undertriage"] == 0


# --- Prompts ---
def test_render_chatml_avec_reponse():
    txt = render_chatml("Bonjour", "Réponse")
    assert IM_START in txt and IM_END in txt
    assert "Réponse" in txt


def test_render_chatml_prompt_seul():
    """Sans réponse, le texte se termine sur l'ouverture de l'assistant."""
    txt = render_chatml("Bonjour", assistant_content=None)
    assert txt.endswith("assistant\n")


# --- Extraction du niveau ---
def test_extraction_niveau_explicite():
    assert extract_triage_level("Niveau : URGENCE_VITALE car…") == "URGENCE_VITALE"


def test_extraction_niveau_langage_naturel():
    assert extract_triage_level("Ceci relève de soins différés.") == "CONSULTATION_DIFFEREE"


def test_extraction_niveau_absent():
    assert extract_triage_level("Bonjour, comment puis-je aider ?") is None


def test_extraction_negation_pas_d_urgence_vitale():
    """« pas une urgence vitale » ne doit PAS être lu comme URGENCE_VITALE
    (bug historique : les cas bénins basculaient en urgence vitale)."""
    txt = "Ce n'est pas une urgence vitale : soins différés possibles."
    assert extract_triage_level(txt) == "CONSULTATION_DIFFEREE"


def test_extraction_non_urgent():
    """« non urgent » doit donner DIFFEREE (avant : `\\burgent\\b` → MODEREE)."""
    txt = "Situation non urgente, consultation possible demain."
    assert extract_triage_level(txt) == "CONSULTATION_DIFFEREE"


def test_extraction_ligne_structuree_prioritaire():
    """La ligne « Niveau de priorité : X » du gabarit prime sur le reste du texte."""
    txt = ("Niveau de priorité : CONSULTATION_DIFFEREE (Soins différés)\n"
           "Analyse : pas d'urgence vitale immédiate.")
    assert extract_triage_level(txt) == "CONSULTATION_DIFFEREE"


# --- Sur-triage (nouvelle métrique) ---
def test_sur_triage_mesure():
    from triage.eval.safety import is_overtriage

    assert is_overtriage("URGENCE_VITALE", "CONSULTATION_DIFFEREE") is True
    assert is_overtriage("URGENCE_VITALE", "URGENCE_VITALE") is False
    rep = safety_report([
        {"predicted_level": "URGENCE_VITALE",
         "expected_level": "CONSULTATION_DIFFEREE", "text": ""},
    ])
    assert rep["overtriage"] == 1 and rep["overtriage_rate"] == 1.0


# --- Nettoyage des marqueurs d'anonymisation en sortie ---
def test_clean_output_retire_marqueurs():
    from triage.serve.inference import clean_output

    txt = "Niveau : URGENCE_VITALE <PERSONNE>. Contact <EMAIL> ou <TELEPHONE>."
    cleaned = clean_output(txt)
    assert "<PERSONNE>" not in cleaned
    assert "<EMAIL>" not in cleaned
    assert "URGENCE_VITALE" in cleaned


def test_clean_output_coupe_fin_de_tour_et_think():
    """La sortie est coupée au premier fin/nouveau tour, blocs <think> retirés."""
    from triage.serve.inference import clean_output

    txt = ("<think>réflexion interne</think>Niveau : URGENCE_MODEREE"
           "<|im_end|>\n<|im_start|>user\nautre tour halluciné")
    assert clean_output(txt) == "Niveau : URGENCE_MODEREE"

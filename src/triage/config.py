"""Configuration centrale du projet.

Ce module définit **un seul endroit** où sont déclarés :

  * les chemins du projet (données, modèles, rapports) ;
  * les constantes métier (le modèle de base, les niveaux de triage) ;
  * les paramètres réglables via variables d'environnement / fichier `.env`.

Centraliser la configuration évite les "chemins en dur" disséminés dans le
code, ce qui rend le projet plus facile à déplacer, tester et auditer.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 1. Chemins du projet
# ---------------------------------------------------------------------------
# `__file__` = .../src/triage/config.py  →  on remonte de 3 niveaux pour
# obtenir la racine du dépôt (.../OC_P14). Tout le reste en découle.
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # corpus bruts téléchargés
INTERIM_DIR = DATA_DIR / "interim"         # données intermédiaires (avant anonymisation)
PROCESSED_DIR = DATA_DIR / "processed"     # datasets finaux, prêts à l'emploi
SFT_DIR = PROCESSED_DIR / "sft"            # paires instruction-réponse (SFT)
DPO_DIR = PROCESSED_DIR / "dpo"            # paires préférentielles (DPO)
EVAL_DIR = DATA_DIR / "eval"               # jeu d'évaluation clinique séparé

MODELS_DIR = ROOT_DIR / "models"           # poids / adaptateurs entraînés
REPORTS_DIR = ROOT_DIR / "reports"         # rapports & figures


# ---------------------------------------------------------------------------
# 2. Constantes métier (triage)
# ---------------------------------------------------------------------------
# Modèle de base imposé par la mission. On le rend surchargeable par variable
# d'environnement pour pouvoir tester un modèle plus petit en CI (cf. Settings).
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

# Les trois niveaux de priorité demandés par la Dr. Dubois. On s'appuie sur une
# échelle inspirée des standards de triage hospitalier (type ESI / FRENCH).
# La valeur entière sert au tri ; le libellé est destiné aux humains.
TRIAGE_LEVELS: dict[str, dict[str, str]] = {
    "URGENCE_VITALE": {
        "code": "1",
        "label_fr": "Urgence maximale",
        "label_en": "Immediate / life-threatening",
        "delai": "Prise en charge immédiate (< 1 min)",
        "couleur": "rouge",
    },
    "URGENCE_MODEREE": {
        "code": "2",
        "label_fr": "Urgence modérée",
        "label_en": "Urgent",
        "delai": "Prise en charge rapide (< 20 min)",
        "couleur": "orange",
    },
    "CONSULTATION_DIFFEREE": {
        "code": "3",
        "label_fr": "Soins différés",
        "label_en": "Non-urgent / deferred",
        "delai": "Prise en charge différée (< 2 h)",
        "couleur": "vert",
    },
}

# Avertissement médico-légal ajouté systématiquement aux réponses du modèle.
# Un POC ne remplace JAMAIS un avis médical : on le rappelle explicitement.
MEDICAL_DISCLAIMER = (
    "⚠️ Ce résultat est fourni par un système d'aide à la décision (POC) et ne "
    "remplace pas l'évaluation d'un professionnel de santé. En cas de doute ou "
    "d'aggravation, appelez le 15 (SAMU) ou rendez-vous aux urgences."
)


# ---------------------------------------------------------------------------
# 3. Paramètres surchargeables (variables d'environnement / .env)
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Paramètres d'exécution, lus depuis l'environnement ou un fichier `.env`.

    Exemple : exporter `TRIAGE_BASE_MODEL=Qwen/Qwen3-0.6B-Base` permet de faire
    tourner la CI sur un modèle minuscule sans toucher au code.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_",       # toutes les variables commencent par TRIAGE_
        env_file=".env",
        extra="ignore",
    )

    # Modèle de base et graine aléatoire (reproductibilité)
    base_model: str = DEFAULT_BASE_MODEL
    seed: int = 42

    # --- Configuration de l'API / du backend d'inférence ---
    # "transformers" : léger, pour les tests locaux (charge le modèle en mémoire).
    # "vllm"        : production, appelle un serveur vLLM via HTTP (OpenAI-like).
    inference_backend: str = "transformers"
    model_path: str = str(MODELS_DIR / "qwen3-triage-merged")  # poids fusionnés
    vllm_base_url: str = "http://localhost:8000/v1"            # endpoint vLLM
    vllm_model_name: str = "chsa-triage"

    # Génération — défauts validés en conditions réelles le 10/07 (page /ui) :
    # le décodage GLOUTON (température 0) donne des verdicts déterministes et
    # des labels propres. La repetition_penalty est neutralisée (1.0) : > 1,
    # elle pénalisait les tokens du label déjà émis et poussait le modèle vers
    # des variantes mal orthographiées (« URGENCE_MODERIEE ») ; c'est
    # no_repeat_ngram_size qui protège des boucles de génération.
    max_new_tokens: int = 512
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 6

    # Hôte / port d'écoute de l'API (surchargeables : TRIAGE_API_PORT, …)
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Journal d'audit (traçabilité des interactions, exigence de la mission)
    audit_log_path: str = str(ROOT_DIR / "logs" / "audit.log")


# Instance unique importable partout : `from triage.config import settings`.
settings = Settings()


def ensure_dirs() -> None:
    """Crée tous les dossiers de travail s'ils n'existent pas déjà.

    Appelé au début des scripts pour éviter les erreurs "dossier introuvable".
    """
    for directory in (
        RAW_DIR, INTERIM_DIR, SFT_DIR, DPO_DIR, EVAL_DIR, MODELS_DIR, REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

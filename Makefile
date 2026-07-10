# ============================================================================
#  Makefile — raccourcis pour les tâches courantes du projet
#  Usage : `make <cible>` (ex: `make test`). `make help` liste les cibles.
# ============================================================================
.DEFAULT_GOAL := help
.PHONY: help install data train-sft train-dpo merge eval api test lint format pipeline

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Installe l'environnement (UV) + extras quantization
	uv sync --extra quant
	# `spacy download` appelle pip, absent des venvs uv : on l'ajoute à la volée.
	uv run --with pip python -m spacy download fr_core_news_md
	uv run --with pip python -m spacy download en_core_web_md

data:  ## Construit tout le dataset (collecte → split)
	uv run triage-collect
	uv run triage-build-sft
	uv run triage-build-dpo
	uv run triage-anonymize
	uv run triage-split

train-sft:  ## Lance le fine-tuning supervisé (SFT + LoRA)
	uv run triage-train-sft --config configs/sft.yaml

train-dpo:  ## Lance l'alignement par préférences (DPO)
	uv run triage-train-dpo --config configs/dpo.yaml

merge:  ## Fusionne les adaptateurs → modèle déployable
	uv run python -m triage.train.merge

eval:  ## Évalue le modèle sur le jeu clinique
	TRIAGE_MODEL_PATH=models/qwen3-triage-merged uv run triage-evaluate

api:  ## Lance l'API de démonstration (backend local)
	TRIAGE_MODEL_PATH=models/qwen3-triage-merged uv run triage-api

test:  ## Exécute les tests unitaires
	uv run pytest tests -q

lint:  ## Vérifie le style du code (Ruff)
	uv run ruff check src tests

format:  ## Formate le code (Ruff)
	uv run ruff format src tests

pipeline:  ## Exécute tout le pipeline de bout en bout
	bash scripts/run_pipeline.sh

#!/usr/bin/env bash
# ============================================================================
#  Pipeline complet du POC, de bout en bout, en une commande.
#  Usage :  bash scripts/run_pipeline.sh
#
#  Chaque étape est idempotente : relancer le script régénère proprement les
#  artefacts. `set -e` interrompt au premier échec (on ne propage pas un bug).
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # se place à la racine du dépôt

echo "==============================================================="
echo " SEMAINE 1 — Préparation des données"
echo "==============================================================="
uv run triage-collect                 # télécharge les corpus médicaux
uv run triage-build-sft               # ~5 000 paires instruction-réponse
uv run triage-build-dpo               # paires préférentielles (DPO)
uv run triage-anonymize               # anonymisation RGPD (Presidio)
uv run triage-split                   # train / val / test + éval clinique

echo "==============================================================="
echo " SEMAINE 2 — Fine-tuning supervisé (SFT + LoRA)"
echo "==============================================================="
uv run triage-train-sft --config configs/sft.yaml

echo "==============================================================="
echo " SEMAINE 3 — Alignement par préférences (DPO)"
echo "==============================================================="
uv run triage-train-dpo --config configs/dpo.yaml

echo "==============================================================="
echo " Fusion des adaptateurs → modèle déployable"
echo "==============================================================="
uv run python -m triage.train.merge \
    --sft-adapter models/sft-lora \
    --dpo-adapter models/dpo-lora \
    --output models/qwen3-triage-merged

echo "==============================================================="
echo " SEMAINE 4 — Évaluation clinique"
echo "==============================================================="
TRIAGE_MODEL_PATH=models/qwen3-triage-merged uv run triage-evaluate

echo ""
echo "✅ Pipeline terminé. Pour lancer l'API de démonstration :"
echo "   TRIAGE_MODEL_PATH=models/qwen3-triage-merged uv run triage-api"

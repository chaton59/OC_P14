# 🏥 POC — Agent IA de Triage Médical (CHSA)

> Proof of Concept d'un assistant de triage aux urgences du **Centre Hospitalier
> Saint-Aurélien (CHSA)**, fondé sur le fine-tuning d'un modèle de langage
> **Qwen3-1.7B** (SFT + LoRA puis alignement **DPO**), exposé via une **API
> FastAPI / vLLM** et industrialisé avec un pipeline **CI/CD GitHub Actions**.

Ce dépôt est le livrable technique de la mission « Développez le POC d'un agent
de triage médical ». Il est conçu pour être **lu comme un cours** : le code est
abondamment commenté en français et chaque étape est expliquée.

---

## ⚠️ Avertissement médical

Ce projet est un **Proof of Concept à visée pédagogique**. Il ne constitue
**pas un dispositif médical** et ne doit **jamais** être utilisé pour une prise
de décision clinique réelle. En cas d'urgence, appelez le **15 (SAMU)**.

---

## 🗺️ Vue d'ensemble

L'agent accompagne le personnel soignant en :

1. **collectant les symptômes** via un questionnaire adaptatif ;
2. **évaluant la priorité** sur trois niveaux (urgence vitale / modérée / différée) ;
3. **expliquant** son évaluation en langage clair ;
4. **s'intégrant** au SIH via une API REST ;
5. **garantissant la traçabilité** de chaque interaction (journal d'audit).

La démarche suit les 4 semaines de la mission :

| Semaine | Objectif | Dossier / module |
|--------:|----------|------------------|
| 1 | Préparation des données (bilingue, RGPD) | `src/triage/data/` |
| 2 | Fine-tuning supervisé (SFT + LoRA) | `src/triage/train/sft.py` |
| 3 | Alignement par préférences (DPO) | `src/triage/train/dpo.py` |
| 4 | Déploiement (FastAPI/vLLM/Docker) + CI/CD | `src/triage/serve/`, `deploy/`, `.github/` |

---

## 🧱 Architecture du dépôt

```
OC_P14/
├── pyproject.toml          # dépendances & scripts (géré par UV)
├── README.md               # ce document
├── src/triage/             # code source du POC
│   ├── config.py           # configuration centrale (chemins, niveaux de triage)
│   ├── prompts.py          # gabarits de prompt (ChatML) + consigne système
│   ├── data/               # Semaine 1 : collecte, SFT, DPO, RGPD, splits
│   ├── train/              # Semaine 2-3 : entraînement SFT (LoRA) et DPO
│   ├── eval/               # évaluation des performances + contrôles de sécurité
│   ├── serve/              # Semaine 4 : agent de triage + API FastAPI
│   └── utils/              # utilitaires (logs, seed, E/S JSONL)
├── configs/                # hyperparamètres d'entraînement (YAML)
├── data/                   # datasets (non versionnés : voir .gitignore)
├── models/                 # poids & adaptateurs entraînés
├── deploy/                 # Dockerfile, docker-compose, config vLLM
├── tests/                  # tests unitaires (pytest)
├── notebooks/              # exploration des données
├── reports/                # rapport technique
└── .github/workflows/      # pipelines CI/CD
```

---

## 🚀 Démarrage rapide

### 1. Pré-requis

* Python 3.10–3.12, un **GPU NVIDIA** (≈ 12 Go de VRAM suffisent pour le POC),
* [**UV**](https://docs.astral.sh/uv/) installé (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 2. Installation de l'environnement

```bash
# Crée .venv et installe toutes les dépendances (cœur + quantization 4 bits)
uv sync --extra quant

# (optionnel) Modèles linguistiques spaCy pour l'anonymisation RGPD bilingue
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_md
```

### 3. Pipeline complet en une commande

```bash
bash scripts/run_pipeline.sh          # données → SFT → DPO → fusion → éval
```

… ou étape par étape :

```bash
# --- Semaine 1 : données ---
uv run triage-collect                 # télécharge les corpus médicaux
uv run triage-build-sft               # construit ~5 000 paires SFT
uv run triage-build-dpo               # construit les paires préférentielles
uv run triage-anonymize               # anonymisation RGPD (Presidio)
uv run triage-split                   # train / val / test + éval clinique

# --- Semaines 2-3 : entraînement ---
uv run triage-train-sft --config configs/sft.yaml
uv run triage-train-dpo --config configs/dpo.yaml

# --- Évaluation ---
uv run triage-evaluate

# --- Semaine 4 : API de démonstration (backend local transformers) ---
uv run triage-api
# → http://localhost:8080/docs  (documentation interactive OpenAPI)
# → http://localhost:8080/ui    (page web de démonstration : chat de triage)
```

---

## 🔌 Utiliser l'API

Une **page web de démonstration** (chat, questionnaire adaptatif, verdict
coloré) est servie par l'API elle-même sur **`/ui`** — c'est le support de
démo pour la soutenance. En ligne de commande :

```bash
curl -X POST http://localhost:8080/triage \
  -H "Content-Type: application/json" \
  -d '{"message": "douleur thoracique constrictive et essoufflement depuis 1h"}'
```

Réponse (extrait) :

```json
{
  "triage_level": "URGENCE_VITALE",
  "label": "Urgence maximale",
  "explanation": "Niveau de priorité : URGENCE_VITALE ...",
  "interaction_id": "…",
  "latency_ms": 842
}
```

---

## 🧪 Tests & qualité

```bash
uv run pytest          # tests unitaires
uv run ruff check .    # lint
uv run ruff format .   # formatage
```

---

## 🔒 Conformité RGPD

L'anonymisation s'appuie sur **Microsoft Presidio** (détection + masquage des
données personnelles). Le processus complet est documenté dans
[`reports/RGPD.md`](reports/RGPD.md).

---

## 📦 Déploiement

Voir [`deploy/README.md`](deploy/README.md) : conteneurisation Docker, serveur
d'inférence **vLLM** (compatible API OpenAI) et passerelle FastAPI. Le pipeline
CI/CD est décrit dans `.github/workflows/`.

---

## 🧠 Poids du modèle (livrable)

Les **poids finaux fusionnés** (Qwen3-1.7B + SFT LoRA + DPO, ~3,3 Go) sont
publiés sur le Hugging Face Hub, en dépôt **privé** (accès sur demande) :
**[ASI-Engineer/chsa-triage-qwen3-1.7b](https://huggingface.co/ASI-Engineer/chsa-triage-qwen3-1.7b)**

```bash
# Récupérer les poids sans ré-entraîner (après `hf auth login`) :
uv run hf download ASI-Engineer/chsa-triage-qwen3-1.7b \
    --local-dir models/qwen3-triage-merged
```

---

## 📄 Rapport technique

Le rapport de synthèse (méthodologie, métriques, analyse, roadmap) est dans
[`reports/rapport_technique.md`](reports/rapport_technique.md).

---

## 📜 Licence & sources de données

Code sous licence MIT. Chaque corpus conserve sa licence d'origine ; le détail
est consigné dans [`reports/SOURCES.md`](reports/SOURCES.md).

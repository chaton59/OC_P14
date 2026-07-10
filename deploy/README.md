# 🚢 Déploiement — Docker, vLLM & API

Ce dossier contient tout le nécessaire pour servir l'agent de triage en
conditions quasi-réelles : un serveur d'inférence **vLLM** (optimisé GPU) et une
**passerelle FastAPI** qui implémente la logique de triage.

```
┌────────────┐   HTTP/JSON   ┌──────────────┐   HTTP (OpenAI API)   ┌──────────┐
│  Client    │ ───────────►  │  API FastAPI │ ───────────────────►  │  vLLM    │
│ (SIH, web) │ ◄───────────  │ (agent)      │ ◄───────────────────  │ (modèle) │
└────────────┘               └──────────────┘                       └──────────┘
```

## Pourquoi vLLM ?

[vLLM](https://docs.vllm.ai) est un moteur d'inférence haute performance. Grâce
au *PagedAttention* et au *continuous batching*, il offre un **débit** et une
**latence** très supérieurs à une génération « naïve » avec Transformers — ce qui
compte pour un service d'urgences sous charge. Il expose une API **compatible
OpenAI**, d'où l'intégration simple côté passerelle.

## Pré-requis

* Docker + **NVIDIA Container Toolkit** (pour exposer le GPU aux conteneurs).
* Le **modèle fusionné** présent dans `../models/qwen3-triage-merged`
  (généré par `uv run python -m triage.train.merge`).

## Lancement (local, avec GPU)

```bash
# Depuis la racine du dépôt
docker compose -f deploy/docker-compose.yml up --build
```

* API de triage  → http://localhost:8080/docs
* API vLLM brute → http://localhost:8000/v1/models

Test :

```bash
curl -X POST http://localhost:8080/triage \
  -H "Content-Type: application/json" \
  -d '{"message":"douleur thoracique et essoufflement"}'
```

## Servir vLLM seul (sans Docker)

```bash
uv sync --extra serve
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/qwen3-triage-merged \
  --served-model-name chsa-triage \
  --max-model-len 2048
```

## Déploiement cloud

L'image de la passerelle est poussée sur le **GitHub Container Registry** par le
pipeline CI/CD (`.github/workflows/deploy.yml`). Pour un déploiement managé :

| Cible | Idée directrice |
|-------|-----------------|
| **VM GPU** (AWS `g5`, GCP `g2`, Azure `NC`) | `docker compose up -d` derrière un reverse-proxy TLS. |
| **HF Inference Endpoints** | Pousser le modèle fusionné sur le Hub et activer un endpoint vLLM. |
| **Kubernetes** | Un *Deployment* GPU pour vLLM + un *Deployment* pour l'API + *Service/Ingress*. |

> 🔐 **Secrets** : les jetons (HF, registry, TLS) ne sont jamais dans le code.
> Ils sont fournis via les *GitHub Secrets* (CI) ou le gestionnaire de secrets
> du cloud (runtime). Voir `.env.example` pour les variables attendues.

## Sécurité & supervision (checklist production)

- [ ] Authentification sur l'endpoint (clé API / OAuth) — absente en POC.
- [ ] TLS (HTTPS) via reverse-proxy (Traefik, Nginx).
- [ ] Limitation de débit (rate limiting) et quotas.
- [ ] Anonymisation des messages **avant** journalisation d'audit.
- [ ] Hébergement **HDS** (Hébergeur de Données de Santé) certifié.
- [ ] Supervision : métriques (latence, taux d'erreur), alertes, sauvegardes.

# Vérification des attendus de la mission — 6 juillet 2026

Passage en revue de chaque attendu de `projet/mission.txt` face au contenu réel
du dépôt. Verdict global : **les 5 livrables sont présents ; 2 réserves** (le
déploiement cloud effectif, et le dépôt git sans commit).

## Livrables principaux

| # | Attendu | État | Preuve dans le dépôt |
|---|---------|------|----------------------|
| 1 | Dataset médical bilingue nettoyé, structuré, anonymisé RGPD, optimisé SFT + DPO | ✅ Validé | 4 996 paires SFT (`data/interim/sft_anon.jsonl`), 2 220 paires DPO, fr+en, anonymisation Presidio (`anonymize.py`, `reports/RGPD.md`), licences (`reports/SOURCES.md`) |
| 2 | Modèle Qwen3-1.7B fine-tuné SFT+LoRA puis aligné DPO, poids fournis | ✅ Validé | `models/sft-lora` (156 Mo), `models/dpo-lora` (289 Mo), fusion `models/qwen3-triage-merged` (3,3 Go) |
| 3 | Endpoint de démonstration via API, optimisé vLLM, **déployé sur le cloud** | ⚠️ Partiel | API FastAPI + config vLLM + Docker complets (`deploy/`), image publiée sur GHCR par `deploy.yml` — mais le déploiement cloud effectif est laissé en exemple commenté (choix d'hébergeur + secrets à fournir) |
| 4 | Pipeline CI/CD GitHub Actions (tests + déploiement automatisés) | ✅ Validé* | `.github/workflows/ci.yml` (lint + tests) et `deploy.yml` (build + push image). *Nécessite un push sur GitHub pour tourner réellement — voir réserve git ci-dessous |
| 5 | Rapport technique ≤ 20 pages : méthodologie, métriques, analyse, roadmap | ✅ Validé | `reports/rapport_technique.md` (9 sections : méthodo semaines 1-4, métriques, analyse, limites, roadmap + check-list go/no-go) + `pitch_soutenance.md` |

## Attendus détaillés par étape

| Attendu (étapes 1-3) | État | Preuve |
|---|---|---|
| ~5 000 paires SFT | ✅ | 4 996 paires |
| Paires préférentielles DPO (UltraMedical + validées) | ✅ | 2 220 paires (2 000 UltraMedical + 220 synthétiques après dédup) |
| Anonymisation documentée (Presidio, fr_core_news_md, stratégies) | ✅ | `anonymize.py` + `RGPD.md` |
| Schéma des métadonnées (symptômes, antécédents, constantes, source, confiance) | ✅ | `data/schema.py` (Pydantic : `VitalSigns`, `Metadata`, `SFTRecord`, `DPORecord`) |
| Jeux train/val/test + éval clinique séparés | ✅ | `data/processed/` (3996/500/500) + `data/eval/clinical_eval.jsonl` (119 cas) |
| Ne pas mélanger entraînement et évaluation | ⚠️ | Split strict au niveau enregistrement, MAIS quasi-doublons entre vignettes (cf. `AUDIT_2026-07-06.md` §4.1) — à mentionner honnêtement en soutenance |
| Trace de chaque transformation (auditabilité) | ✅ | Logs, compteurs d'entités masquées, seeds fixées, licences |
| SFT LoRA (empreinte GPU limitée) | ✅ | `train/sft.py` + QLoRA 4 bits, calibré 12 Go VRAM |
| DPO sur paires préférentielles | ✅ | `train/dpo.py` + `configs/dpo.yaml` (beta 0.1, lr 5e-6) |
| Hyperparamètres + seed documentés, checkpoints | ✅ | YAML commentés, `save_steps`, TensorBoard |
| Contrôles de sécurité (hallucinations, conseils dangereux) | ✅ | `eval/safety.py` (sous-triage, sur-triage, conseils dangereux, sensibilité vitale) |
| Questionnaire intelligent adaptatif | ✅ | `serve/questionnaire.py` + endpoint `/chat` |
| Explications claires de l'évaluation | ✅ | Réponse structurée (niveau, délai, analyse, recommandation, disclaimer) |
| Intégration SIH via API REST | ✅ | FastAPI (`/triage`, `/chat`, `/health`, `/audit`) |
| Traçabilité des interactions | ✅ | `serve/audit.py` (JSONL append-only, `logs/audit.log`) |
| Tests de latence en conditions quasi réelles | ✅ | p50 = 2 589 ms, p95 mesuré (`eval_results.json`) |
| Docker + vLLM | ✅ | `deploy/Dockerfile.api`, `docker-compose.yml` |
| Protection des secrets | ✅ | `.env.example` (pas de secret commité), `GITHUB_TOKEN` auto |

## Réserves à lever avant la soutenance

1. **Le dépôt git n'a AUCUN commit** (tout est seulement indexé). Or la mission
   exige un dataset « versionné » et un pipeline CI/CD fonctionnel : commiter,
   pousser sur GitHub, et vérifier que `ci.yml` passe au vert.
2. **Déploiement cloud** : l'attendu littéral est « endpoint déployé sur le
   cloud ». Soit déployer réellement (l'image GHCR est prête ; un VPS GPU ou
   un service type RunPod/Scaleway suffit pour la démo), soit assumer en
   soutenance le choix « déployable en une commande, non déployé » (coût GPU).
3. **Métriques à régénérer** : `eval_results.json` date d'avant les corrections
   de l'audit ; relancer le pipeline (données + SFT + DPO + merge + éval) pour
   des chiffres cohérents avec le code corrigé.

> **Mise à jour du 10 juillet 2026** — second audit : voir
> `reports/AUDIT_2026-07-10.md`. Nouveaux correctifs (contamination DPO→éval,
> red flags déclenchés par les questions de l'agent, Dockerfile…), page de
> démonstration web ajoutée (`GET /ui`) et guide de déploiement cloud rédigé
> (`deploy/GUIDE_DEPLOIEMENT_CLOUD.md`). Les réserves 1 et 2 restent à lever ;
> la 3 est renforcée : régénérer données + modèles est indispensable car les
> splits ont changé.

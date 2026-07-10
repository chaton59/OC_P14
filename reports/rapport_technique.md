# Rapport technique — POC Agent IA de Triage Médical (CHSA)

**Auteur :** IA Engineer (mission CHSA) · **Version :** 1.1 (10 juillet 2026) · **Modèle de base :** Qwen3-1.7B-Base
**Pour :** Dr. Marie Dubois, Directrice Innovation Médicale

> ⚠️ **Avertissement.** Ce document décrit un *Proof of Concept* à visée
> démonstrative. Le système n'est **pas un dispositif médical** et ne doit pas
> être utilisé en conditions cliniques réelles.

---

## 1. Synthèse exécutive

Ce POC démontre la **faisabilité technique** d'un agent d'aide au triage aux
urgences, construit par spécialisation d'un modèle de langage compact
(Qwen3-1.7B) sur un corpus médical bilingue. La chaîne complète a été réalisée
et **exécutée de bout en bout** :

1. constitution d'un **dataset bilingue (~5 000 paires SFT)** anonymisé (RGPD) ;
2. **fine-tuning supervisé** (SFT) avec **LoRA/QLoRA** ;
3. **alignement par préférences** (DPO) orienté **sécurité** ;
4. **API de démonstration** (FastAPI) servie via **vLLM**, conteneurisée, avec
   **CI/CD GitHub Actions** et **journal d'audit**.

Le système combine le modèle avec un **filet de sécurité à base de règles** qui
garantit qu'un signe vital évident n'est jamais sous-évalué — un choix de
conception dicté par la primauté de la sécurité patient.

---

## 2. Contexte et objectifs

Le service des urgences du CHSA subit une surcharge chronique. L'objectif du POC
est d'**assister** (et non remplacer) le personnel de triage en :

* collectant les symptômes via un **questionnaire adaptatif** ;
* évaluant la **priorité** sur trois niveaux ;
* **expliquant** son raisonnement ;
* s'**intégrant** au SIH (API REST) ;
* garantissant la **traçabilité** (audit).

### Niveaux de triage retenus

| Niveau | Code | Délai cible | Couleur |
|--------|:----:|-------------|:-------:|
| `URGENCE_VITALE` | 1 | Immédiat (< 1 min) | 🔴 |
| `URGENCE_MODEREE` | 2 | Rapide (< 20 min) | 🟠 |
| `CONSULTATION_DIFFEREE` | 3 | Différé (< 2 h) | 🟢 |

Échelle inspirée des standards de triage hospitalier (ESI / FRENCH), simplifiée
à trois niveaux pour le POC.

---

## 3. Méthodologie — Préparation des données (Semaine 1)

### 3.1 Sources et agrégation

Quatre corpus publics + un jeu synthétique (cf. `reports/SOURCES.md`) :

| Source | Langue | Rôle | Volume |
|--------|:------:|------|-------:|
| MedQuAD | EN | SFT (Q/R) | 2 500 |
| MedQA-USMLE | EN | SFT (QCM, proxy MediQA) | 2 500 |
| FrenchMedMCQA | FR | SFT (QCM) | 2 171 |
| UltraMedical-Preference | EN | DPO | 2 000 |
| Vignettes de triage (synthétiques) | FR+EN | SFT + DPO | ~1 200 |

**Choix de conception clé :** les corpus publics apportent la *connaissance
médicale* mais pas le *format de triage* attendu. On a donc généré, à partir
d'une table de **18 tableaux cliniques** validés, des **vignettes synthétiques**
bilingues « symptômes → niveau + justification structurée ». Elles enseignent au
modèle le comportement cible et, étant synthétiques, ne posent aucun risque RGPD.

### 3.2 Composition finale du dataset SFT (~5 000 paires)

* **Total :** 4 998 paires construites, 4 994 réparties en partitions (les
  doublons inter-partitions sont écartés) · **Langues :** 62 % EN / 38 % FR.
* **Tâches :** QCM 41 %, Q/R 20 %, triage 24 % (le reste réparti).
* **Niveaux de triage** (sur les vignettes) : équilibrés (~33 % chacun), pour
  éviter tout biais du modèle vers un niveau.
* **Partitions :** train 3 840 / val 572 / test 582 + **200 cas d'évaluation
  clinique** strictement séparés. Le découpage des vignettes se fait **par
  présentation clinique** (une présentation vit dans une seule partition) et le
  jeu DPO **écarte** toute paire recoupant le test ou l'évaluation — correctif
  issu de l'audit du 10 juillet (cf. §6, note de transparence).

### 3.3 Anonymisation et conformité RGPD

Pipeline **Microsoft Presidio** (détaillé dans `reports/RGPD.md`) :
détection bilingue (spaCy `fr`/`en`) + reconnaisseurs regex (téléphone FR, NIR),
stratégie de masquage `replace`. **6 538 entités `PERSON`** masquées sur les
corpus externes. Deux arbitrages assumés : (1) on **ne** masque **pas**
`LOCATION`/`DATE_TIME` (notions cliniques) ; (2) on **n'anonymise pas** les
vignettes 100 % synthétiques — les y soumettre introduisait des faux positifs
(« Délai » détecté comme nom propre) qui **corrompaient le format cible**.
Traçabilité : chaque enregistrement porte `anonymized=true` et le compte de PII.

### 3.4 Schéma de métadonnées

Validé par Pydantic (`src/triage/data/schema.py`) : `symptoms`, `antecedents`,
`vitals` (T°, FC, TA, FR, SpO₂, douleur), `triage_level`, `source`, `license`,
`confidence`, `anonymized`. Toute donnée non conforme est rejetée à la
construction — gage de qualité et d'auditabilité.

---

## 4. Méthodologie — Entraînement (Semaines 2 & 3)

### 4.1 SFT avec LoRA / QLoRA

* **LoRA** (Low-Rank Adaptation) : on n'entraîne que de petites matrices de rang
  faible (~1 % des paramètres), injectées dans les couches d'attention et MLP.
* **QLoRA** : modèle de base chargé en **4 bits (NF4)** → empreinte VRAM divisée
  par ~4, entraînement possible sur une **RTX 4070 (12 Go)**.
* Hyperparamètres (cf. `configs/sft.yaml`) : `r=16`, `alpha=32`, LR `2e-4`,
  batch effectif 16, séquences 1 024 tokens, **2 époques** sur l'ensemble du
  train (3 840 exemples).

**Résultats SFT (validation, run final du 10 juillet) :**

| Métrique | Valeur |
|----------|-------:|
| `eval_loss` | **0.64** |
| Exactitude au token (`mean_token_accuracy`) | **88.0 %** |
| Durée d'entraînement (3 840 ex. × 2 époques, RTX 4070) | ~31 min |

### 4.2 Alignement DPO

* **DPO** (Direct Preference Optimization) : aligne le modèle sur des paires
  *(chosen, rejected)* sans modèle de récompense ni RL — plus stable et léger.
* Données : préférences UltraMedical + **paires de triage synthétiques** dont la
  réponse *rejetée* attribue un **mauvais niveau** (décalé d'un cran) avec un
  conseil inadapté. On apprend ainsi au modèle à **préférer le bon niveau**.
* Démarrage à partir du modèle SFT (adaptateur **fusionné**), nouvel adaptateur
  LoRA, `beta=0.1`, LR `5e-6` (cf. `configs/dpo.yaml`). Après filtrage
  anti-fuite (exclusion de toute paire recoupant le test/l'éval) : **1 334
  paires** d'entraînement / 147 de validation.

**Résultats DPO (validation, run final du 10 juillet) :** `eval_loss` **0.39** ·
exactitude des préférences (`rewards/accuracies`) **0.83** · marge de récompense
(`rewards/margins`) **+2.81** → le modèle distingue nettement la bonne réponse
de la mauvaise.

> **Deux notes d'ingénierie, tracées à dessein.**
> 1. **OOM.** Un premier run DPO a saturé les 12 Go : la taille de batch
>    d'évaluation par défaut (8) provoquait un pic mémoire. Corrigé via
>    `per_device_eval_batch_size=1`, `max_length=768` et `expandable_segments`.
> 2. **Biais d'escalade.** Une première version construisait *toujours* la
>    réponse rejetée avec le niveau `CONSULTATION_DIFFEREE` : le DPO a alors
>    appris à **fuir ce niveau** et sur-triait un simple rhume en urgence vitale.
>    Corrigé en **équilibrant** les mauvais niveaux (décalage d'un cran, réparti
>    sur les trois niveaux). Illustration concrète de l'importance de la
>    construction des préférences en DPO.

---

## 5. Architecture du système (Semaine 4)

```
Client (SIH / web)
      │  POST /triage  (JSON)
      ▼
┌─────────────────────────────────────────────┐
│  API FastAPI  (src/triage/serve)            │
│  1. Questionnaire adaptatif (relances)      │
│  2. Filet de sécurité (règles red flags)    │
│  3. Modèle de langage  ───────────────┐     │
│  4. Décision prudente + explication    │     │
│  5. Journal d'audit (traçabilité)      │     │
└────────────────────────────────────────┼─────┘
                                         ▼
                          Backend d'inférence
                   transformers (local)  |  vLLM (prod)
```

* **Questionnaire adaptatif** : si le message est trop pauvre (et sans signe
  d'alerte), l'agent pose 1–2 questions ciblées (durée, intensité, signes
  associés) avant de trancher.
* **Filet de sécurité** : détection par mots-clés bilingues de signes vitaux
  (douleur thoracique, AVC, détresse respiratoire, hémorragie, anaphylaxie,
  trouble de conscience). En cas de détection, la priorité est **forcée** à
  `URGENCE_VITALE`, indépendamment du modèle. **Principe : ne jamais sous-évaluer.**
* **Décision prudente** : en cas de doute (modèle non concluant), on classe par
  défaut en `URGENCE_MODEREE` (jamais en différé) avec signal de faible confiance.
* **Backends interchangeables** : `transformers` pour les tests locaux/CI,
  **vLLM** (API compatible OpenAI) pour la production.

### 5.1 Déploiement & CI/CD

* **Docker Compose** : service `vLLM` (GPU) + passerelle `API` légère.
* **CI** (`.github/workflows/ci.yml`) : lint Ruff + tests Pytest (sans GPU).
* **CD** (`.github/workflows/deploy.yml`) : build et publication de l'image API
  sur GHCR ; déploiement cloud documenté (VM GPU / HF Endpoints / K8s).

### 5.2 Traçabilité (audit)

Chaque interaction est consignée en JSONL append-only : horodatage, message,
niveau décidé, signes d'alerte, override de sécurité, confiance, latence,
version du modèle. Endpoint `GET /audit` pour relecture.

---

## 6. Résultats d'évaluation clinique

_(Section alimentée par `reports/eval_results.json`, généré par `triage-evaluate`
sur les 200 cas cliniques tenus à l'écart de l'entraînement.)_

> **Note de transparence — métriques régénérées.** Les métriques ci-dessous ont
> été **régénérées le 10 juillet** après la correction d'une **contamination
> train/évaluation** découverte lors de l'audit interne : une partie des prompts
> et des réponses de référence du jeu d'évaluation figurait dans le train DPO,
> et des vignettes quasi identiques traversaient les partitions SFT. Les splits
> se font désormais **par présentation clinique** et le jeu DPO **exclut** toute
> paire recoupant le test ou l'évaluation ; données et modèle ont été
> régénérés puis ré-évalués. Les chiffres sont **plus bas** que ceux de la
> première version du rapport (ex. exactitude 69 % contre 81 %), mais ils
> mesurent la **vraie généralisation** du système, et non sa mémorisation.

<!-- RESULTATS_AUTO_DEBUT -->

**Cas évalués :** 200 · **Exactitude globale :** 69.0 %

| Niveau | Précision | Rappel | F1 |
|--------|:---------:|:------:|:--:|
| `URGENCE_VITALE` | 0.69 | 1.00 | 0.82 |
| `URGENCE_MODEREE` | 0.55 | 0.54 | 0.54 |
| `CONSULTATION_DIFFEREE` | 0.94 | 0.52 | 0.67 |

**Sécurité clinique :**

- Sous-triage dangereux : **2** (1.0 %)
- Conseils dangereux : **0** (0.0 %)
- Sensibilité sur urgences vitales : **100.0 %** (69 cas)

**Latence (ms)** — p50 : 2334 · p95 : 2956 · moyenne : 2371

*(Backend : `configuré`)*

<!-- RESULTATS_AUTO_FIN -->

### 6.1 Lecture des résultats — un compromis « sécurité d'abord »

Le chiffre à retenir n'est **pas** l'exactitude globale (69 %) mais le **profil
de sécurité** :

* **Sensibilité de 100 % sur les urgences vitales** (69/69 cas détectés) et
  **0 sous-triage critique** : aucun cas vital n'est classé en dessous de sa
  gravité. Les 2 seuls sous-triages (1 %) sont **non critiques** (urgence
  modérée classée différée). C'est l'objectif numéro un d'un outil de triage.
* La contrepartie est un **sur-triage de 30 %** (60 cas classés un cran trop
  haut) : la précision sur `URGENCE_VITALE` est de 0.69 et le rappel sur
  `CONSULTATION_DIFFEREE` de 0.52 — le modèle classe « trop haut » des cas
  bénins. Quand il *ose* dire « différé », il a presque toujours raison
  (précision 0.94), mais il ne l'ose qu'une fois sur deux.
* Ce biais est **voulu et cohérent** : il résulte de l'alignement DPO orienté
  prudence et de la règle de décision « en cas de doute, ne jamais classer en
  différé ». En triage, sur-évaluer un rhume (inefficace) est bien moins grave
  que sous-évaluer un infarctus (létal).
* **Le sur-triage vient du modèle, pas du filet de sécurité** : sur l'évaluation,
  les règles de red flags couvrent les 69 cas vitaux **sans aucun faux positif**
  — les 60 sur-classements sont des verdicts du modèle seul.

**En clair :** le POC est **sûr mais sur-prudent**. La marge de progrès porte
sur la *spécificité* (réduire le sur-triage), via un entraînement à plus grande
échelle et une calibration avec les cliniciens — **sans** dégrader la sécurité.

> **Latence.** Mesurée ici avec le backend `transformers` (p50 ≈ 2.3 s). En
> production, **vLLM** réduit fortement ce temps (batching continu, PagedAttention).

**Métriques suivies :** exactitude ; précision/rappel/F1 par niveau ; taux de
sous-triage dangereux ; taux de conseils dangereux ; sensibilité sur les
urgences vitales ; latence (p50/p95/moyenne).

---

## 7. Limites du POC

* **Modèle compact + couverture clinique étroite** : 1.7B paramètres et
  **18 présentations cliniques** dans les vignettes de triage → preuve de
  faisabilité, **pas** un modèle clinique. Hors de cette distribution (ex.
  message vague type « j'ai soif »), le modèle peut halluciner des symptômes.
* **Sur-triage / spécificité faible** : le modèle escalade trop de cas bénins
  (30 % de sur-triage, rappel `CONSULTATION_DIFFEREE` = 0.52). Sûr, mais peu
  efficient en l'état.
* **Données partiellement synthétiques** : le format de triage est appris sur
  des vignettes générées par règles, non sur de vrais dossiers validés.
* **Sur-masquage RGPD** possible des éponymes médicaux (« Parkinson ») dans les
  corpus externes.
* **Pas de constantes vitales structurées** en entrée de l'API (texte libre).
* **Sécurité applicative** non finalisée (auth, TLS, rate-limit) — cf. checklist.
* **Validation clinique** par des soignants **non réalisée** (hors périmètre POC).

---

## 8. Roadmap — Passage à l'échelle

| Phase | Action | Bénéfice attendu |
|------:|--------|------------------|
| **Court terme** | Entraînement pleine échelle (tout le corpus, 3 epochs) ; ajout de vraies constantes vitales structurées ; liste blanche d'éponymes (RGPD). | Qualité & robustesse accrues. |
| **Moyen terme** | Données cliniques réelles **validées par des soignants** (sous AIPD/DPIA) ; modèle plus grand (**7B–32B**) ; évaluation prospective encadrée. | Performance clinique, acceptabilité. |
| **Industrialisation** | Hébergement **HDS** certifié ; sécurité (auth, TLS, RBAC, chiffrement) ; supervision (latence, dérive, alertes) ; intégration SIH réelle. | Mise en production conforme. |
| **Gouvernance** | Comité éthique & médico-légal ; marquage **dispositif médical** (MDR) si usage décisionnel ; plan de gestion des risques. | Cadre réglementaire. |

### Check-list « Go / No-Go » avant pilote

- [ ] Sensibilité sur urgences vitales ≥ seuil défini avec les cliniciens.
- [ ] Taux de sous-triage dangereux ≈ 0 sur jeu de validation clinique élargi.
- [ ] Latence p95 < 2 s en charge.
- [ ] Validation par un panel de soignants.
- [ ] Conformité RGPD/HDS et sécurité applicative vérifiées.

---

## 9. Conclusion

Le POC démontre qu'un modèle compact, spécialisé par **SFT + LoRA** puis aligné
par **DPO**, peut produire des évaluations de triage **structurées, bilingues et
explicables**, servies par une **API industrialisable** (vLLM, Docker, CI/CD) et
**traçable**. L'association *modèle + garde-fous à base de règles* fournit un
socle **prudent** par construction. La valeur clinique réelle reste à confirmer
par une validation soignante et un entraînement à plus grande échelle, conformément
à la roadmap.

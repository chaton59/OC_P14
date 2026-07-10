# 🔒 Conformité RGPD — Anonymisation des données

> Document justificatif du traitement des données personnelles, conformément à
> l'exigence de la mission (« anonymiser toutes les données et documenter le
> processus RGPD »).

## 1. Principe : minimisation et anonymisation à la source

Le projet applique le principe de **minimisation des données** (art. 5 RGPD) :

* Les **vignettes de triage** — le cœur de l'apprentissage du comportement
  cible — sont **100 % synthétiques** (générées par règles). Elles ne
  contiennent **aucune donnée patient réelle**.
* Les **corpus externes** (MedQuAD, MedQA, FrenchMedMCQA, UltraMedical) sont des
  jeux de données médicaux **publics** de type questions/connaissances ; ils ne
  visent pas des individus. Par précaution, ils sont néanmoins passés au filtre
  d'anonymisation décrit ci-dessous.

## 2. Outil : Microsoft Presidio

L'anonymisation s'appuie sur **Presidio** (open source), via deux moteurs :

| Moteur | Rôle |
|--------|------|
| `AnalyzerEngine`   | détecte les entités personnelles (PII) dans le texte |
| `AnonymizerEngine` | remplace les PII détectées par des marqueurs neutres |

Configuration (cf. `src/triage/data/anonymize.py`) :

* **Modèles linguistiques bilingues** : `fr_core_news_md` (français) et
  `en_core_web_md` (anglais) — pour reconnaître les noms de personnes dans les
  deux langues.
* **Reconnaisseurs regex personnalisés** indépendants de la langue :
  * numéro de sécurité sociale français (**NIR**) ;
  * numéro de téléphone français (`06 12 34 56 78`, `+33 …`).
* **Stratégie de masquage** : `replace` (remplacement par un marqueur lisible,
  ex. `<PERSONNE>`, `<TELEPHONE>`, `<NIR>`). On préfère `replace` à `redact`
  car il conserve la structure de la phrase, utile pour l'entraînement.

## 3. Entités ciblées et arbitrage qualité / confidentialité

| Entité | Masquée ? | Justification |
|--------|:---------:|---------------|
| `PERSON` (nom, prénom) | ✅ | **Priorité explicite de la mission.** |
| `EMAIL_ADDRESS`, `PHONE_NUMBER`, `FR_SSN` | ✅ | Identifiants directs à fort risque. |
| `IP_ADDRESS`, `CREDIT_CARD` | ✅ | Identifiants techniques/financiers. |
| `LOCATION`, `DATE_TIME` | ❌ (volontaire) | Dans un corpus médical, ils désignent presque toujours des notions **cliniques** (anatomie, ancienneté des symptômes…). Les masquer provoquerait un **sur-masquage massif** dégradant la qualité d'entraînement, pour un gain de confidentialité quasi nul. |

Un **seuil de confiance** (`score_threshold = 0.6`) filtre les détections les
moins sûres, réduisant les faux positifs.

> **Limite connue & honnête.** La détection de `PERSON` peut masquer à tort
> certains **éponymes médicaux** (ex. « maladie de Parkinson » → « maladie de
> `<PERSONNE>` »). C'est le prix d'une anonymisation prudente côté noms. En
> production, on affinerait avec une *liste blanche* d'éponymes médicaux.

## 4. Traçabilité (auditabilité)

Pour chaque enregistrement, le pipeline conserve dans les métadonnées :

* `anonymized: true` — preuve que l'enregistrement a été traité ;
* `pii_masked: <n>` — nombre d'entités masquées.

Un **rapport agrégé** (nombre d'entités masquées par type) est journalisé à
chaque exécution. Aucune donnée personnelle d'origine n'est jamais conservée :
seules sont stockées les versions anonymisées (`data/interim/*_anon.jsonl`).

## 5. Vérification qualité

* Inspection manuelle d'un échantillon des sorties anonymisées.
* Tests unitaires (`tests/test_anonymize.py`) vérifiant que des PII synthétiques
  (e-mail, téléphone, NIR, nom) sont bien masquées.

## 6. Portée et responsabilités

Ce POC ne traite **aucune donnée patient réelle**. Avant tout passage en
production avec de vraies données, il faudra : une analyse d'impact (AIPD/DPIA),
la validation par le DPO de l'établissement, et un hébergement **HDS**
(Hébergeur de Données de Santé) certifié.

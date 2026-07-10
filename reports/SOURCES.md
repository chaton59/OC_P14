# 📚 Sources de données — origine, licence et usage

Ce document assure la **traçabilité** des corpus utilisés (exigence
d'auditabilité de la mission). Chaque source est citée avec son origine, sa
licence et son rôle dans le projet.

| Corpus | Langue | Type | Rôle | Licence | Volume collecté |
|--------|:------:|------|------|---------|----------------:|
| [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | EN | Q/R médicales (NIH) | SFT | Usage recherche (NIH) | 2 500 |
| [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | EN | QCM cliniques | SFT (proxy « MediQA ») | MIT | 2 500 |
| [FrenchMedMCQA](https://huggingface.co/datasets/qanastek/FrenchMedMCQA) | FR | QCM pharmacie/médecine | SFT | Apache-2.0 | 2 171 |
| [UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | EN | Paires préférentielles | DPO | MIT | 2 000 |
| Vignettes de triage (synthétiques) | FR + EN | Symptômes → priorité | SFT + DPO | CC0 (généré par règles) | ~1 200 SFT / 720 DPO |

## Notes d'ingénierie

* **FrenchMedMCQA** était historiquement distribué via un *script de chargement*
  Python, désormais refusé par `datasets >= 3`. On le charge donc depuis la
  branche **Parquet auto-convertie** du Hub (`@~parquet`), ce qui reste robuste.
  Le champ `correct_answers` contient des **indices entiers** (0→A … 4→E) que
  l'on reconvertit en lettres.
* **MediQA** : le nom « MediQA » recouvre plusieurs jeux hétérogènes (RQE, NLI,
  QA…), souvent sous accès restreint. On utilise **MedQA-USMLE** comme substitut
  ouvert et représentatif de QCM cliniques en anglais.
* **Vignettes synthétiques** : générées par règles à partir d'une table de
  tableaux cliniques (cf. `src/triage/data/triage_knowledge.py`). Elles
  apprennent au modèle le **format de sortie** attendu (niveau de priorité +
  justification structurée), absent des corpus publics. Étant 100 % synthétiques,
  elles ne présentent **aucun risque RGPD** à la source.

## Bilinguisme

Le corpus final est bilingue : le français est porté par FrenchMedMCQA et la
moitié des vignettes de triage ; l'anglais par MedQuAD, MedQA et l'autre moitié
des vignettes. La tâche cible elle-même (le triage) est ainsi enseignée dans les
deux langues.

## Reproductibilité

Toute la chaîne est rejouable avec une graine fixe (`--seed 42`) :

```bash
uv run triage-collect
uv run triage-build-sft
uv run triage-build-dpo
uv run triage-anonymize
uv run triage-split
```

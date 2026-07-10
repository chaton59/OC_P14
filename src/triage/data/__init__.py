"""Sous-package `data` — préparation des corpus médicaux bilingues.

Chaîne de traitement (cf. Semaine 1 de la mission) :

    collect   →  télécharge / charge les corpus sources bruts
    build_sft →  unifie tout au format "instruction-réponse" (≈5 000 paires)
    build_dpo →  construit les paires préférentielles (chosen / rejected)
    anonymize →  masque les informations personnelles (RGPD, via Presidio)
    split     →  partitionne en train / val / test + eval clinique séparé
"""

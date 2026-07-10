"""Package `triage` — POC d'agent IA de triage médical pour le CHSA.

Ce package regroupe l'ensemble de la chaîne de valeur du Proof of Concept :

    triage.data   → collecte, construction et anonymisation des datasets
    triage.train  → fine-tuning supervisé (SFT/LoRA) et alignement (DPO)
    triage.eval   → évaluation des performances et contrôles de sécurité
    triage.serve  → agent de triage + API FastAPI (backends transformers/vLLM)
    triage.utils  → utilitaires transverses (logs, graines aléatoires)

La logique métier (les "protocoles de triage") est volontairement séparée de
l'infrastructure (entraînement, API) pour rester lisible et testable.
"""

__version__ = "0.1.0"

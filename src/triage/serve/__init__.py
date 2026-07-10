"""Sous-package `serve` — agent de triage et API de démonstration.

  inference.py    →  abstraction du backend d'inférence (transformers ou vLLM)
  questionnaire.py→  logique du questionnaire adaptatif et filet de sécurité
  triage_agent.py →  orchestration : du message patient au verdict de triage
  audit.py        →  journal d'audit (traçabilité des interactions)
  api.py          →  API REST FastAPI exposant l'agent
"""

"""Démonstration en ligne de commande de l'agent de triage.

Fait passer quelques situations cliniques types à l'agent et affiche le verdict.
Pratique pour une démo rapide ou une vérification visuelle après entraînement.

Usage :
    TRIAGE_MODEL_PATH=models/qwen3-triage-merged uv run python scripts/demo_triage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triage.serve.triage_agent import TriageAgent  # noqa: E402

# Cas de démonstration couvrant les trois niveaux de priorité + bilingue.
CAS_DEMO = [
    "Douleur thoracique constrictive et essoufflement depuis une heure.",
    "I have a runny nose and a few sneezes since this morning.",
    "Douleur vive à la cheville après une chute, je n'arrive plus à marcher.",
    "Mal de gorge léger et petite fièvre depuis hier.",
    "Sudden facial drooping on one side and slurred speech.",
]


def main() -> None:
    agent = TriageAgent()
    for message in CAS_DEMO:
        # `allow_followup=False` : pour la démo, on force une décision immédiate.
        result = agent.assess(message, allow_followup=False)
        print("=" * 70)
        print(f"PATIENT  : {message}")
        print(f"NIVEAU   : {result['triage_level']} ({result.get('label')})")
        print(f"DÉLAI    : {result.get('delai')}")
        print(f"ALERTES  : {result.get('red_flags')}")
        print(f"CONF.    : {result.get('confidence')}  |  LATENCE : {result['latency_ms']} ms")
        print(f"\nEXPLICATION :\n{result.get('explanation', '')[:500]}")
        print()


if __name__ == "__main__":
    main()

"""Injecte les résultats d'évaluation dans le rapport technique.

Lit `reports/eval_results.json` et remplace le bloc délimité par les marqueurs
`<!-- RESULTATS_AUTO_DEBUT -->` / `<!-- RESULTATS_AUTO_FIN -->` du rapport par un
tableau de métriques formaté. Ainsi le rapport reste synchronisé avec la
dernière évaluation, sans copier-coller manuel.

Usage : uv run python scripts/inject_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "rapport_technique.md"
RESULTS = ROOT / "reports" / "eval_results.json"
BEGIN = "<!-- RESULTATS_AUTO_DEBUT -->"
END = "<!-- RESULTATS_AUTO_FIN -->"


def format_results(data: dict) -> str:
    """Construit le bloc Markdown des résultats à partir du JSON d'évaluation."""
    lines = [BEGIN, ""]
    lines.append(f"**Cas évalués :** {data['n_cases']} · "
                 f"**Exactitude globale :** {data['accuracy'] * 100:.1f} %\n")

    lines.append("| Niveau | Précision | Rappel | F1 |")
    lines.append("|--------|:---------:|:------:|:--:|")
    for level, m in data["per_level"].items():
        lines.append(f"| `{level}` | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
    lines.append("")

    s = data["safety"]
    lines.append("**Sécurité clinique :**\n")
    lines.append(f"- Sous-triage dangereux : **{s['dangerous_undertriage']}** "
                 f"({s['dangerous_undertriage_rate'] * 100:.1f} %)")
    lines.append(f"- Conseils dangereux : **{s['dangerous_advice']}** "
                 f"({s['dangerous_advice_rate'] * 100:.1f} %)")
    lines.append(f"- Sensibilité sur urgences vitales : "
                 f"**{s['vital_sensitivity'] * 100:.1f} %** ({s['vital_total']} cas)\n")

    lat = data["latency_ms"]
    lines.append(f"**Latence (ms)** — p50 : {lat['p50']:.0f} · "
                 f"p95 : {lat['p95']:.0f} · moyenne : {lat['mean']:.0f}\n")
    lines.append(f"*(Backend : `{data.get('model_path', 'n/a')}`)*")
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def main() -> None:
    if not RESULTS.exists():
        print(f"Fichier de résultats introuvable : {RESULTS}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")

    start, end = report.find(BEGIN), report.find(END)
    if start == -1 or end == -1:
        print("Marqueurs RESULTATS_AUTO introuvables dans le rapport.", file=sys.stderr)
        sys.exit(1)

    new_report = report[:start] + format_results(data) + report[end + len(END):]
    REPORT.write_text(new_report, encoding="utf-8")
    print(f"Résultats injectés dans {REPORT}")


if __name__ == "__main__":
    main()

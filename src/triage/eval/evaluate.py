"""Évaluation clinique du modèle de triage.

On mesure, sur le **jeu d'évaluation clinique** (séparé de l'entraînement) :

  * l'**exactitude** du niveau de triage prédit vs attendu ;
  * la **matrice de confusion** (quels niveaux sont confondus) ;
  * des **métriques de sécurité** (taux de sous-triage dangereux, sensibilité
    sur les urgences vitales — cf. `safety.py`) ;
  * la **latence** par requête (p50 / p95).

Les résultats sont affichés et sauvegardés en JSON (pour le rapport et le suivi).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from triage.config import EVAL_DIR, REPORTS_DIR, TRIAGE_LEVELS
from triage.eval.safety import safety_report
from triage.serve.triage_agent import TriageAgent, extract_triage_level
from triage.utils.common import get_logger, read_jsonl, set_seed

logger = get_logger("eval")


def _build_agent(model_path: str | None) -> TriageAgent:
    """Construit l'agent, éventuellement avec un modèle spécifique (sinon config)."""
    if model_path:
        from triage.serve.inference import TransformersBackend

        return TriageAgent(backend=TransformersBackend(model_path))
    return TriageAgent()


def evaluate(eval_file: str, model_path: str | None, max_cases: int | None) -> dict:
    """Évalue le modèle sur le jeu clinique et renvoie le rapport de métriques."""
    cases = list(read_jsonl(eval_file))
    if max_cases:
        cases = cases[:max_cases]
    if not cases:
        logger.error("Aucun cas d'évaluation trouvé dans %s.", eval_file)
        return {}

    # Reproductibilité : la génération est échantillonnée (temperature > 0) ;
    # sans graine fixée, deux évaluations donneraient des métriques différentes,
    # ce qui contredit l'exigence d'auditabilité.
    set_seed(42)
    agent = _build_agent(model_path)

    correct = 0
    latencies: list[float] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    safety_inputs: list[dict] = []

    logger.info("Évaluation sur %d cas cliniques…", len(cases))
    for i, case in enumerate(cases, 1):
        expected = case.get("expected_triage_level")
        start = time.perf_counter()
        # `allow_followup=False` : en évaluation, on force une décision immédiate.
        result = agent.assess(case["instruction"], allow_followup=False)
        latencies.append((time.perf_counter() - start) * 1000)

        predicted = result.get("triage_level") or extract_triage_level(
            result.get("explanation", "")
        )
        # Une prédiction absente est comptée comme classe explicite « AUCUN »
        # (et non ignorée) : elle doit peser dans la matrice et le rappel.
        confusion[expected][predicted or "AUCUN"] += 1
        if predicted == expected:
            correct += 1
        safety_inputs.append({
            "predicted_level": predicted,
            "expected_level": expected,
            "text": result.get("explanation", ""),
        })
        if i % 20 == 0:
            logger.info("  %d/%d traités…", i, len(cases))

    accuracy = correct / len(cases)
    safety = safety_report(safety_inputs)

    # Précision/rappel par niveau (one-vs-rest).
    per_level = {}
    for level in TRIAGE_LEVELS:
        tp = confusion[level][level]
        fp = sum(confusion[other][level] for other in confusion if other != level)
        # Les faux négatifs incluent TOUTES les autres colonnes (y compris
        # « AUCUN ») : sinon le rappel serait artificiellement gonflé.
        fn = sum(cnt for pred, cnt in confusion[level].items() if pred != level)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_level[level] = {"precision": round(prec, 3), "recall": round(rec, 3),
                            "f1": round(f1, 3)}

    report = {
        "n_cases": len(cases),
        "accuracy": round(accuracy, 4),
        "per_level": per_level,
        "safety": safety,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            # `statistics.quantiles` interpole correctement le 95e centile
            # (l'ancien indexage `int(0.95*n)-1` donnait le p80 pour n=10).
            "p95": round(
                statistics.quantiles(latencies, n=100)[94]
                if len(latencies) > 1 else latencies[0],
                1,
            ),
            "mean": round(statistics.mean(latencies), 1),
        },
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "model_path": model_path or "configuré",
    }
    return report


def _print_report(report: dict) -> None:
    """Affiche un résumé lisible du rapport d'évaluation."""
    logger.info("=" * 60)
    logger.info("RÉSULTATS D'ÉVALUATION CLINIQUE")
    logger.info("  Cas évalués          : %d", report["n_cases"])
    logger.info("  Exactitude (accuracy): %.1f %%", report["accuracy"] * 100)
    logger.info("  --- Par niveau (P / R / F1) ---")
    for level, m in report["per_level"].items():
        logger.info("    %-22s : %.2f / %.2f / %.2f", level, m["precision"], m["recall"], m["f1"])
    s = report["safety"]
    logger.info("  --- Sécurité ---")
    logger.info("    Sous-triage dangereux : %d (%.1f %%)",
                s["dangerous_undertriage"], s["dangerous_undertriage_rate"] * 100)
    logger.info("      dont CRITIQUES (vital manqué) : %d (%.1f %%)",
                s.get("critical_undertriage", 0),
                s.get("critical_undertriage_rate", 0.0) * 100)
    logger.info("    Sur-triage            : %d (%.1f %%)",
                s.get("overtriage", 0), s.get("overtriage_rate", 0.0) * 100)
    logger.info("    Conseils dangereux    : %d (%.1f %%)",
                s["dangerous_advice"], s["dangerous_advice_rate"] * 100)
    logger.info("    Sensibilité urgences vitales : %.1f %% (%d cas)",
                s["vital_sensitivity"] * 100, s["vital_total"])
    lat = report["latency_ms"]
    logger.info("  --- Latence (ms) --- p50=%.0f | p95=%.0f | moy=%.0f",
                lat["p50"], lat["p95"], lat["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation clinique du modèle de triage.")
    parser.add_argument("--eval-file", default=str(EVAL_DIR / "clinical_eval.jsonl"))
    parser.add_argument("--model-path", default=None,
                        help="Chemin d'un modèle fusionné spécifique (sinon config).")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output", default=str(REPORTS_DIR / "eval_results.json"))
    args = parser.parse_args()

    report = evaluate(args.eval_file, args.model_path, args.max_cases)
    if not report:
        return
    _print_report(report)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Rapport sauvegardé : %s", args.output)


if __name__ == "__main__":
    main()

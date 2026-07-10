"""Statistiques descriptives des datasets (pour le rapport et le notebook).

Fournit des fonctions réutilisables qui résument la composition des jeux de
données : répartition par source, langue, type de tâche et niveau de triage.
"""

from __future__ import annotations

import argparse
from collections import Counter

from triage.config import EVAL_DIR, SFT_DIR
from triage.utils.common import get_logger, read_jsonl

logger = get_logger("data.stats")


def sft_stats(jsonl_path) -> dict:
    """Calcule les répartitions clés d'un fichier SFT."""
    by_source: Counter = Counter()
    by_lang: Counter = Counter()
    by_task: Counter = Counter()
    by_level: Counter = Counter()
    n = 0
    for r in read_jsonl(jsonl_path):
        n += 1
        meta = r.get("metadata", {})
        by_source[meta.get("source", "?")] += 1
        by_lang[r.get("lang", "?")] += 1
        by_task[r.get("task_type", "?")] += 1
        if meta.get("triage_level"):
            by_level[meta["triage_level"]] += 1
    return {
        "total": n,
        "par_source": dict(by_source),
        "par_langue": dict(by_lang),
        "par_tache": dict(by_task),
        "par_niveau_triage": dict(by_level),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistiques des datasets.")
    parser.add_argument("--file", default=str(SFT_DIR / "train.jsonl"))
    args = parser.parse_args()

    stats = sft_stats(args.file)
    logger.info("Statistiques de %s", args.file)
    logger.info("  Total           : %d", stats["total"])
    logger.info("  Par langue      : %s", stats["par_langue"])
    logger.info("  Par tâche       : %s", stats["par_tache"])
    logger.info("  Par source      : %s", stats["par_source"])
    logger.info("  Par niveau triage: %s", stats["par_niveau_triage"])

    eval_file = EVAL_DIR / "clinical_eval.jsonl"
    if eval_file.exists():
        n_eval = sum(1 for _ in read_jsonl(eval_file))
        logger.info("  Cas d'évaluation clinique : %d", n_eval)


if __name__ == "__main__":
    main()

"""Étape 1a — Collecte des corpus médicaux sources (bruts).

Ce script télécharge un échantillon de chaque corpus demandé par la mission et
le normalise en un format brut commun, écrit dans `data/raw/`. On ne fait ici
AUCUN nettoyage métier : on se contente d'extraire les champs utiles. Les
étapes suivantes (`build_sft`, `build_dpo`, `anonymize`) s'en chargeront.

Sources (selon disponibilité publique sur le Hugging Face Hub) :
  * MedQuAD            — Q/R médicales (anglais)         → SFT
  * FrenchMedMCQA      — QCM médicaux (français)         → SFT
  * MedQA (USMLE)      — QCM cliniques (anglais)          → SFT  (proxy « MediQA »)
  * UltraMedical-Preference — paires préférentielles      → DPO

ROBUSTESSE : chaque source est chargée en *streaming* (on ne télécharge que les
N premiers exemples, pas le corpus entier) et encadrée par un try/except. Si une
source est indisponible (réseau, accès restreint), on journalise l'erreur et on
poursuit avec les autres — le pipeline ne casse jamais entièrement.
"""

from __future__ import annotations

import argparse
import itertools
import re

from triage.config import RAW_DIR, ensure_dirs
from triage.utils.common import get_logger, write_jsonl

logger = get_logger("collect")


def _take(stream, limit: int):
    """Renvoie au plus `limit` éléments d'un dataset en streaming (itérable)."""
    return list(itertools.islice(stream, limit))


# ---------------------------------------------------------------------------
# Collecteurs : un par source. Chacun renvoie une liste de dicts bruts.
# ---------------------------------------------------------------------------
def collect_medquad(limit: int) -> list[dict]:
    """MedQuAD : questions/réponses médicales grand public, en anglais."""
    from datasets import load_dataset

    ds = load_dataset("lavita/MedQuAD", split="train", streaming=True)
    out = []
    for ex in _take(ds, limit):
        question = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()
        if question and answer:
            out.append(
                {
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "lang": "en",
                    "task_type": "qa",
                    "source": "medquad",
                    "license": "MedQuAD (NIH, usage recherche)",
                }
            )
    return out


def collect_frenchmedmcqa(limit: int) -> list[dict]:
    """FrenchMedMCQA : questions à choix multiples médicales, en français.

    Ce corpus était historiquement distribué via un *script* de chargement, que
    les versions récentes de `datasets` (>= 3) ne supportent plus. On contourne
    en lisant directement la branche Parquet auto-convertie par le Hub
    (`@~parquet`), ce qui reste robuste et sans script.
    """
    import pandas as pd

    url = "hf://datasets/qanastek/FrenchMedMCQA@~parquet/default/train/0000.parquet"
    df = pd.read_parquet(url)
    letters = ["a", "b", "c", "d", "e"]
    out = []
    for ex in df.head(limit).to_dict(orient="records"):
        question = (ex.get("question") or "").strip()
        # Reconstitution des options proposées (answer_a … answer_e).
        options = {ltr: (ex.get(f"answer_{ltr}") or "").strip() for ltr in letters}
        options = {ltr: v for ltr, v in options.items() if v}
        if not question or not options:
            continue
        options_str = "\n".join(f"{ltr.upper()}) {v}" for ltr, v in options.items())

        # `correct_answers` : INDICES entiers des bonnes réponses (0→a, 1→b, …).
        # Le champ peut être un tableau numpy, une liste ou une chaîne de lettres.
        raw_correct = ex.get("correct_answers")
        correct: list[str] = []
        if raw_correct is None:
            pass
        elif isinstance(raw_correct, str):
            # Lettres ISOLÉES uniquement (\b) : sans les bornes, chaque lettre
            # a-e À L'INTÉRIEUR des mots était capturée (« Answer » → a, e…).
            correct = [c.lower() for c in re.findall(r"\b([a-eA-E])\b", raw_correct)]
        else:  # liste / array d'indices entiers
            for c in list(raw_correct):
                try:
                    correct.append(letters[int(c)])  # index entier → lettre
                except (ValueError, IndexError, TypeError):
                    if str(c).lower() in letters:
                        correct.append(str(c).lower())
        good = [options[c] for c in correct if c in options]
        if not good:
            continue
        answer = (
            "Bonne(s) réponse(s) : "
            + ", ".join(f"{c.upper()}) {options[c]}" for c in correct if c in options)
        )

        out.append(
            {
                "instruction": f"{question}\n\n{options_str}",
                "input": "",
                "output": answer,
                "lang": "fr",
                "task_type": "mcqa",
                "source": "frenchmedmcqa",
                "license": "FrenchMedMCQA (Apache-2.0)",
            }
        )
    return out


def collect_medqa(limit: int) -> list[dict]:
    """MedQA (USMLE, 4 options) : QCM cliniques en anglais (proxy « MediQA »)."""
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", streaming=True)
    out = []
    for ex in _take(ds, limit):
        question = (ex.get("question") or "").strip()
        options = ex.get("options") or {}
        answer = (ex.get("answer") or "").strip()
        if not question or not answer:
            continue
        if isinstance(options, dict) and options:
            options_str = "\n".join(f"{k}) {v}" for k, v in options.items())
            instruction = f"{question}\n\n{options_str}"
        else:
            instruction = question
        out.append(
            {
                "instruction": instruction,
                "input": "",
                # Préfixe en ANGLAIS : ces exemples sont en anglais (le préfixe
                # français « Réponse : » créait du bruit de mélange de langues).
                "output": f"Answer: {answer}",
                "lang": "en",
                "task_type": "mcqa",
                "source": "medqa",
                "license": "MedQA (MIT)",
            }
        )
    return out


def collect_ultramedical_preference(limit: int) -> list[dict]:
    """UltraMedical-Preference : paires (chosen / rejected) pour l'alignement DPO."""
    from datasets import load_dataset

    ds = load_dataset("TsinghuaC3I/UltraMedical-Preference", split="train", streaming=True)
    out = []
    for ex in _take(ds, limit):
        # Le schéma exact peut varier : on cherche les clés de façon défensive.
        prompt = ex.get("prompt") or ex.get("instruction") or ex.get("question") or ""
        chosen = ex.get("chosen")
        rejected = ex.get("rejected")

        # Certains corpus stockent chosen/rejected comme des listes de messages.
        chosen = _extract_text(chosen)
        rejected = _extract_text(rejected)
        prompt = _extract_text(prompt)

        if prompt and chosen and rejected:
            out.append(
                {
                    "prompt": prompt.strip(),
                    "chosen": chosen.strip(),
                    "rejected": rejected.strip(),
                    "lang": "en",
                    "task_type": "qa",
                    "source": "ultramedical_preference",
                    "license": "UltraMedical-Preference (MIT)",
                }
            )
    return out


def _extract_text(value) -> str:
    """Extrait du texte d'un champ qui peut être une str, un dict ou une liste de messages."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content") or value.get("text") or ""
    if isinstance(value, list):
        # Liste de messages type [{"role": ..., "content": ...}] → on prend le dernier.
        for msg in reversed(value):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        return " ".join(str(v) for v in value)
    return str(value)


# Registre : nom de fichier de sortie → (fonction, limite par défaut).
SFT_SOURCES = {
    "medquad": collect_medquad,
    "frenchmedmcqa": collect_frenchmedmcqa,
    "medqa": collect_medqa,
}
DPO_SOURCES = {
    "ultramedical_preference": collect_ultramedical_preference,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collecte des corpus médicaux sources.")
    parser.add_argument("--limit-sft", type=int, default=2000,
                        help="Nb max d'exemples par source SFT (streaming).")
    parser.add_argument("--limit-dpo", type=int, default=2000,
                        help="Nb max d'exemples pour la source DPO.")
    args = parser.parse_args()

    ensure_dirs()

    def _collect(name: str, fn, limit: int) -> None:
        """Collecte une source et n'écrit le fichier QUE si elle a produit des
        données : une source vide (dérive de schéma, panne) n'écrase plus un
        fichier brut précédent encore valide."""
        try:
            logger.info("Collecte de '%s' (max %d exemples)…", name, limit)
            records = fn(limit)
            if not records:
                logger.warning("  ✗ Source '%s' : 0 exemple, fichier existant conservé.", name)
                return
            n = write_jsonl(records, RAW_DIR / f"{name}.jsonl")
            logger.info("  → %d exemples écrits dans data/raw/%s.jsonl", n, name)
        except Exception as exc:  # noqa: BLE001 — on veut continuer malgré une source KO
            logger.warning("  ✗ Source '%s' indisponible : %s", name, exc)

    # --- Sources SFT ---
    for name, fn in SFT_SOURCES.items():
        _collect(name, fn, args.limit_sft)

    # --- Source DPO ---
    for name, fn in DPO_SOURCES.items():
        _collect(name, fn, args.limit_dpo)

    logger.info("Collecte terminée. Fichiers bruts dans : %s", RAW_DIR)


if __name__ == "__main__":
    main()

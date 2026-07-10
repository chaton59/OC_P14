"""Étape 1c — Construction du dataset DPO (paires préférentielles).

Le DPO (Direct Preference Optimization) apprend au modèle à PRÉFÉRER une bonne
réponse (`chosen`) à une moins bonne (`rejected`), sans modèle de récompense.

On combine deux sources de préférences :
  1. UltraMedical-Preference — paires validées issues du Hub (anglais) ;
  2. des paires de triage synthétiques où `chosen` = réponse prudente et bien
     structurée, `rejected` = réponse qui SOUS-ÉVALUE la gravité (l'erreur la
     plus dangereuse en triage). On aligne ainsi le modèle vers la sécurité.

Sortie : `data/interim/dpo.jsonl`, validé au schéma `DPORecord`.
"""

from __future__ import annotations

import argparse
import random

from pydantic import ValidationError

from triage.config import INTERIM_DIR, RAW_DIR, ensure_dirs
from triage.data.schema import DPORecord, Metadata
from triage.data.triage_knowledge import (
    _QUESTION_TEMPLATES_EN as Q_EN,
)
from triage.data.triage_knowledge import (
    _QUESTION_TEMPLATES_FR as Q_FR,
)
from triage.data.triage_knowledge import (
    PRESENTATIONS,
    format_triage_answer,
    make_bad_triage_answer,
)
from triage.utils.common import get_logger, read_jsonl, write_jsonl

logger = get_logger("build_dpo")


def build_triage_preferences(n_per_presentation: int, seed: int) -> list[DPORecord]:
    """Génère des paires de préférence (bonne vs mauvaise réponse de triage)."""
    # Graine dérivée volontairement DIFFÉRENTE de celle des vignettes SFT : avec
    # la même graine et les mêmes gabarits, les prompts DPO étaient identiques
    # octet pour octet aux instructions SFT (recouvrement train/éval).
    rng = random.Random(f"dpo-{seed}")
    out: list[DPORecord] = []
    seen: set[tuple[str, str]] = set()
    for p in PRESENTATIONS:
        for i in range(n_per_presentation):
            lang = "fr" if i % 2 == 0 else "en"
            symptoms = p.symptoms_fr if lang == "fr" else p.symptoms_en
            templates = Q_FR if lang == "fr" else Q_EN
            k = rng.randint(2, len(symptoms))
            sampled_symptoms = rng.sample(symptoms, k)
            symptoms_str = ", ".join(sampled_symptoms)
            age = rng.randint(18, 88)
            prompt = rng.choice(templates).format(symptoms=symptoms_str, age=age)

            chosen = format_triage_answer(p, lang)          # prudente & structurée
            rejected = make_bad_triage_answer(p, lang, variant=i)  # mauvais niveau
            # Déduplication : le pool de variantes est petit, on évite que des
            # paires identiques se retrouvent à la fois en train et en val.
            key = (prompt, rejected)
            if key in seen:
                continue
            seen.add(key)
            try:
                out.append(
                    DPORecord(
                        id=f"dpo-triage-{p.key}-{lang}-{i:03d}",
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        lang=lang,
                        task_type="triage",
                        metadata=Metadata(
                            # Les MÊMES symptômes que dans le prompt (un second
                            # tirage aléatoire désynchronisait les métadonnées).
                            symptoms=sampled_symptoms,
                            triage_level=p.level,
                            source="triage_synth_pref",
                            license="CC0 (synthétique)",
                            confidence=0.9,
                            # Clé de présentation : permet au split d'ÉCARTER du
                            # train DPO les présentations réservées au test/à
                            # l'éval clinique (anti-contamination train/éval).
                            presentation_key=p.key,
                        ),
                    )
                )
            except ValidationError as exc:
                logger.debug("Paire de triage rejetée : %s", exc)
    return out


def load_external_preferences(limit: int, max_chars: int = 2500) -> list[DPORecord]:
    """Charge les paires préférentielles externes (UltraMedical-Preference).

    `max_chars` borne la longueur `prompt + réponse la plus longue` : au-delà,
    la paire serait TRONQUÉE par `max_length` à l'entraînement DPO (768 tokens
    ≈ 2 500-3 000 caractères) et le signal de préférence sur la FIN des
    réponses serait perdu — même famille de bug que les réponses SFT tronquées
    sans `<|im_end|>` (cf. `build_sft.py --max-chars`).
    """
    out: list[DPORecord] = []
    skipped_len = 0
    path = RAW_DIR / "ultramedical_preference.jsonl"
    if not path.exists():
        logger.warning("Source DPO externe absente (%s). Lancez `triage-collect`.", path.name)
        return out
    for row in read_jsonl(path):
        if len(out) >= limit:
            break
        longest = max(len(row.get("chosen", "")), len(row.get("rejected", "")))
        if len(row.get("prompt", "")) + longest > max_chars:
            skipped_len += 1
            continue
        idx = len(out)
        try:
            out.append(
                DPORecord(
                    id=f"ultramed-{idx:06d}",
                    prompt=row["prompt"],
                    chosen=row["chosen"],
                    rejected=row["rejected"],
                    lang=row.get("lang", "en"),
                    task_type=row.get("task_type", "qa"),
                    metadata=Metadata(
                        source=row.get("source", "ultramedical_preference"),
                        license=row.get("license", "unknown"),
                        confidence=0.8,
                    ),
                )
            )
        except (ValidationError, KeyError) as exc:
            logger.debug("Paire externe rejetée : %s", exc)
    if skipped_len:
        logger.info(
            "Paires externes écartées car trop longues (> %d caractères) : %d",
            max_chars, skipped_len,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction du dataset DPO.")
    parser.add_argument("--triage-per-presentation", type=int, default=40)
    parser.add_argument("--external-limit", type=int, default=1500)
    parser.add_argument("--max-chars", type=int, default=2500,
                        help="Longueur max (prompt + réponse) des paires externes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    triage_prefs = build_triage_preferences(args.triage_per_presentation, args.seed)
    logger.info("Paires de préférence triage générées : %d", len(triage_prefs))

    external_prefs = load_external_preferences(args.external_limit, args.max_chars)
    logger.info("Paires de préférence externes chargées : %d", len(external_prefs))

    all_prefs = triage_prefs + external_prefs
    random.Random(args.seed).shuffle(all_prefs)

    out_path = INTERIM_DIR / "dpo.jsonl"
    n = write_jsonl((r.model_dump() for r in all_prefs), out_path)
    logger.info("Dataset DPO construit : %d paires → %s", n, out_path)


if __name__ == "__main__":
    main()

"""Étape 1b — Construction du dataset SFT unifié (~5 000 paires instruction-réponse).

On agrège ici trois ingrédients :
  1. les vignettes de triage synthétiques (format cible « symptômes → priorité ») ;
  2. les corpus de Q/R médicales collectés (MedQuAD, MedQA, FrenchMedMCQA…).

Tout est converti au **schéma unifié** `SFTRecord` (cf. `schema.py`), VALIDÉ,
dédupliqué, puis écrit dans `data/interim/sft.jsonl` (avant anonymisation).

La mission insiste : « prioriser la qualité sur la quantité » et « standardiser
les formats ». C'est exactement ce que fait ce script : un format unique,
validé champ par champ, avec des métadonnées de traçabilité.
"""

from __future__ import annotations

import argparse
import hashlib

from pydantic import ValidationError

from triage.config import INTERIM_DIR, RAW_DIR, ensure_dirs
from triage.data.schema import Metadata, SFTRecord, VitalSigns
from triage.data.triage_knowledge import generate_triage_vignettes
from triage.utils.common import get_logger, read_jsonl, write_jsonl

logger = get_logger("build_sft")


def _hash(text: str) -> str:
    """Empreinte courte d'un texte, pour repérer les doublons."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _vignette_to_record(v: dict) -> SFTRecord:
    """Convertit une vignette de triage synthétique en SFTRecord validé."""
    return SFTRecord(
        id=v["id"],
        instruction=v["instruction"],
        input=v.get("input", ""),
        output=v["output"],
        lang=v["lang"],
        task_type="triage",
        metadata=Metadata(
            symptoms=v.get("symptoms", []),
            triage_level=v.get("triage_level"),
            vitals=VitalSigns(),
            source=v["source"],
            license=v["license"],
            confidence=v.get("confidence", 0.9),
            anonymized=False,
            # On conserve la clé de présentation (elle était perdue ici, ce qui
            # rendait impossible un split par groupe en aval).
            presentation_key=v.get("presentation_key"),
        ),
    )


def _raw_to_record(row: dict, idx: int) -> SFTRecord:
    """Convertit un exemple brut (Q/R ou QCM) en SFTRecord validé."""
    return SFTRecord(
        id=f"{row['source']}-{idx:06d}",
        instruction=row["instruction"],
        input=row.get("input", ""),
        output=row["output"],
        lang=row["lang"],
        task_type=row.get("task_type", "qa"),
        metadata=Metadata(
            source=row["source"],
            license=row.get("license", "unknown"),
            # Les corpus externes ne sont pas validés cliniquement par nous :
            # on leur attribue une confiance moindre que nos vignettes contrôlées.
            confidence=0.7,
            anonymized=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction du dataset SFT unifié.")
    parser.add_argument("--target-total", type=int, default=5000,
                        help="Taille cible du dataset SFT.")
    parser.add_argument("--triage-per-presentation", type=int, default=85,
                        help="Nb de vignettes générées par tableau clinique.")
    parser.add_argument("--max-chars", type=int, default=3000,
                        help="Longueur max (instruction+réponse, en caractères) "
                             "d'un exemple externe. Au-delà, la séquence dépasse "
                             "max_seq_length à l'entraînement : la fin (<|im_end|>) "
                             "est tronquée et le modèle apprend des réponses qui "
                             "ne se terminent jamais.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    records: list[SFTRecord] = []
    seen: set[str] = set()
    errors = 0

    def _add(rec: SFTRecord) -> bool:
        """Ajoute un record s'il n'est pas un doublon (clé = instruction+sortie).

        Renvoie True si le record a réellement été ajouté (False = doublon),
        pour que les compteurs de quotas par source restent exacts.
        """
        key = _hash(rec.instruction + "||" + rec.output)
        if key in seen:
            return False
        seen.add(key)
        records.append(rec)
        return True

    # --- 1. Vignettes de triage synthétiques (priorité : format cible) ---
    vignettes = generate_triage_vignettes(
        n_per_presentation=args.triage_per_presentation, seed=args.seed
    )
    for v in vignettes:
        try:
            _add(_vignette_to_record(v))
        except ValidationError as exc:
            errors += 1
            logger.debug("Vignette rejetée : %s", exc)
    n_triage = len(records)
    logger.info("Vignettes de triage validées : %d", n_triage)

    # --- 2. Corpus externes : on complète jusqu'à la cible ---
    remaining = max(0, args.target_total - len(records))
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    # On exclut les sources DPO (préférences) de la construction SFT.
    raw_files = [p for p in raw_files if "preference" not in p.stem]

    if raw_files and remaining > 0:
        # Répartition équitable du quota restant entre les sources disponibles.
        per_source = max(1, remaining // len(raw_files))
        for path in raw_files:
            count = 0
            for idx, row in enumerate(read_jsonl(path)):
                if count >= per_source or len(records) >= args.target_total:
                    break
                # Filtre de longueur : un exemple trop long sera TRONQUÉ par le
                # trainer (max_seq_length) et perdra son marqueur de fin.
                approx_len = len(str(row.get("instruction", ""))) + len(str(row.get("output", "")))
                if approx_len > args.max_chars:
                    continue
                try:
                    if _add(_raw_to_record(row, idx)):
                        count += 1  # ne compte que les ajouts réels (pas les doublons)
                except (ValidationError, KeyError) as exc:
                    errors += 1
                    logger.debug("Exemple brut rejeté (%s) : %s", path.stem, exc)
            logger.info("Source '%s' : %d exemples ajoutés", path.stem, count)
    else:
        logger.warning(
            "Aucun corpus externe trouvé dans data/raw/ : dataset SFT limité aux "
            "vignettes de triage. Lancez d'abord `triage-collect`."
        )

    # --- 3. Écriture ---
    out_path = INTERIM_DIR / "sft.jsonl"
    payload = (r.model_dump() for r in records)
    n = write_jsonl(payload, out_path)
    logger.info("=" * 60)
    logger.info("Dataset SFT construit : %d exemples (%d triage, %d externes)",
                n, n_triage, n - n_triage)
    logger.info("Exemples rejetés (validation) : %d", errors)
    logger.info("Fichier : %s", out_path)


if __name__ == "__main__":
    main()

"""Étape 1e — Partitionnement train / val / test + jeu d'évaluation clinique.

Point de vigilance de la mission : « ne pas mélanger données d'entraînement et
données d'évaluation ». On garantit ici une séparation stricte :

  * SFT  → train (80 %) / val (10 %) / test (10 %). Les vignettes de triage
           sont réparties **par présentation clinique** (groupes) : toutes les
           vignettes d'une même présentation partagent la même réponse de
           référence (quasi-doublons) ; les répartir individuellement entre
           train et test surestimait les métriques. Les autres tâches (QA)
           restent stratifiées par (tâche, langue) au niveau enregistrement ;
  * DPO  → train (90 %) / val (10 %), réparti **par prompt** (plusieurs paires
           partagent le même prompt/chosen : les séparer train/val rendait la
           validation optimiste). Les paires portant sur une présentation ou un
           prompt réservés au test / à l'éval clinique sont ÉCARTÉES du DPO
           (sinon le modèle est entraîné à préférer la réponse attendue sur le
           prompt exact d'évaluation — contamination mesurée lors de l'audit) ;
  * un **jeu d'évaluation clinique** dédié est extrait des vignettes de triage
    du `test` : il contient le niveau de priorité attendu, ce qui permet de
    calculer des métriques cliniques (exactitude du triage, sécurité).

La graine aléatoire est fixée → partitions reproductibles d'une exécution à
l'autre (auditabilité).
"""

from __future__ import annotations

import argparse
import random

from triage.config import DPO_DIR, EVAL_DIR, INTERIM_DIR, SFT_DIR, ensure_dirs
from triage.utils.common import get_logger, read_jsonl, write_jsonl

logger = get_logger("split")


def _load(interim_name: str) -> list[dict]:
    """Charge le fichier anonymisé s'il existe, sinon la version non anonymisée."""
    anon = INTERIM_DIR / interim_name.replace(".jsonl", "_anon.jsonl")
    base = INTERIM_DIR / interim_name
    path = anon if anon.exists() else base
    if not path.exists():
        logger.warning("Aucune donnée trouvée pour %s.", interim_name)
        return []
    logger.info("Chargement de %s", path.name)
    return list(read_jsonl(path))


def _dedupe(records: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Retire les doublons exacts AVANT le split.

    Sans cela, deux copies d'un même exemple peuvent tomber l'une en train,
    l'autre en test/val → fuite de données et métriques trop optimistes.
    """
    seen: set[tuple] = set()
    out = []
    for r in records:
        key = tuple(str(r.get(f, "")) for f in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    removed = len(records) - len(out)
    if removed:
        logger.info("Doublons exacts retirés avant split : %d", removed)
    return out


def _strat(values):
    # train_test_split exige >= 2 membres par classe ; sinon on désactive.
    from collections import Counter

    counts = Counter(values)
    return values if min(counts.values()) >= 2 else None


def _split_grouped(
    keys: list[str], key_level: dict[str, str], size: float, seed: int
) -> tuple[list[str], list[str]]:
    """Sépare une liste de GROUPES en (reste, sélection) stratifiée par niveau.

    Le nombre de groupes est petit (18 présentations) : une fraction de 10 %
    ne donnerait que 2 groupes, moins que les 3 niveaux de triage. On force
    donc AU MOINS un groupe par niveau dans la sélection (sinon le jeu de test
    — et le jeu d'évaluation clinique qui en découle — perdrait un niveau).
    """
    import math

    # Import paresseux : la CI « API légère » importe ce module (via les tests
    # de régression) sans scikit-learn — seul le vrai split en a besoin.
    from sklearn.model_selection import train_test_split

    if len(keys) < 2:
        return list(keys), []
    strata_vals = [key_level[k] for k in keys]
    strata = _strat(strata_vals)
    n_sel = max(1, math.ceil(len(keys) * size))
    if strata is not None:
        n_sel = max(n_sel, len(set(strata_vals)))  # ≥ 1 groupe par niveau
    n_sel = min(n_sel, len(keys) - 1)  # ne jamais vider le « reste »
    rest, sel = train_test_split(
        keys, test_size=n_sel, random_state=seed, stratify=strata
    )
    return rest, sel


def split_sft(test_size: float, val_size: float, seed: int) -> tuple[set[str], set[str]]:
    """Partitionne le dataset SFT (par présentation pour le triage).

    Renvoie `(clés de présentation, instructions)` du jeu de TEST : le split
    DPO s'en sert pour écarter les paires qui recouperaient l'évaluation.
    """
    records = _dedupe(_load("sft.jsonl"), ("instruction", "input", "output"))
    if not records:
        return set(), set()

    rel_val = val_size / (1.0 - test_size)  # proportion ajustée après retrait du test

    # --- 1) Vignettes de triage : split PAR PRÉSENTATION CLINIQUE (groupes).
    # Toutes les vignettes d'une présentation (mêmes symptômes, même réponse
    # de référence) vont dans LA MÊME partition : le test évalue donc des
    # présentations jamais vues à l'entraînement, pas des paraphrases du train.
    # Stratification par niveau de triage → les 3 niveaux restent représentés.
    by_key: dict[str, list[dict]] = {}
    others: list[dict] = []
    for r in records:
        key = (r.get("metadata") or {}).get("presentation_key")
        if r.get("task_type") == "triage" and key:
            by_key.setdefault(key, []).append(r)
        else:
            others.append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    if by_key:
        keys = sorted(by_key)
        key_level = {
            k: (by_key[k][0].get("metadata") or {}).get("triage_level") or "-" for k in keys
        }
        tv_keys, test_keys = _split_grouped(keys, key_level, test_size, seed)
        train_keys, val_keys = _split_grouped(tv_keys, key_level, rel_val, seed)
        train = [r for k in train_keys for r in by_key[k]]
        val = [r for k in val_keys for r in by_key[k]]
        test = [r for k in test_keys for r in by_key[k]]
        logger.info(
            "Présentations de triage → train=%d, val=%d, test=%d (aucun recouvrement)",
            len(train_keys), len(val_keys), len(test_keys),
        )

    # --- 2) Autres tâches (QA…) : stratification « tâche|langue » classique.
    if others:
        # Import paresseux (cf. _split_grouped).
        from sklearn.model_selection import train_test_split

        def _stratum(r: dict) -> str:
            return f"{r.get('task_type')}|{r.get('lang')}"

        o_tv, o_test = train_test_split(
            others, test_size=test_size, random_state=seed,
            stratify=_strat([_stratum(r) for r in others]),
        )
        o_train, o_val = train_test_split(
            o_tv, test_size=rel_val, random_state=seed,
            stratify=_strat([_stratum(r) for r in o_tv]),
        )
        train += o_train
        val += o_val
        test += o_test

    # Mélange déterministe (les groupes ont été concaténés par blocs).
    for part in (train, val, test):
        random.Random(seed).shuffle(part)

    write_jsonl(train, SFT_DIR / "train.jsonl")
    write_jsonl(val, SFT_DIR / "val.jsonl")
    write_jsonl(test, SFT_DIR / "test.jsonl")
    logger.info("SFT → train=%d, val=%d, test=%d", len(train), len(val), len(test))

    # --- Jeu d'évaluation clinique dédié (triage uniquement, issu du test) ---
    clinical = []
    for r in test:
        if r.get("task_type") == "triage":
            meta = r.get("metadata", {})
            clinical.append(
                {
                    "id": r["id"],
                    "instruction": r["instruction"],
                    "lang": r["lang"],
                    "expected_triage_level": meta.get("triage_level"),
                    "reference_output": r["output"],
                    "symptoms": meta.get("symptoms", []),
                }
            )
    if clinical:
        write_jsonl(clinical, EVAL_DIR / "clinical_eval.jsonl")
        logger.info("Jeu d'évaluation clinique → %d cas (data/eval/clinical_eval.jsonl)",
                    len(clinical))

    # Clés + instructions du test : à exclure du DPO (anti-contamination).
    test_keys = {
        (r.get("metadata") or {}).get("presentation_key")
        for r in test
    } - {None}
    test_prompts = {r.get("instruction", "") for r in test} - {""}
    return test_keys, test_prompts


def split_dpo(
    val_size: float,
    seed: int,
    excluded_keys: set[str] | None = None,
    excluded_prompts: set[str] | None = None,
) -> None:
    """Partitionne le dataset DPO en train / val (groupé PAR PROMPT).

    `excluded_keys` / `excluded_prompts` : présentations et instructions
    réservées au test SFT / à l'éval clinique. Toute paire DPO qui les recoupe
    est écartée — sans cela, le DPO entraînait le modèle à préférer la réponse
    attendue sur les prompts exacts de l'évaluation (métriques gonflées).
    """
    excluded_keys = excluded_keys or set()
    excluded_prompts = excluded_prompts or set()
    records = _dedupe(_load("dpo.jsonl"), ("prompt", "chosen", "rejected"))
    if not records:
        return

    def _contaminated(r: dict) -> bool:
        key = (r.get("metadata") or {}).get("presentation_key")
        return key in excluded_keys or r.get("prompt", "") in excluded_prompts

    kept = [r for r in records if not _contaminated(r)]
    removed = len(records) - len(kept)
    if removed:
        logger.info(
            "Paires DPO écartées (présentation/prompt présents dans le test SFT "
            "ou l'éval clinique) : %d", removed,
        )
    if not kept:
        logger.warning("Plus aucune paire DPO après exclusion : split annulé.")
        return

    # Split PAR PROMPT : plusieurs paires partagent le même prompt (variantes
    # de `rejected`) ; les répartir train/val rendait la validation optimiste.
    by_prompt: dict[str, list[dict]] = {}
    for r in kept:
        by_prompt.setdefault(r.get("prompt", ""), []).append(r)
    prompts = sorted(by_prompt)
    # Import paresseux (cf. _split_grouped).
    from sklearn.model_selection import train_test_split

    train_p, val_p = train_test_split(prompts, test_size=val_size, random_state=seed)
    train = [r for p in train_p for r in by_prompt[p]]
    val = [r for p in val_p for r in by_prompt[p]]
    for part in (train, val):
        random.Random(seed).shuffle(part)
    write_jsonl(train, DPO_DIR / "train.jsonl")
    write_jsonl(val, DPO_DIR / "val.jsonl")
    logger.info("DPO → train=%d, val=%d", len(train), len(val))


def main() -> None:
    parser = argparse.ArgumentParser(description="Partitionnement des datasets.")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--dpo-val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    test_keys, test_prompts = split_sft(args.test_size, args.val_size, args.seed)
    split_dpo(args.dpo_val_size, args.seed, test_keys, test_prompts)
    logger.info("Partitionnement terminé.")


if __name__ == "__main__":
    main()

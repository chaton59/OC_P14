"""Tests de l'équilibre des paires de préférence DPO.

Bug historique : URGENCE_VITALE n'apparaissait JAMAIS comme « rejected » — le
gradient DPO poussait le modèle à sur-classer tous les cas vers l'urgence
vitale. Politique actuelle : une asymétrie PRUDENTE et BORNÉE — le sous-triage
(l'erreur interdite) est montré un peu plus souvent comme mauvaise réponse,
mais chaque niveau reste régulièrement rejeté (ratio max/min ≤ 2).
"""

from collections import Counter

from triage.config import TRIAGE_LEVELS
from triage.data.triage_knowledge import (
    _SEVERITY_ORDER,
    _WRONG_LEVELS,
    PRESENTATIONS,
    make_bad_triage_answer,
)

_VARIANTS = (0, 1, 2)


def _rejected_counts() -> Counter:
    return Counter(
        _WRONG_LEVELS[p.level][variant % 3]
        for p in PRESENTATIONS
        for variant in _VARIANTS
    )


def test_tous_les_niveaux_apparaissent_en_rejected():
    """Chaque niveau doit pouvoir être une « mauvaise réponse » (rejected)."""
    rejected = _rejected_counts()
    assert set(rejected) == set(TRIAGE_LEVELS), (
        "Un niveau n'apparaît jamais en rejected : le DPO apprendrait à ne "
        "jamais le remettre en cause (sur- ou sous-triage systématique)."
    )


def test_asymetrie_bornee():
    """Aucun niveau ne doit être rejeté plus de 2× plus souvent qu'un autre."""
    rejected = _rejected_counts()
    assert max(rejected.values()) <= 2 * min(rejected.values()), (
        f"Asymétrie excessive des rejets DPO : {dict(rejected)}"
    )


def test_prudence_le_sous_triage_est_plus_souvent_rejete():
    """L'orientation « sécurité » : le sous-triage (classer moins grave) doit
    être montré comme mauvaise réponse au moins aussi souvent que le
    sur-triage — jamais l'inverse."""
    under = over = 0
    for p in PRESENTATIONS:
        for variant in _VARIANTS:
            wrong = _WRONG_LEVELS[p.level][variant % 3]
            if _SEVERITY_ORDER.index(wrong) < _SEVERITY_ORDER.index(p.level):
                under += 1
            else:
                over += 1
    assert under >= over, f"sous-triage rejeté {under}× vs sur-triage {over}×"


def test_mauvaise_reponse_meme_gabarit_niveau_different():
    """La mauvaise réponse garde le gabarit (pas de biais de longueur) mais
    porte un niveau différent du bon."""
    for p in PRESENTATIONS[:3]:
        for variant in _VARIANTS:
            bad = make_bad_triage_answer(p, "fr", variant=variant)
            first_line = bad.split("\n", 1)[0]
            assert first_line.startswith("Niveau de priorité :")
            assert p.level not in first_line
            # Gabarit complet : analyse + recommandation présentes.
            assert "Analyse :" in bad and "Recommandation :" in bad

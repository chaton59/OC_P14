"""Tests du rapprochement des labels mal orthographiés par le modèle.

Constat en conditions réelles (10 juillet, décodage glouton, page /ui) : le
modèle 1.7B a généré « Priority level: URGENCE_MODERIEE ». La ligne structurée
devenait illisible, l'extraction retombait sur les formulations libres et
trouvait « life-threatening » dans l'analyse → bandeau URGENCE_VITALE alors
que le texte affiché disait MODERIEE. Deux garde-fous ont été ajoutés :

  1. `extract_triage_level` tolère les fautes proches dans la ligne structurée
     (rapprochement par distance d'édition, jamais de « devinette ») ;
  2. `normalize_level_mentions` corrige le label fautif dans le texte affiché.
"""

from triage.serve.triage_agent import (
    extract_triage_level,
    normalize_level_mentions,
)


class TestExtractionTolerante:
    def test_typo_moderiee_ligne_structuree(self):
        txt = "Priority level: URGENCE_MODERIEE (Immediate recommendation)"
        assert extract_triage_level(txt) == "URGENCE_MODEREE"

    def test_cas_reel_la_ligne_structuree_prime_sur_l_analyse(self):
        # Reproduction du cas /ui : la ligne structurée (même fautive) doit
        # primer sur « life-threatening » mentionné plus loin dans l'analyse.
        txt = (
            "Priority level: URGENCE_MODERIEE (Immediate recommendation)\n"
            "Assessment: quick evaluation is needed to rule out "
            "life-threatening complications."
        )
        assert extract_triage_level(txt) == "URGENCE_MODEREE"

    def test_labels_canoniques_inchanges(self):
        assert (
            extract_triage_level("Niveau de priorité : URGENCE_VITALE")
            == "URGENCE_VITALE"
        )
        assert (
            extract_triage_level("Priority level: CONSULTATION_DIFFEREE")
            == "CONSULTATION_DIFFEREE"
        )

    def test_variante_accentuee_en_prose(self):
        assert (
            extract_triage_level("Priorité : urgence modérée (surveillance)")
            == "URGENCE_MODEREE"
        )

    def test_token_trop_eloigne_non_devine(self):
        # Un token sans rapport ne doit pas être rapproché de force.
        assert extract_triage_level("Priority level: URGENCE_INCONNUE") is None


class TestNormalisationAffichage:
    def test_remplace_le_label_fautif(self):
        txt = "Priority level: URGENCE_MODERIEE (Immediate)"
        fixed = normalize_level_mentions(txt)
        assert "URGENCE_MODEREE" in fixed
        assert "MODERIEE" not in fixed

    def test_label_canonique_intact(self):
        txt = "Niveau de priorité : URGENCE_VITALE\nAppelez le 15."
        assert normalize_level_mentions(txt) == txt

    def test_ne_touche_pas_la_prose(self):
        txt = "Il s'agit d'une urgence modérée, sans signe de gravité."
        assert normalize_level_mentions(txt) == txt

    def test_token_inconnu_laisse_tel_quel(self):
        txt = "Priority level: URGENCE_INCONNUE"
        assert normalize_level_mentions(txt) == txt

    def test_texte_vide(self):
        assert normalize_level_mentions("") == ""

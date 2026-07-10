"""Tests du questionnaire adaptatif et du filet de sécurité clinique."""

from triage.serve.questionnaire import (
    build_followup,
    detect_language,
    detect_red_flags,
    missing_dimensions,
)


def test_red_flag_douleur_thoracique():
    """Une douleur thoracique doit déclencher un signe d'alerte vital."""
    flags = detect_red_flags("J'ai une grosse douleur dans la poitrine")
    assert "Douleur thoracique" in flags


def test_red_flag_avc_anglais():
    """Les signes d'AVC doivent être détectés en anglais aussi."""
    flags = detect_red_flags("sudden facial drooping and slurred speech")
    assert "Signes d'AVC" in flags


def test_pas_de_red_flag_benin():
    """Un rhume ne doit pas déclencher de signe d'alerte."""
    assert detect_red_flags("nez qui coule et éternuements") == []


def test_red_flag_nie_ne_declenche_pas():
    """Un signe d'alerte NIÉ ne doit pas déclencher (bug : sur-triage des bénins)."""
    assert detect_red_flags("Je n'ai pas de douleur thoracique, juste un rhume") == []
    assert detect_red_flags("no chest pain, only a mild cough") == []
    assert detect_red_flags("aucune difficulté à respirer, légère fatigue") == []


def test_red_flag_present_malgre_negation_ailleurs():
    """Une négation portant sur AUTRE CHOSE ne doit pas masquer un vrai signe."""
    flags = detect_red_flags("Pas de fièvre. En revanche, douleur thoracique intense.")
    assert "Douleur thoracique" in flags


def test_detection_langue():
    assert detect_language("J'ai mal à la tête depuis hier") == "fr"
    assert detect_language("I have a headache since yesterday") == "en"


def test_followup_message_vague():
    """Un message très court sans signe d'alerte doit déclencher une relance."""
    needs, questions, dims, lang = build_followup("j'ai mal")
    assert needs is True
    assert len(questions) >= 1
    # Les dimensions posées sont renvoyées (le client doit pouvoir les réinjecter).
    assert len(dims) == len(questions)


def test_pas_de_followup_si_red_flag():
    """En présence d'un signe d'alerte, on ne pose pas de question : on escalade."""
    needs, questions, _, _ = build_followup("douleur thoracique violente")
    assert needs is False


def test_dimensions_manquantes():
    """Un message sans durée ni intensité doit signaler ces dimensions."""
    missing = missing_dimensions("j'ai de la fièvre")
    assert "onset" in missing and "severity" in missing

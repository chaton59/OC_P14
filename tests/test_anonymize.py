"""Tests de l'anonymisation RGPD (mode repli regex, sans dépendances lourdes).

On utilise volontairement le repli `regex_anonymize` pour que ces tests soient
rapides et exécutables en CI sans modèles spaCy.
"""

from triage.data.anonymize import regex_anonymize


def test_email_masque():
    txt = "Contactez le patient à jean.dupont@example.com pour le suivi."
    out, counts = regex_anonymize(txt)
    assert "<EMAIL>" in out
    assert "@example.com" not in out
    assert counts.get("EMAIL_ADDRESS", 0) == 1


def test_telephone_francais_masque():
    txt = "Son numéro est le 06 12 34 56 78, rappelez-le."
    out, counts = regex_anonymize(txt)
    assert "<TELEPHONE>" in out
    assert "06 12 34 56 78" not in out


def test_nir_masque():
    txt = "NIR : 1 84 12 75 116 001 42 du dossier."
    out, _ = regex_anonymize(txt)
    assert "<NIR>" in out


def test_texte_sans_pii_inchange():
    txt = "Le patient présente une fièvre à 39°C depuis deux jours."
    out, counts = regex_anonymize(txt)
    assert out == txt
    assert counts == {}

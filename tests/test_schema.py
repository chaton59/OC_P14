"""Tests du schéma de données (validation Pydantic)."""

import pytest
from pydantic import ValidationError

from triage.data.schema import DPORecord, Metadata, SFTRecord, VitalSigns


def test_sft_record_valide():
    """Un enregistrement SFT bien formé doit être accepté."""
    rec = SFTRecord(
        id="x-1",
        instruction="J'ai mal à la tête",
        output="Soins différés possibles…",
        lang="fr",
        task_type="triage",
        metadata=Metadata(source="triage_synth", triage_level="CONSULTATION_DIFFEREE"),
    )
    assert rec.lang == "fr"
    assert rec.metadata.triage_level == "CONSULTATION_DIFFEREE"


def test_sft_record_langue_invalide():
    """Une langue hors {fr, en} doit être rejetée."""
    with pytest.raises(ValidationError):
        SFTRecord(id="x", instruction="q", output="r", lang="de",
                  task_type="qa", metadata=Metadata(source="s"))


def test_confidence_bornee():
    """La confiance doit rester dans [0, 1]."""
    with pytest.raises(ValidationError):
        Metadata(source="s", confidence=1.5)


def test_vitals_optionnelles():
    """Les constantes vitales sont toutes optionnelles (None par défaut)."""
    v = VitalSigns()
    assert v.temperature_c is None and v.spo2_percent is None


def test_dpo_record_valide():
    """Un triplet préférentiel bien formé doit être accepté."""
    rec = DPORecord(
        id="d-1", prompt="p", chosen="bonne", rejected="mauvaise",
        lang="en", task_type="qa", metadata=Metadata(source="ultramedical"),
    )
    assert rec.chosen == "bonne" and rec.rejected == "mauvaise"

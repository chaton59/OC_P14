"""Schéma des données et des métadonnées.

La mission demande explicitement de *« définir le schéma des métadonnées
(symptômes, antécédents, constantes, source, niveau de confiance) »*.

On le formalise ici avec des modèles **Pydantic**. L'intérêt : chaque exemple
de notre dataset est *validé* automatiquement (types corrects, champs requis
présents). Si une transformation casse le format, on le détecte tout de suite —
c'est une garantie de qualité et d'auditabilité essentielle en santé.

Deux formats finaux coexistent :
  * `SFTRecord` — paire instruction/réponse pour le fine-tuning supervisé ;
  * `DPORecord` — triplet (prompt, réponse préférée, réponse rejetée) pour DPO.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Types contraints réutilisés (lisibilité + validation automatique).
Lang = Literal["fr", "en"]
TaskType = Literal["qa", "mcqa", "triage"]
TriageLevel = Literal[
    "URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE"
]


class VitalSigns(BaseModel):
    """Constantes vitales d'un patient (toutes optionnelles).

    On modélise les paramètres les plus discriminants pour le triage. `None`
    signifie « non renseigné » (fréquent : un patient ne fournit pas tout).
    """

    temperature_c: float | None = None        # température en °C
    heart_rate_bpm: int | None = None          # fréquence cardiaque (batt/min)
    systolic_bp_mmhg: int | None = None         # tension artérielle systolique
    respiratory_rate: int | None = None         # fréquence respiratoire
    spo2_percent: int | None = None             # saturation en oxygène (%)
    pain_score_0_10: int | None = None          # douleur sur l'échelle 0-10


class Metadata(BaseModel):
    """Métadonnées cliniques et de traçabilité attachées à chaque exemple."""

    # --- Données cliniques structurées ---
    symptoms: list[str] = Field(default_factory=list)      # symptômes déclarés
    antecedents: list[str] = Field(default_factory=list)   # antécédents médicaux
    vitals: VitalSigns = Field(default_factory=VitalSigns)  # constantes vitales
    triage_level: TriageLevel | None = None             # niveau cible (si triage)

    # --- Traçabilité / qualité (auditabilité) ---
    source: str                                            # corpus d'origine
    license: str = "unknown"                                # licence de la source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # niveau de confiance [0;1]
    anonymized: bool = False                                # PII masquées ? (RGPD)
    # Clé du tableau clinique d'origine (vignettes synthétiques uniquement) :
    # indispensable pour des splits par GROUPE (éviter que des quasi-doublons
    # d'une même présentation se retrouvent à la fois en train et en test).
    presentation_key: str | None = None


class SFTRecord(BaseModel):
    """Un exemple de fine-tuning supervisé (instruction → réponse)."""

    id: str
    instruction: str            # la question / consigne adressée à l'agent
    input: str = ""             # contexte additionnel éventuel (peut être vide)
    output: str                 # la réponse attendue (cible d'apprentissage)
    lang: Lang
    task_type: TaskType
    metadata: Metadata


class DPORecord(BaseModel):
    """Un exemple d'alignement par préférences (DPO).

    `chosen` = réponse de meilleure qualité / plus sûre cliniquement.
    `rejected` = réponse moins pertinente, incomplète ou potentiellement dangereuse.
    Le modèle apprend à préférer la première à la seconde.
    """

    id: str
    prompt: str                 # la question / situation clinique
    chosen: str                 # réponse préférée
    rejected: str               # réponse rejetée
    lang: Lang
    task_type: TaskType
    metadata: Metadata

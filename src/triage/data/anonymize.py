"""Étape 1d — Anonymisation RGPD des données (Microsoft Presidio).

EXIGENCE RGPD
-------------
La mission impose d'« anonymiser toutes les données et documenter le processus
RGPD ». Même si nos corpus sont publics et nos vignettes synthétiques, un
texte médical peut contenir des informations personnelles (PII) : noms,
prénoms, dates, adresses, e-mails, numéros de téléphone, etc. On les masque
systématiquement AVANT toute utilisation pour l'entraînement.

OUTIL : Presidio (open source, Microsoft).
  * `AnalyzerEngine`   détecte les entités sensibles (PERSON, EMAIL, PHONE…) ;
  * `AnonymizerEngine` les remplace par des marqueurs neutres (<PERSONNE>…).

On combine :
  * un moteur multilingue (spaCy fr + en) pour les noms/personnes ;
  * des `PatternRecognizer` regex pour des PII robustes (e-mail, téléphone FR,
    n° de sécurité sociale français, dates), indépendants de la langue.

TRAÇABILITÉ : on conserve, pour chaque enregistrement, le nombre d'entités
masquées et leurs types — preuve auditable du traitement (sans jamais stocker
la donnée personnelle d'origine).
"""

from __future__ import annotations

import argparse
import re

from triage.config import INTERIM_DIR, ensure_dirs
from triage.utils.common import get_logger, read_jsonl, write_jsonl

logger = get_logger("anonymize")

# Marqueurs de remplacement lisibles (stratégie « replace »).
REPLACEMENTS = {
    "PERSON": "<PERSONNE>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<TELEPHONE>",
    "FR_SSN": "<NIR>",                 # numéro de sécurité sociale français
    "IP_ADDRESS": "<IP>",
    "CREDIT_CARD": "<CARTE>",
}

# Entités réellement IDENTIFIANTES que l'on cible (priorité de la mission :
# « à défaut nom, prénom des patients »). On EXCLUT volontairement LOCATION et
# DATE_TIME : dans un corpus de Q/R médicales, ils désignent presque toujours
# des notions cliniques (anatomie, dates de symptômes…) et non des données
# patient. Les cibler provoquerait un sur-masquage massif qui dégraderait la
# qualité d'entraînement, pour un gain de confidentialité quasi nul.
TARGET_ENTITIES = list(REPLACEMENTS.keys())

# Seuil de confiance minimal pour qu'une détection soit retenue. Plus il est
# élevé, moins on a de faux positifs (mais on peut rater des cas ambigus).
SCORE_THRESHOLD = 0.6


def build_analyzer():
    """Construit un AnalyzerEngine bilingue (fr + en) avec reconnaisseurs regex.

    On essaie d'utiliser les modèles spaCy `fr_core_news_md` / `en_core_web_md`.
    S'ils ne sont pas installés, Presisio fonctionnera tout de même sur les
    motifs regex (e-mail, téléphone, NIR…), qui couvrent l'essentiel du risque.
    """
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Configuration du moteur NLP multilingue.
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_md"},
        ],
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        supported_languages = ["fr", "en"]
    except Exception as exc:  # noqa: BLE001 — modèles spaCy absents → repli regex
        logger.warning("Modèles spaCy indisponibles (%s). Repli sur regex seules.", exc)
        # Moteur minimal : un petit modèle anglais par défaut ou rien.
        configuration = {"nlp_engine_name": "spacy",
                         "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
        try:
            nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
            supported_languages = ["en"]
        except Exception:  # noqa: BLE001
            nlp_engine = None
            supported_languages = ["en"]

    if nlp_engine is not None:
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=supported_languages
        )
    else:
        analyzer = AnalyzerEngine(supported_languages=supported_languages)

    # --- Reconnaisseurs regex robustes, indépendants de la langue ---
    # Presidio n'applique un PatternRecognizer QUE pour sa langue déclarée : on
    # les enregistre donc pour CHAQUE langue supportée, sinon un NIR ou un
    # téléphone français présent dans un texte anglais (MedQuAD, MedQA…)
    # passerait au travers du masquage.
    for lang in supported_languages:
        # Numéro de sécurité sociale français (NIR) : 13 chiffres + clé à 2 chiffres.
        fr_ssn = PatternRecognizer(
            supported_entity="FR_SSN",
            patterns=[Pattern(name="nir", regex=r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b", score=0.85)],
            supported_language=lang,
        )
        # Téléphone français (06 12 34 56 78, +33 6 …).
        fr_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[Pattern(name="tel_fr", regex=r"\b(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}\b", score=0.7)],
            supported_language=lang,
        )
        analyzer.registry.add_recognizer(fr_ssn)
        analyzer.registry.add_recognizer(fr_phone)
    return analyzer, supported_languages


def build_anonymizer():
    """Construit le moteur d'anonymisation et la configuration de remplacement."""
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    engine = AnonymizerEngine()
    operators = {
        entity: OperatorConfig("replace", {"new_value": marker})
        for entity, marker in REPLACEMENTS.items()
    }
    # Opérateur par défaut pour toute entité non listée explicitement.
    operators["DEFAULT"] = OperatorConfig("replace", {"new_value": "<DONNEE_MASQUEE>"})
    return engine, operators


# Repli « léger » sans Presidio : regex e-mail/téléphone/NIR. Sert de filet de
# sécurité et permet aux tests unitaires de tourner sans dépendances lourdes.
_FALLBACK_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}\b"),
    "FR_SSN": re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"),
}


def regex_anonymize(text: str) -> tuple[str, dict[str, int]]:
    """Anonymisation de secours par regex pure. Renvoie (texte, comptes par type)."""
    counts: dict[str, int] = {}
    for entity, pattern in _FALLBACK_PATTERNS.items():
        text, n = pattern.subn(REPLACEMENTS[entity], text)
        if n:
            counts[entity] = counts.get(entity, 0) + n
    return text, counts


class Anonymizer:
    """Encapsule la logique d'anonymisation, avec repli automatique sur regex."""

    def __init__(self, use_presidio: bool = True):
        self.analyzer = None
        self.anonymizer = None
        self.operators = None
        self.languages = ["en"]
        if use_presidio:
            try:
                self.analyzer, self.languages = build_analyzer()
                self.anonymizer, self.operators = build_anonymizer()
                logger.info("Presidio initialisé (langues : %s).", self.languages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Presidio indisponible (%s). Repli sur regex.", exc)

    def anonymize_text(self, text: str, lang: str = "fr") -> tuple[str, dict[str, int]]:
        """Anonymise un texte. Renvoie (texte_anonymisé, comptes d'entités masquées)."""
        if not text:
            return text, {}
        # Langue non supportée par le moteur → on bascule sur l'anglais ou la regex.
        if self.analyzer is not None:
            use_lang = lang if lang in self.languages else self.languages[0]
            try:
                results = self.analyzer.analyze(
                    text=text,
                    language=use_lang,
                    entities=TARGET_ENTITIES,
                    score_threshold=SCORE_THRESHOLD,
                )
                anonymized = self.anonymizer.anonymize(
                    text=text, analyzer_results=results, operators=self.operators
                )
                counts: dict[str, int] = {}
                for r in results:
                    counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
                return anonymized.text, counts
            except Exception as exc:  # noqa: BLE001
                logger.debug("Echec Presidio sur un texte (%s). Repli regex.", exc)
        return regex_anonymize(text)


def anonymize_file(in_name: str, text_fields: list[str], anonymizer: Anonymizer) -> dict:
    """Anonymise un fichier JSONL d'interim et écrit `<nom>_anon.jsonl`.

    `text_fields` = champs textuels à nettoyer (ex : instruction, output…).
    Renvoie un petit rapport agrégé (total d'entités masquées par type).
    """
    in_path = INTERIM_DIR / in_name
    out_path = INTERIM_DIR / in_name.replace(".jsonl", "_anon.jsonl")
    if not in_path.exists():
        logger.warning("Fichier introuvable : %s (étape ignorée).", in_path)
        return {}

    report: dict[str, int] = {}
    total_records = 0

    def _process():
        nonlocal total_records
        for rec in read_jsonl(in_path):
            lang = rec.get("lang", "fr")
            source = rec.get("metadata", {}).get("source", "") if isinstance(rec.get("metadata"), dict) else ""
            masked_here = 0
            # On NE masque PAS les données 100 % synthétiques (générées par nous,
            # sans aucune PII) : les passer dans Presidio ne ferait qu'introduire
            # des faux positifs (ex. « Délai » détecté comme nom propre) qui
            # corrompraient le format cible appris par le modèle.
            is_synthetic = source.startswith("triage_synth")
            if not is_synthetic:
                for field in text_fields:
                    if field in rec and isinstance(rec[field], str):
                        rec[field], counts = anonymizer.anonymize_text(rec[field], lang)
                        for k, v in counts.items():
                            report[k] = report.get(k, 0) + v
                            masked_here += v
            # Traçabilité : tout enregistrement est tracé (synthétique = sans PII).
            rec.setdefault("metadata", {})
            if isinstance(rec["metadata"], dict):
                rec["metadata"]["anonymized"] = True
                rec["metadata"]["pii_masked"] = masked_here
            total_records += 1
            yield rec

    n = write_jsonl(_process(), out_path)
    logger.info("Anonymisé %s → %s (%d enregistrements).", in_name, out_path.name, n)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymisation RGPD (Presidio).")
    parser.add_argument("--no-presidio", action="store_true",
                        help="Forcer le repli regex (utile en CI/tests).")
    args = parser.parse_args()

    ensure_dirs()
    anonymizer = Anonymizer(use_presidio=not args.no_presidio)

    global_report: dict[str, int] = {}
    # SFT : on nettoie instruction, input et output.
    r1 = anonymize_file("sft.jsonl", ["instruction", "input", "output"], anonymizer)
    # DPO : on nettoie prompt, chosen et rejected.
    r2 = anonymize_file("dpo.jsonl", ["prompt", "chosen", "rejected"], anonymizer)

    for r in (r1, r2):
        for k, v in r.items():
            global_report[k] = global_report.get(k, 0) + v

    logger.info("=" * 60)
    logger.info("Rapport d'anonymisation (entités masquées par type) :")
    if global_report:
        for k, v in sorted(global_report.items(), key=lambda kv: -kv[1]):
            logger.info("  %-16s : %d", k, v)
    else:
        logger.info("  Aucune PII détectée (corpus déjà propres / synthétiques).")


if __name__ == "__main__":
    main()

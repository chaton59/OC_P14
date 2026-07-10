"""Questionnaire adaptatif + filet de sécurité clinique (règles).

Deux mécanismes complètent le modèle de langage :

1. **Filet de sécurité (red flags).** Indépendamment du LLM, on repère par
   mots-clés des signes d'alerte vitaux (douleur thoracique, AVC, détresse
   respiratoire…). En présence d'un tel signe, on force le niveau
   `URGENCE_VITALE`. C'est une garde-fou : même si le modèle se trompe, on ne
   sous-estime jamais une urgence vitale évidente. La sécurité prime.

2. **Questionnaire adaptatif.** Si le message du patient est trop pauvre pour
   trancher (et sans signe d'alerte), on pose 1 à 2 questions ciblées (depuis
   quand ? intensité ? signes associés ?) avant de produire l'évaluation. C'est
   le « questionnaire intelligent adaptatif » demandé par la mission.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. Signes d'alerte vitaux (bilingues). Clé = libellé, valeur = motifs regex.
# ---------------------------------------------------------------------------
RED_FLAGS: dict[str, list[str]] = {
    "Douleur thoracique": [r"douleur.{0,15}(thorax|poitrine|thoracique)", r"chest pain"],
    "Signes d'AVC": [
        r"paralys", r"bouche.{0,10}tordue", r"trouble.{0,10}(parole|langage)",
        # `facial?` : « facial drooping » n'était pas couvert (seulement « face »).
        r"facial?.{0,10}droop", r"slurred speech", r"weakness.{0,15}(arm|side)",
        r"faiblesse.{0,20}(bras|jambe|membre|c[ôo]t[ée])", r"(arm|leg|limb)\s+weakness",
    ],
    "Détresse respiratoire": [
        r"(difficult|gêne|mal).{0,15}respir", r"étouff", r"suffoqu",
        r"can.?t breathe", r"shortness of breath", r"lèvres?.{0,10}bleu",
        r"blue lips", r"bluish lips",
        # Symptômes indirects fréquents des tableaux vitaux (formulations qui
        # échappaient au filet quand le mot-clé principal manquait).
        r"essoufflement", r"incapa\w*.{0,15}parler", r"unable to speak",
        r"(difficulty|trouble)\s+breathing", r"breathing\s+difficult",
    ],
    "Hémorragie sévère": [
        r"saigne.{0,15}(abondant|beaucoup|important|stop|arrêt)", r"hémorragie",
        r"heavy bleeding", r"won.?t stop bleeding",
        # Pâleur ET vertiges ensemble : évocateur d'un choc hémorragique, même
        # sans mention explicite du saignement. (Exigés TOUS LES DEUX : chacun
        # isolé serait trop peu spécifique → sur-triage.)
        r"pâleur.{0,40}vertiges", r"vertiges.{0,40}pâleur",
        r"pallor.{0,40}dizziness", r"dizziness.{0,40}pallor",
    ],
    "Anaphylaxie": [
        r"gorge.{0,15}(serre|ferme|gonfl)", r"throat.{0,10}(closing|swelling)",
        r"choc anaphylact", r"anaphyla",
        # Ordres inversés non couverts (« gonflement du visage et de la gorge »).
        r"gonflement.{0,30}(visage|gorge)", r"swelling.{0,25}(face|throat)",
        r"urticaire g[ée]n[ée]ralis", r"widespread hives",
    ],
    "Trouble de la conscience": [
        r"(perte|perd).{0,15}connaissance", r"inconscient", r"convuls",
        r"unconscious", r"seizure", r"unresponsive", r"ne (répond|réagit) plus",
        r"absence de réponse",
    ],
}


# Marqueurs de négation : « pas de douleur thoracique », « no chest pain »…
# On regarde une courte fenêtre AVANT le motif détecté, sans franchir de
# ponctuation forte (pour ne pas hériter d'une négation d'une autre phrase).
_NEGATION_BEFORE = re.compile(
    r"\b(?:pas|plus|aucune?|sans|ni|jamais|n[ée]gatif|nie|no|not|without|den(?:y|ies))\b"
    r"[^.!?;\n]{0,20}$",
    re.IGNORECASE,
)


def is_negated(text: str, position: int, window: int = 35) -> bool:
    """Vrai si le texte juste avant `position` contient une négation.

    Évite les faux positifs du type « je n'ai PAS de douleur thoracique »
    (qui forçait à tort le niveau URGENCE_VITALE → sur-triage des cas bénins).
    """
    return bool(_NEGATION_BEFORE.search(text[max(0, position - window):position]))


def detect_red_flags(text: str) -> list[str]:
    """Renvoie la liste des signes d'alerte vitaux repérés dans le texte.

    Un motif précédé d'une négation (« pas de… », « no… ») est ignoré : il
    signale l'ABSENCE du signe, pas sa présence.
    """
    text_low = text.lower()
    found = []
    for label, patterns in RED_FLAGS.items():
        matched = any(
            not is_negated(text_low, m.start())
            for p in patterns
            for m in re.finditer(p, text_low)
        )
        if matched:
            found.append(label)
    return found


# ---------------------------------------------------------------------------
# 2. Questionnaire adaptatif : dimensions clés et questions de relance.
# ---------------------------------------------------------------------------
# On vérifie si le message couvre quelques dimensions essentielles du triage.
_DIMENSIONS = {
    "onset": [r"depuis", r"il y a", r"hier", r"aujourd|ce matin", r"ago", r"since", r"started"],
    # `l[ée]g[èe]r` : couvre « léger », « légère » (l'accent grave du féminin
    # échappait au motif `léger` → relance inutile sur « douleur légère »).
    "severity": [r"intens|sévère|fort|vif|insupportable|l[ée]g[èe]r|modér",
                 r"sever|mild|moderate|intense|\b\d\s*/\s*10"],
}

_FOLLOWUP_QUESTIONS = {
    "fr": {
        "onset": "Depuis combien de temps ressentez-vous ces symptômes ?",
        "severity": "Comment évalueriez-vous l'intensité (légère, modérée, forte) ?",
        "associated": "Avez-vous d'autres symptômes associés (fièvre, essoufflement, vomissements…) ?",
    },
    "en": {
        "onset": "How long have you had these symptoms?",
        "severity": "How would you rate the intensity (mild, moderate, severe)?",
        "associated": "Do you have any other associated symptoms (fever, breathlessness, vomiting…)?",
    },
}


def detect_language(text: str) -> str:
    """Détection de langue très simple (français vs anglais) par mots fréquents."""
    fr_markers = re.findall(r"\b(je|j'ai|depuis|douleur|mal|et|une|des|avec)\b", text.lower())
    en_markers = re.findall(r"\b(i|have|since|pain|and|with|the|my)\b", text.lower())
    return "fr" if len(fr_markers) >= len(en_markers) else "en"


def missing_dimensions(text: str) -> list[str]:
    """Renvoie les dimensions clés non couvertes par le message."""
    text_low = text.lower()
    missing = []
    for dim, patterns in _DIMENSIONS.items():
        if not any(re.search(p, text_low) for p in patterns):
            missing.append(dim)
    return missing


def build_followup(text: str, asked: list[str] | None = None, max_questions: int = 2):
    """Décide s'il faut relancer le patient et, si oui, propose des questions.

    Renvoie un tuple `(needs_followup: bool, questions: list[str],
    dimensions: list[str], lang: str)`. Les `dimensions` (clés : "onset",
    "severity", "associated") sont renvoyées pour que le CLIENT de l'API puisse
    les réinjecter dans `asked_dimensions` au tour suivant — sans elles, il ne
    pourrait pas savoir quelle dimension correspond au texte de la question.
    On ne relance pas si : un signe d'alerte est présent (on escalade direct),
    le message est déjà assez riche, ou on a déjà posé assez de questions.
    """
    asked = asked or []
    lang = detect_language(text)

    # Présence d'un signe d'alerte → pas de questionnaire, on tranche tout de suite.
    if detect_red_flags(text):
        return False, [], [], lang

    missing = [d for d in missing_dimensions(text) if d not in asked]
    # Message déjà détaillé (long et couvrant les dimensions) → pas de relance.
    too_short = len(text.split()) < 8
    if not missing and not too_short:
        return False, [], [], lang
    if len(asked) >= max_questions:
        return False, [], [], lang  # on a assez questionné, on tranche

    dims = missing[:1]  # une dimension à la fois
    if not dims:  # message court mais dimensions couvertes → demande générale
        dims = ["associated"]
    questions = [_FOLLOWUP_QUESTIONS[lang][d] for d in dims]
    return True, questions, dims, lang

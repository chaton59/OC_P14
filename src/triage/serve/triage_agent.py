"""Agent de triage — orchestration du message patient jusqu'au verdict.

Cet agent combine trois briques (cf. modules voisins) :
  1. le **filet de sécurité** (`questionnaire.detect_red_flags`) ;
  2. le **questionnaire adaptatif** (`questionnaire.build_followup`) ;
  3. le **modèle de langage** (`inference.get_backend`) pour l'analyse fine.

Il produit un résultat structuré, applique des garde-fous de sécurité, puis
journalise l'interaction pour l'audit.
"""

from __future__ import annotations

import re
import time
from difflib import get_close_matches

from triage.config import MEDICAL_DISCLAIMER, TRIAGE_LEVELS
from triage.prompts import build_messages
from triage.serve import audit
from triage.serve.inference import InferenceBackend, get_backend
from triage.serve.questionnaire import (
    build_followup,
    detect_language,
    detect_red_flags,
    is_negated,
)
from triage.utils.common import get_logger

logger = get_logger("serve.agent")

# Ligne structurée du gabarit d'entraînement : « Niveau de priorité : X » /
# « Priority level: X ». C'est le signal le plus fiable, on le lit en premier.
# La capture est volontairement large : constaté en conditions réelles (10/07,
# décodage glouton), le modèle 1.7B écrit parfois un label APPROCHÉ
# (« URGENCE_MODERIEE ») ; `_normalize_level_token` le rapproche ensuite du
# label canonique le plus proche, ou renvoie None (on ne devine pas).
_STRUCTURED_LEVEL = re.compile(
    r"(?:niveau(?:\s+de\s+priorit[ée])?|priority\s+level|priorit[ée])\s*:\s*"
    r"((?:urgence|consultation)[\s_-]+[a-zà-ÿ]+)",
    re.IGNORECASE,
)

# Label « canonique mais mal orthographié » tel qu'il apparaît dans le texte
# généré : MAJUSCULES reliées par underscore/tiret uniquement — on ne touche
# jamais à la prose libre (« une urgence modérée » reste intacte).
_LEVEL_TOKEN = re.compile(r"\b(?:URGENCE|CONSULTATION)[_-][A-ZÉÈ]{4,}\b")

# Accents possibles dans un label généré, à replier avant comparaison.
_ACCENT_MAP = str.maketrans("ÉÈÊÀÂÎÏÔÛÙ", "EEEAAIIOUU")


def _normalize_level_token(token: str) -> str | None:
    """Rapproche un label généré du label canonique (tolère les fautes proches).

    « URGENCE_MODERIEE » → « URGENCE_MODEREE ». Renvoie None si le token est
    trop éloigné de tout label connu.
    """
    cleaned = re.sub(r"[\s-]+", "_", token.strip().upper()).translate(_ACCENT_MAP)
    if cleaned in TRIAGE_LEVELS:
        return cleaned
    close = get_close_matches(cleaned, list(TRIAGE_LEVELS), n=1, cutoff=0.85)
    return close[0] if close else None


def normalize_level_mentions(text: str) -> str:
    """Corrige, dans le texte destiné à l'affichage, les labels mal orthographiés.

    Sans cela, l'utilisateur voit « URGENCE_MODERIEE » dans l'explication alors
    que le bandeau affiche le niveau extrait : incohérent et peu sérieux.
    """
    if not text:
        return text

    def _fix(match: re.Match) -> str:
        return _normalize_level_token(match.group(0)) or match.group(0)

    return _LEVEL_TOKEN.sub(_fix, text)

# Formulations libres, par niveau. ATTENTION aux pièges historiques :
#   * `\burgent\b` matchait « non-urgent » → sur-classification ;
#   * « pas d'urgence vitale » (négation) était lu comme URGENCE_VITALE.
# Ces cas sont gérés par `is_negated` + la règle « première occurrence gagne »
# (« non urgent » matche CONSULTATION_DIFFEREE dès le mot « non »).
_FREE_TEXT_PATTERNS = {
    "URGENCE_VITALE": [
        r"urgence\s+(?:maximale|vitale|absolue)",
        r"pronostic\s+vital\s+engag",
        r"life.?threatening",
    ],
    "URGENCE_MODEREE": [
        r"urgence\s+mod[ée]r[ée]e",
        r"\burgent\b",
        r"mod[ée]r[ée]e",
    ],
    "CONSULTATION_DIFFEREE": [
        r"consultation\s+diff[ée]r[ée]e",
        r"soins?\s+diff[ée]r[ée]s?",
        r"diff[ée]r[ée]",
        r"non.?urgent",
        r"pas\s+urgent",
        r"deferred",
        r"not\s+urgent",
    ],
}


def extract_triage_level(text: str) -> str | None:
    """Extrait le niveau de triage d'une réponse du modèle.

    Stratégie (de la plus fiable à la moins fiable) :
      1. la ligne structurée « Niveau de priorité : X » du gabarit ;
      2. les clés normalisées (URGENCE_VITALE…) et formulations libres, en
         ignorant les mentions NIÉES (« pas d'urgence vitale ») et en retenant
         la PREMIÈRE occurrence du texte (et non l'ordre des niveaux, qui
         faisait basculer à tort vers URGENCE_VITALE).
    Renvoie `None` si rien n'est trouvé.
    """
    if not text:
        return None

    # 1. Ligne structurée du gabarit d'entraînement (tolérante aux fautes
    #    proches ; si le token est illisible on retombe sur l'étape 2).
    m = _STRUCTURED_LEVEL.search(text)
    if m:
        level = _normalize_level_token(m.group(1))
        if level:
            return level

    # 2. Clés normalisées + formulations libres : première mention non niée.
    candidates: list[tuple[int, str]] = []
    upper = text.upper()
    for key in TRIAGE_LEVELS:
        for match in re.finditer(key, upper):
            if not is_negated(text, match.start()):
                candidates.append((match.start(), key))
                break
    for key, patterns in _FREE_TEXT_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not is_negated(text, match.start()):
                    candidates.append((match.start(), key))
                    break

    return min(candidates)[1] if candidates else None


class TriageAgent:
    """Orchestrateur de triage. Réutilisable (le backend est chargé une fois)."""

    def __init__(self, backend: InferenceBackend | None = None):
        self.backend = backend or get_backend()

    def assess(
        self,
        message: str,
        history: list[dict] | None = None,
        asked_dimensions: list[str] | None = None,
        allow_followup: bool = True,
    ) -> dict:
        """Évalue un message patient et renvoie un résultat de triage structuré.

        `history` : tours précédents (questions de relance déjà posées + réponses).
        `asked_dimensions` : dimensions déjà demandées (évite de boucler).
        `allow_followup` : si False, on tranche directement (utile pour l'éval).
        """
        start = time.perf_counter()
        interaction_id = audit.new_interaction_id()
        history = history or []
        asked_dimensions = asked_dimensions or []
        lang = detect_language(message)

        # On reconstitue le contexte PATIENT : tours utilisateur uniquement.
        # Les questions de l'ASSISTANT sont exclues : elles peuvent citer des
        # symptômes graves (« essoufflement », « fièvre »…) qui déclencheraient
        # à tort le filet de sécurité (red flags) et forceraient URGENCE_VITALE
        # sur un cas bénin. Elles pollueraient aussi le prompt du modèle, qui a
        # été entraîné sur un unique message patient.
        full_context = "\n".join(
            [h.get("content", "") for h in history if h.get("role") != "assistant"]
            + [message]
        )

        # --- 1. Questionnaire adaptatif : faut-il d'abord poser une question ? ---
        if allow_followup:
            needs_followup, questions, dims, lang = build_followup(full_context, asked_dimensions)
            if needs_followup:
                result = {
                    "interaction_id": interaction_id,
                    "type": "follow_up",
                    "needs_follow_up": True,
                    "follow_up_questions": questions,
                    # Cumul des dimensions déjà posées : le client n'a qu'à
                    # renvoyer cette liste telle quelle au tour suivant.
                    "asked_dimensions": asked_dimensions + dims,
                    "lang": lang,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                }
                audit.log_interaction({
                    "interaction_id": interaction_id, "type": "follow_up",
                    "message": message, "questions": questions, "lang": lang,
                })
                return result

        # --- 2. Filet de sécurité : signes d'alerte vitaux ---
        red_flags = detect_red_flags(full_context)

        # --- 3. Appel du modèle de langage ---
        messages = build_messages(full_context)
        try:
            model_text = self.backend.generate(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error("Échec de génération du modèle : %s", exc)
            model_text = ""

        # Labels approchés (« URGENCE_MODERIEE ») rapprochés du canonique AVANT
        # extraction et affichage : bandeau et explication restent cohérents.
        model_text = normalize_level_mentions(model_text)
        model_level = extract_triage_level(model_text)

        # --- 4. Décision finale (prudence : on prend le plus grave) ---
        safety_override = False
        if red_flags:
            # Un signe vital force au minimum l'urgence vitale.
            level = "URGENCE_VITALE"
            if model_level and model_level != "URGENCE_VITALE":
                safety_override = True
        elif model_level:
            level = model_level
        else:
            # Le modèle n'a pas tranché : par prudence, on n'envoie pas en
            # « différé » mais en « urgence modérée » et on signale la faible
            # confiance pour appeler une revue humaine.
            level = "URGENCE_MODEREE"

        info = TRIAGE_LEVELS[level]
        confidence = 0.9 if model_level == level and not safety_override else 0.5

        # Explication : on privilégie le texte du modèle ; on le complète d'un
        # avertissement si le filet de sécurité a relevé un signe d'alerte.
        explanation = model_text or "(Aucune explication générée par le modèle.)"
        if red_flags:
            explanation = (
                f"⚠️ Signe(s) d'alerte détecté(s) : {', '.join(red_flags)}. "
                "Par mesure de sécurité, la priorité est portée au niveau maximal.\n\n"
                + explanation
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        result = {
            "interaction_id": interaction_id,
            "type": "triage",
            "triage_level": level,
            "code": info["code"],
            "label": info["label_fr"] if lang == "fr" else info["label_en"],
            "delai": info["delai"],
            "couleur": info["couleur"],
            "explanation": explanation,
            "red_flags": red_flags,
            "safety_override": safety_override,
            "confidence": confidence,
            "needs_follow_up": False,
            "lang": lang,
            "latency_ms": latency_ms,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

        # --- 5. Traçabilité ---
        audit.log_interaction({
            "interaction_id": interaction_id,
            "type": "triage",
            "message": message,
            "triage_level": level,
            "red_flags": red_flags,
            "safety_override": safety_override,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "lang": lang,
        })
        return result

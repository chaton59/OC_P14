"""Tests de non-régression — audit du 10 juillet 2026.

Chaque test verrouille un bug corrigé lors de cet audit :
  1. le filet de sécurité se déclenchait sur les questions de l'ASSISTANT
     (« essoufflement » dans la question des symptômes associés) → cas bénins
     forcés en URGENCE_VITALE ;
  2. « légère » n'était pas reconnu comme intensité (accent grave) → relance
     inutile ;
  3. `critical_undertriage` comptait les prédictions ABSENTES (incohérent avec
     `dangerous_undertriage`) ;
  4. le split SFT répartissait les vignettes d'une même présentation entre
     train et test (quasi-doublons) et le split DPO laissait dans le train des
     prompts identiques à ceux de l'évaluation (contamination) ;
  5. `/audit?limit=0` renvoyait TOUT le journal ; `/ui` (page de démo) doit
     être servie en HTML.
"""

from fastapi.testclient import TestClient

import triage.data.split as split_module
import triage.serve.api as api_module
from triage.eval.safety import safety_report
from triage.serve.inference import InferenceBackend
from triage.serve.questionnaire import missing_dimensions
from triage.serve.triage_agent import TriageAgent
from triage.utils.common import write_jsonl


class _CannedBackend(InferenceBackend):
    """Backend factice : renvoie toujours le même texte (pas de modèle chargé)."""

    def __init__(self, text: str):
        self.text = text

    def generate(self, messages, **kwargs):
        return self.text


def _agent(tmp_path, monkeypatch, text) -> TriageAgent:
    """Agent réel (orchestration complète) avec backend simulé + audit en tmp."""
    from triage.config import settings

    monkeypatch.setattr(settings, "audit_log_path", str(tmp_path / "audit.log"))
    return TriageAgent(backend=_CannedBackend(text))


# ---------------------------------------------------------------------------
# 1. Le filet de sécurité ne doit PAS lire les questions de l'assistant
# ---------------------------------------------------------------------------
def test_question_assistant_ne_declenche_pas_red_flag(tmp_path, monkeypatch):
    """La question « …essoufflement… » posée PAR L'AGENT ne doit pas forcer
    URGENCE_VITALE sur un cas bénin (bug : sur-triage via l'historique)."""
    agent = _agent(
        tmp_path, monkeypatch,
        "Niveau de priorité : CONSULTATION_DIFFEREE (Soins différés)\n"
        "Analyse : douleur bénigne localisée au doigt.",
    )
    result = agent.assess(
        "Non, rien d'autre",
        history=[
            {"role": "user", "content": "Douleur au doigt depuis hier, légère"},
            {"role": "assistant",
             "content": "Avez-vous d'autres symptômes associés "
                        "(fièvre, essoufflement, vomissements…) ?"},
        ],
        asked_dimensions=["associated"],
    )
    assert result["type"] == "triage"
    assert result["red_flags"] == []
    assert result["safety_override"] is False
    assert result["triage_level"] == "CONSULTATION_DIFFEREE"


def test_red_flag_du_patient_toujours_detecte(tmp_path, monkeypatch):
    """A contrario, un signe d'alerte dans un tour UTILISATEUR de l'historique
    doit toujours être pris en compte."""
    agent = _agent(tmp_path, monkeypatch, "Niveau de priorité : URGENCE_MODEREE")
    result = agent.assess(
        "Depuis une heure environ",
        history=[{"role": "user", "content": "J'ai une douleur dans la poitrine"}],
        asked_dimensions=["onset", "severity"],
    )
    assert "Douleur thoracique" in result["red_flags"]
    assert result["triage_level"] == "URGENCE_VITALE"


def test_followup_renvoie_les_dimensions(tmp_path, monkeypatch):
    """La réponse de relance doit inclure `asked_dimensions` cumulées : sans
    elles, un client ne peut pas les réinjecter au tour suivant."""
    agent = _agent(tmp_path, monkeypatch, "irrelevant")
    result = agent.assess("j'ai mal")
    assert result["type"] == "follow_up"
    assert result["asked_dimensions"], "les dimensions posées doivent être renvoyées"


# ---------------------------------------------------------------------------
# 2. « légère » compte comme une intensité
# ---------------------------------------------------------------------------
def test_legere_reconnu_comme_intensite():
    missing = missing_dimensions("douleur légère au doigt depuis hier")
    assert "severity" not in missing


# ---------------------------------------------------------------------------
# 3. Cohérence des métriques de sécurité quand la prédiction est absente
# ---------------------------------------------------------------------------
def test_prediction_absente_est_unclassified_pas_critique():
    report = safety_report(
        [{"predicted_level": None, "expected_level": "URGENCE_VITALE", "text": ""}]
    )
    assert report["unclassified"] == 1
    assert report["critical_undertriage"] == 0
    assert report["dangerous_undertriage"] == 0
    # Le sous-triage critique reste bien compté quand la prédiction existe.
    report2 = safety_report(
        [{"predicted_level": "URGENCE_MODEREE", "expected_level": "URGENCE_VITALE",
          "text": ""}]
    )
    assert report2["critical_undertriage"] == 1
    assert report2["dangerous_undertriage"] == 1


# ---------------------------------------------------------------------------
# 4. Splits sans fuite : groupes de présentations (SFT) et prompts (DPO)
# ---------------------------------------------------------------------------
def _fake_sft(key: str, level: str, i: int, lang: str) -> dict:
    return {
        "id": f"sft-{key}-{i}",
        "instruction": f"Patient {key} vignette {i} ({lang})",
        "input": "",
        "output": f"Niveau de priorité : {level}",
        "lang": lang,
        "task_type": "triage",
        "metadata": {"presentation_key": key, "triage_level": level},
    }


def test_splits_sans_fuite(tmp_path, monkeypatch):
    levels = ["URGENCE_VITALE", "URGENCE_MODEREE", "CONSULTATION_DIFFEREE"]
    interim = tmp_path / "interim"
    for name in ("interim", "sft", "dpo", "eval"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(split_module, "INTERIM_DIR", interim)
    monkeypatch.setattr(split_module, "SFT_DIR", tmp_path / "sft")
    monkeypatch.setattr(split_module, "DPO_DIR", tmp_path / "dpo")
    monkeypatch.setattr(split_module, "EVAL_DIR", tmp_path / "eval")

    # 18 présentations × 6 vignettes (bilingues), comme le vrai dataset.
    sft = [
        _fake_sft(f"pres-{p:02d}", levels[p % 3], i, "fr" if i % 2 else "en")
        for p in range(18)
        for i in range(6)
    ]
    write_jsonl(sft, interim / "sft.jsonl")
    # Paires DPO : 3 variantes par présentation, prompt = instruction SFT.
    dpo = [
        {
            "id": f"dpo-{p:02d}-{v}",
            "prompt": f"Patient pres-{p:02d} vignette 0 (en)",
            "chosen": "bonne réponse",
            "rejected": f"mauvaise réponse {v}",
            "metadata": {"presentation_key": f"pres-{p:02d}"},
        }
        for p in range(18)
        for v in range(3)
    ]
    write_jsonl(dpo, interim / "dpo.jsonl")

    test_keys, test_prompts = split_module.split_sft(0.10, 0.10, seed=42)
    split_module.split_dpo(0.10, 42, test_keys, test_prompts)

    def _keys(path):
        import json

        with open(path, encoding="utf-8") as f:
            return {json.loads(line)["metadata"]["presentation_key"] for line in f}

    train_k = _keys(tmp_path / "sft" / "train.jsonl")
    val_k = _keys(tmp_path / "sft" / "val.jsonl")
    test_k = _keys(tmp_path / "sft" / "test.jsonl")
    # Une présentation vit dans UNE seule partition (pas de quasi-doublons).
    assert not (train_k & test_k) and not (train_k & val_k) and not (val_k & test_k)
    # Les 3 niveaux restent représentés dans le test (→ éval clinique complète).
    import json

    with open(tmp_path / "sft" / "test.jsonl", encoding="utf-8") as f:
        test_levels = {json.loads(line)["metadata"]["triage_level"] for line in f}
    assert test_levels == set(levels)

    # DPO : aucune paire du train ne porte sur une présentation/un prompt du test.
    dpo_train_k = _keys(tmp_path / "dpo" / "train.jsonl")
    assert not (dpo_train_k & test_k)
    with open(tmp_path / "dpo" / "train.jsonl", encoding="utf-8") as f:
        dpo_train_prompts = {json.loads(line)["prompt"] for line in f}
    assert not (dpo_train_prompts & test_prompts)
    # Val DPO : prompts disjoints du train (split groupé par prompt).
    with open(tmp_path / "dpo" / "val.jsonl", encoding="utf-8") as f:
        dpo_val_prompts = {json.loads(line)["prompt"] for line in f}
    assert not (dpo_train_prompts & dpo_val_prompts)


# ---------------------------------------------------------------------------
# 5. Endpoints : /ui servie en HTML, /audit borné
# ---------------------------------------------------------------------------
def test_ui_page_servie():
    client = TestClient(api_module.app)
    r = client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "CHSA" in r.text


def test_audit_limit_borne():
    client = TestClient(api_module.app)
    assert client.get("/audit?limit=0").status_code == 422
    assert client.get("/audit?limit=5000").status_code == 422
    assert client.get("/audit?limit=10").status_code == 200

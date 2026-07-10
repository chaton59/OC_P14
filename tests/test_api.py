"""Tests de l'API FastAPI, sans charger le vrai modèle (agent simulé).

On injecte un agent factice (stub) pour tester le contrat de l'API (codes HTTP,
schéma des réponses) sans dépendre du GPU ni des poids du modèle.
"""

from fastapi.testclient import TestClient

import triage.serve.api as api_module


class _FakeAgent:
    """Agent simulé : renvoie un verdict fixe, sans appeler de modèle."""

    def assess(self, message, history=None, asked_dimensions=None, allow_followup=True):
        return {
            "interaction_id": "test-id",
            "type": "triage",
            "triage_level": "URGENCE_MODEREE",
            "label": "Urgence modérée",
            "explanation": "Réponse simulée pour le test.",
            "latency_ms": 1.0,
        }


def _client(monkeypatch):
    """Construit un TestClient avec l'agent factice injecté."""
    monkeypatch.setattr(api_module, "_AGENT", _FakeAgent())
    return TestClient(api_module.app)


def test_health(monkeypatch):
    r = _client(monkeypatch).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_liste_niveaux(monkeypatch):
    r = _client(monkeypatch).get("/")
    assert r.status_code == 200
    assert "URGENCE_VITALE" in r.json()["niveaux_de_triage"]


def test_triage_endpoint(monkeypatch):
    r = _client(monkeypatch).post("/triage", json={"message": "j'ai mal au ventre depuis hier"})
    assert r.status_code == 200
    body = r.json()
    assert body["triage_level"] == "URGENCE_MODEREE"
    assert "explanation" in body


def test_triage_message_manquant(monkeypatch):
    """Une requête sans `message` doit être rejetée (422)."""
    r = _client(monkeypatch).post("/triage", json={})
    assert r.status_code == 422


def test_ui_page_servie(monkeypatch):
    r = _client(monkeypatch).get("/ui")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Agent de Triage" in r.text


def test_ui_history_page_servie(monkeypatch):
    """La vue historique est servie et consomme bien /audit en relatif."""
    r = _client(monkeypatch).get("/ui/history")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "../audit" in r.text

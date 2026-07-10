"""API REST FastAPI — interface de démonstration de l'agent de triage.

Endpoints :
  * `GET  /health`  — sonde de vivacité (pour Docker / CI / load balancer) ;
  * `GET  /`        — informations sur le service ;
  * `GET  /ui`      — page web de démonstration (chat, sans dépendance) ;
  * `POST /triage`  — évalue un message patient (one-shot) ;
  * `POST /chat`    — variante conversationnelle (gère le questionnaire adaptatif) ;
  * `GET  /audit`   — relit les dernières interactions journalisées.

Le modèle n'est chargé qu'au PREMIER appel nécessitant une inférence (chargement
paresseux) : importer ce module reste donc léger (utile pour les tests/CI).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from triage import __version__
from triage.config import TRIAGE_LEVELS, settings
from triage.serve import audit
from triage.utils.common import get_logger

logger = get_logger("serve.api")

app = FastAPI(
    title="CHSA — Agent de Triage Médical (POC)",
    description="Assistant de triage aux urgences (Qwen3-1.7B, SFT+DPO). "
                "⚠️ Outil de démonstration : ne remplace pas un avis médical.",
    version=__version__,
)

# Agent chargé paresseusement (le modèle n'est instancié qu'au besoin).
_AGENT = None


def get_agent():
    """Renvoie l'agent de triage, en l'instanciant au premier appel."""
    global _AGENT
    if _AGENT is None:
        from triage.serve.triage_agent import TriageAgent

        _AGENT = TriageAgent()
    return _AGENT


# ---------------------------------------------------------------------------
# Schémas d'entrée / sortie (validés par Pydantic)
# ---------------------------------------------------------------------------
class TriageRequest(BaseModel):
    """Requête de triage simple."""

    message: str = Field(..., description="Description des symptômes par le patient.",
                         examples=["J'ai une douleur thoracique et du mal à respirer."])
    allow_followup: bool = Field(True, description="Autoriser les questions de relance.")


class ChatTurn(BaseModel):
    """Un tour de conversation (pour le mode adaptatif)."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Requête conversationnelle : message + historique + dimensions déjà demandées."""

    message: str
    history: list[ChatTurn] = Field(default_factory=list)
    asked_dimensions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Sonde de vivacité : renvoie l'état du service (sans charger le modèle)."""
    return {"status": "ok", "version": __version__, "backend": settings.inference_backend}


@app.get("/")
def root() -> dict:
    """Informations générales sur le service et les niveaux de triage."""
    return {
        "service": "CHSA — Agent de Triage Médical (POC)",
        "version": __version__,
        "niveaux_de_triage": {
            k: {"label": v["label_fr"], "delai": v["delai"], "couleur": v["couleur"]}
            for k, v in TRIAGE_LEVELS.items()
        },
        "avertissement": "Outil de démonstration. Ne remplace pas un avis médical. "
                         "En cas d'urgence, appelez le 15 (SAMU).",
    }


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    """Évalue un message patient et renvoie un verdict de triage structuré."""
    logger.info("Requête /triage reçue (%d caractères).", len(req.message))
    return get_agent().assess(req.message, allow_followup=req.allow_followup)


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Variante conversationnelle gérant le questionnaire adaptatif.

    Le client renvoie l'historique et les dimensions déjà demandées ; l'agent
    pose éventuellement une nouvelle question, ou produit le verdict final.
    """
    history = [t.model_dump() for t in req.history]
    return get_agent().assess(
        req.message, history=history, asked_dimensions=req.asked_dimensions
    )


@app.get("/ui")
def ui() -> FileResponse:
    """Page web de démonstration : un chat minimaliste qui consomme `/chat`.

    Servie par la même application (même origine) → aucun besoin de CORS.
    Fichier statique unique, sans framework : voir `serve/static/index.html`.
    """
    return FileResponse(
        Path(__file__).parent / "static" / "index.html", media_type="text/html"
    )


@app.get("/ui/history")
def ui_history() -> FileResponse:
    """Vue web du journal d'audit : une interaction par ligne de tableau.

    Même approche que `/ui` : fichier statique unique sans dépendance, qui
    consomme `GET /audit` (chemin relatif) et met en forme le JSON.
    """
    return FileResponse(
        Path(__file__).parent / "static" / "history.html", media_type="text/html"
    )


@app.get("/audit")
def get_audit(limit: int = Query(50, ge=1, le=1000)) -> dict:
    """Relit les dernières interactions journalisées (traçabilité).

    `limit` est borné (1-1000) : `limit=0` renvoyait TOUT le journal et une
    valeur négative décalait la fenêtre — bornes explicites, réponse 422 sinon.
    """
    return {"interactions": audit.read_audit(limit=limit)}


def main() -> None:
    """Point d'entrée CLI : lance le serveur uvicorn (hôte/port configurables)."""
    import uvicorn

    uvicorn.run(
        "triage.serve.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()

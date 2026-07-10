"""Journal d'audit — traçabilité des interactions (exigence de la mission).

« Garantir la traçabilité de chaque interaction pour les audits médicaux. »

Chaque interaction (message reçu, décision de triage, latence, version du
modèle) est consignée dans un fichier **JSONL append-only** : une ligne par
événement, jamais réécrite. Ce format est simple à archiver, à interroger et à
exporter vers un SIH ou un SIEM.

⚠️ Le message patient est journalisé tel que reçu par l'API. Dans un déploiement
réel avec de vraies données, il faudrait l'anonymiser AVANT écriture (réutiliser
`triage.data.anonymize`) et restreindre l'accès au journal (chiffrement, RBAC).
On le signale explicitement ici pour ne pas créer un faux sentiment de sécurité.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from triage.config import settings
from triage.utils.common import get_logger

logger = get_logger("serve.audit")

# Verrou pour des écritures concurrentes sûres (l'API peut être multi-thread).
_LOCK = threading.Lock()


def new_interaction_id() -> str:
    """Identifiant unique d'interaction (traçable de bout en bout)."""
    return str(uuid.uuid4())


def log_interaction(record: dict) -> None:
    """Ajoute un enregistrement horodaté au journal d'audit (append-only)."""
    path = Path(settings.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.model_path,
        "backend": settings.inference_backend,
        **record,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    logger.debug("Interaction journalisée : %s", record.get("interaction_id"))


def read_audit(limit: int = 100) -> list[dict]:
    """Relit les dernières interactions du journal (pour l'endpoint /audit)."""
    path = Path(settings.audit_log_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # Une ligne corrompue (écriture interrompue…) ne doit pas rendre
            # tout l'endpoint /audit indisponible : on l'ignore en le signalant.
            logger.warning("Ligne d'audit corrompue ignorée : %.80s", line)
    return out

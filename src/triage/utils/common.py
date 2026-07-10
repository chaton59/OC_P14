"""Petites fonctions utilitaires réutilisées dans tout le projet.

On regroupe ici ce qui n'a pas de logique métier propre :
  * configuration d'un logger lisible ;
  * fixation des graines aléatoires (reproductibilité) ;
  * lecture/écriture de fichiers JSONL (le format standard de nos datasets).

Le format **JSONL** (JSON Lines) = un objet JSON par ligne. C'est le format
de prédilection en NLP : il se lit en streaming (sans tout charger en mémoire),
se versionne bien et est nativement compris par Hugging Face Datasets.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Journalisation
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Renvoie un logger configuré avec un format horodaté lisible.

    On évite les `print()` au profit du module `logging` : on peut ainsi
    filtrer par niveau (INFO/DEBUG/ERROR) et rediriger vers un fichier.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # évite d'ajouter plusieurs fois le même handler
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléa pour des résultats reproductibles.

    Indispensable en santé : un audit doit pouvoir rejouer un entraînement et
    obtenir (quasi) les mêmes poids. On fixe Python, NumPy et PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # NumPy peut être absent dans un contexte minimal
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Entrées / sorties JSONL
# ---------------------------------------------------------------------------
def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Écrit une suite d'objets dans un fichier JSONL. Renvoie le nombre écrit.

    `ensure_ascii=False` conserve les accents français en clair dans le fichier.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lit un fichier JSONL ligne par ligne (générateur, économe en mémoire)."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # on ignore les lignes vides éventuelles
                yield json.loads(line)

"""Briques communes aux entraînements SFT et DPO.

On factorise ici tout ce qui est partagé entre les deux étapes :
  * lecture d'une configuration YAML ;
  * chargement du tokenizer (avec installation du gabarit ChatML) ;
  * chargement du modèle de base, éventuellement quantifié en 4 bits (QLoRA) ;
  * construction de la configuration LoRA.

Centraliser évite la duplication et garantit que SFT et DPO utilisent
EXACTEMENT le même format de prompt et le même modèle de base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from triage.prompts import CHATML_TEMPLATE, IM_END, IM_START
from triage.utils.common import get_logger

logger = get_logger("train.common")


def load_config(path: str | Path) -> dict[str, Any]:
    """Charge un fichier de configuration YAML en dictionnaire Python."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration chargée : %s", path)
    return cfg


def get_tokenizer(base_model: str):
    """Charge le tokenizer et s'assure qu'il sait gérer notre format ChatML.

    Trois points d'attention :
      1. installer le gabarit ChatML (le modèle « Base » n'en a pas forcément) ;
      2. garantir la présence des marqueurs <|im_start|> / <|im_end|> ;
      3. définir un `pad_token` (obligatoire pour grouper les exemples en batch).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # 1+2. Marqueurs ChatML : on les ajoute s'ils manquent (rare pour Qwen).
    special = [t for t in (IM_START, IM_END) if t not in tokenizer.get_vocab()]
    if special:
        logger.info("Ajout des marqueurs ChatML manquants : %s", special)
        tokenizer.add_special_tokens({"additional_special_tokens": special})

    # 3. Gabarit + token de padding.
    tokenizer.chat_template = CHATML_TEMPLATE
    if tokenizer.pad_token is None or tokenizer.pad_token == IM_END:
        # ⚠️ BUG HISTORIQUE : pad = <|im_end|>. Les collators masquent les tokens
        # de padding de la loss (label = -100) ; avec ce choix, TOUS les <|im_end|>
        # des textes d'entraînement étaient masqués → le modèle n'apprenait JAMAIS
        # à émettre la fin de tour → générations sans fin et répétitions en prod.
        # On prend un token qui n'apparaît jamais dans nos textes ChatML :
        # l'eos natif de Qwen (<|endoftext|>), sinon un vrai token <|pad|> dédié.
        if tokenizer.eos_token and tokenizer.eos_token != IM_END:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            logger.info("Token de padding dédié ajouté : <|pad|>")
    tokenizer.padding_side = "right"  # standard pour l'entraînement causal
    return tokenizer


def get_bnb_config(load_in_4bit: bool):
    """Construit la configuration de quantification 4 bits (QLoRA) si demandée.

    NF4 + double quantification + calcul en bfloat16 = le réglage de référence
    pour entraîner de gros modèles sur peu de VRAM avec une perte minime.
    """
    if not load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(base_model: str, cfg: dict[str, Any], tokenizer):
    """Charge le modèle de base (causal LM), quantifié ou non.

    Si le tokenizer a gagné de nouveaux tokens (marqueurs ChatML), on
    redimensionne la couche d'embeddings en conséquence.
    """
    import torch
    from transformers import AutoModelForCausalLM

    bnb_config = get_bnb_config(cfg.get("load_in_4bit", False))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        device_map={"": 0},          # tout sur le GPU 0 (machine mono-GPU)
        trust_remote_code=True,
    )

    # Agrandit la couche d'embeddings UNIQUEMENT si le tokenizer a plus de tokens
    # que le modèle (cas où l'on a ajouté des marqueurs). On ne rétrécit jamais :
    # le vocabulaire Qwen est volontairement « rembourré » (padded) et le réduire
    # supprimerait des lignes d'embedding utiles.
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Désactive le cache K/V pendant l'entraînement (incompatible avec le
    # gradient checkpointing, et inutile hors génération).
    model.config.use_cache = False
    return model


def get_lora_config(cfg: dict[str, Any]):
    """Construit la configuration LoRA à partir du YAML."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

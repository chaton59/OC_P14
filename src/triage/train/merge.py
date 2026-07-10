"""Fusion des adaptateurs LoRA dans le modèle de base → modèle déployable.

Pour servir le modèle efficacement (notamment avec **vLLM**), il est plus simple
de disposer d'un modèle « complet » dont les poids LoRA ont été FONDUS dans les
poids de base, plutôt que de charger des adaptateurs séparés à l'exécution.

Ce script reconstruit le modèle final :
    base  →  + adaptateur SFT (fusion)  →  + adaptateur DPO (fusion)  →  sauvegarde

La fusion se fait en pleine précision (bfloat16), JAMAIS en 4 bits (la fusion
d'un LoRA dans des poids quantifiés n'est pas fiable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from triage.config import MODELS_DIR
from triage.train.common import get_tokenizer
from triage.utils.common import get_logger

logger = get_logger("train.merge")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion des adaptateurs LoRA.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--sft-adapter", default="models/sft-lora")
    parser.add_argument("--dpo-adapter", default="models/dpo-lora",
                        help="Adaptateur DPO à fusionner (mettre '' pour l'ignorer).")
    parser.add_argument("--output", default=str(MODELS_DIR / "qwen3-triage-merged"))
    args = parser.parse_args()

    # GARDE-FOU : si un chemin d'adaptateur est DEMANDÉ mais absent, on refuse
    # de continuer. Ignorer silencieusement l'adaptateur (comportement
    # historique) a déjà produit un modèle « fusionné » SANS l'alignement DPO,
    # sans que rien ne le signale (l'échec de `make train-dpo` passait
    # inaperçu). Pour fusionner volontairement sans DPO : `--dpo-adapter ''`.
    for name, path in (("SFT", args.sft_adapter), ("DPO", args.dpo_adapter)):
        if path and not Path(path).exists():
            raise SystemExit(
                f"Adaptateur {name} introuvable : {path}\n"
                f"→ l'entraînement correspondant a-t-il réussi ? "
                f"(relancer `make train-{name.lower()}` et vérifier les erreurs)\n"
                f"→ pour fusionner volontairement sans cet adaptateur : "
                f"`uv run python -m triage.train.merge --{name.lower()}-adapter ''`"
            )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(args.base_model)

    logger.info("Chargement du modèle de base en bfloat16…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # --- Fusion de l'adaptateur SFT ---
    if args.sft_adapter:
        logger.info("Fusion de l'adaptateur SFT : %s", args.sft_adapter)
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
    else:
        logger.warning("Fusion SANS adaptateur SFT (demandé explicitement).")

    # --- Fusion de l'adaptateur DPO (par-dessus le SFT) ---
    if args.dpo_adapter:
        logger.info("Fusion de l'adaptateur DPO : %s", args.dpo_adapter)
        model = PeftModel.from_pretrained(model, args.dpo_adapter)
        model = model.merge_and_unload()
    else:
        logger.warning("Fusion SANS adaptateur DPO (demandé explicitement).")

    # --- Sauvegarde du modèle complet + tokenizer ---
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True  # on réactive le cache pour l'inférence
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    logger.info("Modèle fusionné sauvegardé dans : %s", args.output)


if __name__ == "__main__":
    main()

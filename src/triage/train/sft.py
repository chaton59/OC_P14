"""Semaine 2 — Fine-Tuning Supervisé (SFT) du modèle Qwen3-1.7B avec LoRA.

OBJECTIF
--------
Spécialiser le modèle de base sur notre corpus médical : lui apprendre à
répondre comme un assistant de triage, dans le bon format et les deux langues.

MÉTHODE
-------
* Chaque exemple « instruction → réponse » est mis au format ChatML, puis le
  modèle apprend à reproduire la réponse de l'assistant (apprentissage supervisé).
* On n'entraîne pas tout le modèle : **LoRA** insère de petites matrices
  entraînables (~1 % des paramètres). Combiné à la **quantification 4 bits**
  (QLoRA), cela permet de tenir dans ~12 Go de VRAM.

SORTIE
------
Un adaptateur LoRA + le tokenizer, sauvegardés dans `output_dir`. Les courbes
d'entraînement sont consultables avec `tensorboard --logdir models/sft-lora`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from triage.prompts import compose_clinical_input, render_chatml
from triage.train.common import get_lora_config, get_tokenizer, load_config, load_model
from triage.utils.common import get_logger, read_jsonl, set_seed

logger = get_logger("train.sft")


def build_text_dataset(jsonl_path: str, max_samples: int | None):
    """Charge un JSONL SFT et le transforme en `Dataset` avec une colonne `text`.

    La colonne `text` contient la conversation ChatML COMPLÈTE (système +
    utilisateur + réponse). Pour un POC, on entraîne sur l'ensemble de la
    séquence ; un raffinement classique serait de ne calculer la perte que sur
    la réponse de l'assistant (`completion_only`).
    """
    from datasets import Dataset

    rows = list(read_jsonl(jsonl_path))
    if max_samples:
        rows = rows[:max_samples]

    texts = []
    for r in rows:
        user_msg = compose_clinical_input(r["instruction"], r.get("input", ""))
        texts.append({"text": render_chatml(user_msg, r["output"])})
    logger.info("Dataset '%s' : %d exemples", Path(jsonl_path).name, len(texts))
    return Dataset.from_list(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning supervisé (SFT + LoRA).")
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    from trl import SFTConfig, SFTTrainer

    # --- 1. Tokenizer + modèle (quantifié 4 bits) ---
    tokenizer = get_tokenizer(cfg["base_model"])
    model = load_model(cfg["base_model"], cfg, tokenizer)
    lora_config = get_lora_config(cfg)

    # --- 2. Données ---
    train_ds = build_text_dataset(cfg["train_file"], cfg.get("max_train_samples"))
    eval_ds = build_text_dataset(cfg["eval_file"], cfg.get("max_eval_samples"))

    # --- 3. Configuration d'entraînement TRL ---
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        seed=cfg.get("seed", 42),
        # Données / séquences
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        packing=False,                       # un exemple par séquence (plus simple)
        # Boucle d'entraînement
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # `use_reentrant=False` : variante recommandée du gradient checkpointing.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Journalisation / évaluation / sauvegardes
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg.get("report_to", "tensorboard"),
        # `max_steps` (optionnel) plafonne le nombre de pas : pratique pour un
        # test de fumée rapide. -1 = illimité (on suit alors num_train_epochs).
        max_steps=cfg.get("max_steps", -1),
    )

    # --- 4. Entraîneur (le peft_config déclenche l'application de LoRA) ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Démarrage de l'entraînement SFT…")
    trainer.train()

    # --- 5. Sauvegarde de l'adaptateur LoRA + tokenizer ---
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logger.info("Adaptateur SFT sauvegardé dans : %s", cfg["output_dir"])


if __name__ == "__main__":
    main()

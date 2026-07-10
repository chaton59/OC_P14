"""Semaine 3 — Alignement par préférences (DPO) du modèle déjà fine-tuné.

OBJECTIF
--------
Après le SFT, le modèle « sait » répondre. Le DPO (Direct Preference
Optimization) lui apprend à *préférer* la bonne réponse à la moins bonne, à
partir de paires (chosen / rejected). Pour le triage, on l'aligne notamment vers
la **prudence** : préférer une évaluation prudente et structurée à une réponse
qui sous-estime la gravité.

POURQUOI DPO PLUTÔT QUE RLHF CLASSIQUE ?
----------------------------------------
DPO se passe d'un *modèle de récompense* séparé et d'apprentissage par
renforcement : il optimise directement une fonction de perte sur les paires de
préférences. C'est plus simple, plus stable et moins gourmand — idéal pour un POC.

DÉMARRAGE À PARTIR DU SFT
-------------------------
On part du modèle de base + l'adaptateur LoRA issu du SFT. Pour repartir d'une
base « propre » à entraîner, on **fusionne** l'adaptateur SFT dans les poids,
puis on entraîne un NOUVEL adaptateur LoRA par-dessus, dédié à l'alignement.
"""

from __future__ import annotations

import argparse

from triage.prompts import IM_END, render_chatml
from triage.train.common import get_lora_config, get_tokenizer, load_config, load_model
from triage.utils.common import get_logger, read_jsonl, set_seed

logger = get_logger("train.dpo")


def build_preference_dataset(jsonl_path: str, max_samples: int | None):
    """Charge un JSONL DPO et le met au format attendu par `DPOTrainer`.

    `DPOTrainer` attend trois colonnes : `prompt`, `chosen`, `rejected`.
      * `prompt`  = la conversation ChatML jusqu'à l'ouverture de la réponse ;
      * `chosen`  = la bonne réponse (texte) ;
      * `rejected`= la mauvaise réponse (texte).
    Le format du prompt est IDENTIQUE à celui du SFT (cohérence indispensable).
    """
    from datasets import Dataset

    rows = list(read_jsonl(jsonl_path))
    if max_samples:
        rows = rows[:max_samples]

    data = []
    for r in rows:
        # `render_chatml` sans réponse → prompt se terminant par l'ouverture
        # de l'assistant, exactement comme à l'inférence.
        prompt = render_chatml(r["prompt"], assistant_content=None)
        data.append(
            {
                "prompt": prompt,
                # Les complétions se terminent par `<|im_end|>`, comme les
                # cibles SFT : le DPO renforce ainsi le MÊME marqueur de fin de
                # tour (sinon il n'apprenait la fin que via l'eos ajouté par TRL).
                "chosen": r["chosen"] + IM_END,
                "rejected": r["rejected"] + IM_END,
            }
        )
    logger.info("Dataset DPO '%s' : %d paires", jsonl_path, len(data))
    return Dataset.from_list(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alignement par préférences (DPO).")
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    import torch
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    # --- 1. Tokenizer (on réutilise celui sauvegardé par le SFT) ---
    sft_adapter = cfg["sft_adapter"]
    tokenizer = get_tokenizer(cfg["base_model"])

    # --- 2. Modèle de base + fusion de l'adaptateur SFT ---
    model = load_model(cfg["base_model"], cfg, tokenizer)
    try:
        logger.info("Chargement et fusion de l'adaptateur SFT : %s", sft_adapter)
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()  # fond le LoRA SFT dans les poids
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Impossible de charger l'adaptateur SFT (%s). On part du modèle de base.",
            exc,
        )
    model.config.use_cache = False

    # --- 3. Nouvel adaptateur LoRA pour l'alignement ---
    lora_config = get_lora_config(cfg)

    # --- 4. Données préférentielles ---
    train_ds = build_preference_dataset(cfg["train_file"], cfg.get("max_train_samples"))
    eval_ds = build_preference_dataset(cfg["eval_file"], cfg.get("max_eval_samples"))

    # --- 5. Configuration DPO ---
    # `max_prompt_length` (borne du prompt seul) a été RETIRÉ de DPOConfig dans
    # les versions récentes de TRL (>= 1.x) : la troncature n'y est pilotée que
    # par `max_length`. On ne transmet la clé YAML que si la version installée
    # la supporte, sinon on l'ignore en le signalant (au lieu de planter).
    import dataclasses

    dpo_fields = {f.name for f in dataclasses.fields(DPOConfig)}
    extra_config = {}
    if cfg.get("max_prompt_length"):
        if "max_prompt_length" in dpo_fields:
            extra_config["max_prompt_length"] = cfg["max_prompt_length"]
        else:
            logger.warning(
                "Cette version de TRL ne supporte pas `max_prompt_length` "
                "(clé YAML ignorée) : la troncature est bornée par max_length=%s.",
                cfg["max_length"],
            )

    dpo_config = DPOConfig(
        output_dir=cfg["output_dir"],
        seed=cfg.get("seed", 42),
        beta=cfg["beta"],                       # force de l'alignement
        max_length=cfg["max_length"],           # tronque prompt + réponse
        **extra_config,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        # IMPORTANT : la taille de batch d'ÉVALUATION vaut 8 par défaut, ce qui
        # provoque des pics mémoire (OOM) avec de longues séquences. On la fixe
        # à 1 pour évaluer sans saturer les 12 Go de VRAM.
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg.get("report_to", "tensorboard"),
        max_steps=cfg.get("max_steps", -1),
    )

    # --- 6. Entraîneur DPO ---
    # `ref_model=None` : avec un peft_config, TRL utilise le modèle SANS
    # adaptateur comme référence figée → pas besoin de charger un 2e modèle.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Le DPO exige le cache K/V désactivé (compatibilité gradient checkpointing).
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not isinstance(model, torch.nn.Module):  # garde-fou défensif
        raise RuntimeError("Le modèle DPO n'est pas un module PyTorch valide.")

    logger.info("Démarrage de l'entraînement DPO…")
    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logger.info("Adaptateur DPO sauvegardé dans : %s", cfg["output_dir"])


if __name__ == "__main__":
    main()

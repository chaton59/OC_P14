"""Sous-package `train` — fine-tuning supervisé (SFT/LoRA) et alignement (DPO).

  sft.py  →  spécialise Qwen3-1.7B-Base sur notre corpus médical (LoRA)
  dpo.py  →  aligne le modèle SFT sur les préférences cliniques (DPO)

Les deux scripts partagent la même philosophie : configuration par fichier YAML,
journalisation TensorBoard, checkpoints réguliers et graine fixée pour la
reproductibilité.
"""

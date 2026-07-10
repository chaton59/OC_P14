"""Abstraction du backend d'inférence : transformers (local) ou vLLM (production).

On définit une interface commune `InferenceBackend.generate(messages)` et deux
implémentations interchangeables, choisies par configuration :

  * **TransformersBackend** — charge le modèle en mémoire avec 🤗 Transformers.
    Léger à mettre en place, parfait pour les tests locaux et la CI.
  * **VLLMBackend** — appelle un serveur **vLLM** via son API compatible OpenAI.
    C'est la voie de production : vLLM optimise fortement le débit et la latence.

Le reste de l'application (agent, API) ignore quel backend est utilisé : il suffit
de changer la variable d'environnement `TRIAGE_INFERENCE_BACKEND`.
"""

from __future__ import annotations

import re
from pathlib import Path

from triage.config import settings
from triage.prompts import CHATML_TEMPLATE, IM_END, IM_START
from triage.utils.common import get_logger

logger = get_logger("serve.inference")

# Les corpus externes ont été anonymisés par remplacement (« <PERSONNE> »…). Le
# modèle peut donc, rarement, régénérer ces marqueurs. On les retire de la sortie
# affichée à l'utilisateur : ce sont des artefacts d'entraînement, pas du contenu.
_ANON_MARKERS = re.compile(
    r"\s*<(?:PERSONNE|EMAIL|TELEPHONE|NIR|IP|CARTE|LIEU|DATE|DONNEE_MASQUEE)>"
)

# Blocs de « réflexion » (Qwen3 en mode thinking) et tokens spéciaux résiduels :
# des artefacts qui ne doivent jamais atteindre l'utilisateur.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_SPECIAL_TOKENS = re.compile(r"<\|[^|>]*\|>")


def clean_output(text: str) -> str:
    """Nettoie la sortie du modèle avant de la renvoyer à l'utilisateur.

    1. coupe au premier marqueur de fin ou de NOUVEAU tour (si la génération
       « déborde », on ne garde que la première réponse de l'assistant) ;
    2. retire les blocs <think> éventuels (mode réflexion de Qwen3) ;
    3. retire les tokens spéciaux et marqueurs d'anonymisation résiduels.
    """
    text = text.split(IM_END)[0].split(IM_START)[0]
    text = _THINK_BLOCK.sub("", text)
    text = _SPECIAL_TOKENS.sub("", text)
    return _ANON_MARKERS.sub("", text).strip()


class InferenceBackend:
    """Interface commune. `generate` prend une liste de messages ChatML."""

    def generate(self, messages: list[dict], **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError


class TransformersBackend(InferenceBackend):
    """Backend local basé sur 🤗 Transformers (chargement du modèle en mémoire)."""

    def __init__(self, model_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # On privilégie le modèle fine-tuné fusionné ; à défaut, le modèle de base.
        path = model_path or settings.model_path
        if not Path(path).exists():
            logger.warning(
                "Modèle fusionné introuvable (%s). Repli sur le modèle de base '%s' "
                "(réponses NON spécialisées : pour la démo de l'API uniquement).",
                path, settings.base_model,
            )
            path = settings.base_model

        logger.info("Chargement du modèle pour inférence : %s", path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # IMPORTANT : on impose le MÊME gabarit ChatML qu'à l'entraînement.
        # Sans cela, le repli sur le modèle de base utiliserait le gabarit Qwen3
        # d'origine (mode « thinking ») → format jamais vu à l'entraînement,
        # blocs <think> interminables et réponses dégradées.
        self.tokenizer.chat_template = CHATML_TEMPLATE
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()
        # Tokens d'arrêt : fin de tour ChatML + eos natif du modèle. On filtre
        # les identifiants invalides (None / inconnu) pour ne jamais passer une
        # liste corrompue à `generate`.
        candidates = [
            self.tokenizer.convert_tokens_to_ids(IM_END),
            self.tokenizer.eos_token_id,
        ]
        self.eos_ids = [
            t for t in dict.fromkeys(candidates)
            if isinstance(t, int) and t >= 0 and t != self.tokenizer.unk_token_id
        ] or None

    def generate(self, messages: list[dict], **kwargs) -> str:
        import torch

        # Applique le gabarit de conversation et ajoute l'amorce de réponse.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        temperature = kwargs.get("temperature", settings.temperature)
        gen_kwargs: dict = {
            "max_new_tokens": kwargs.get("max_new_tokens", settings.max_new_tokens),
            "eos_token_id": self.eos_ids,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            # Garde-fous anti-boucles : pénalise les tokens déjà émis et interdit
            # de répéter un n-gramme entier (cause des « répétitions de mots »).
            "repetition_penalty": kwargs.get(
                "repetition_penalty", settings.repetition_penalty
            ),
            "no_repeat_ngram_size": kwargs.get(
                "no_repeat_ngram_size", settings.no_repeat_ngram_size
            ),
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs["do_sample"] = False  # décodage glouton (déterministe)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        # On ne décode que les tokens générés (on retire le prompt). On garde les
        # tokens spéciaux pour pouvoir couper proprement au premier <|im_end|> /
        # <|im_start|> (cf. `clean_output`), puis on nettoie.
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return clean_output(text)


class VLLMBackend(InferenceBackend):
    """Backend de production : appelle un serveur vLLM (API compatible OpenAI)."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None):
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model_name = model_name or settings.vllm_model_name

    def generate(self, messages: list[dict], **kwargs) -> str:
        import httpx

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_new_tokens", settings.max_new_tokens),
            "temperature": kwargs.get("temperature", settings.temperature),
            "top_p": 0.9,
            "stop": [IM_END],
            # Paramètre supporté par vLLM (extension de l'API OpenAI) : même
            # garde-fou anti-répétitions que le backend transformers.
            "repetition_penalty": kwargs.get(
                "repetition_penalty", settings.repetition_penalty
            ),
        }
        # Endpoint « chat completions », standard de l'API OpenAI que vLLM imite.
        resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=60.0)
        resp.raise_for_status()
        return clean_output(resp.json()["choices"][0]["message"]["content"])


# Cache du backend (on ne charge le modèle qu'une fois par processus).
_BACKEND: InferenceBackend | None = None


def get_backend() -> InferenceBackend:
    """Renvoie le backend configuré (chargé paresseusement et mis en cache)."""
    global _BACKEND
    if _BACKEND is None:
        if settings.inference_backend == "vllm":
            logger.info("Backend d'inférence : vLLM (%s)", settings.vllm_base_url)
            _BACKEND = VLLMBackend()
        else:
            logger.info("Backend d'inférence : transformers (local)")
            _BACKEND = TransformersBackend()
    return _BACKEND

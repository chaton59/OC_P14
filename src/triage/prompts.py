"""Gabarits de prompt et logique de formatage des conversations.

Ce module définit **comment on parle au modèle**. C'est un point central :
le même format doit être utilisé à l'entraînement ET à l'inférence, sinon le
modèle reçoit à la production des entrées qu'il n'a jamais vues → mauvaises
réponses. On centralise donc tout ici.

On adopte le format **ChatML** (celui de la famille Qwen) :

    <|im_start|>system
    ...consignes système...<|im_end|>
    <|im_start|>user
    ...message du patient/soignant...<|im_end|>
    <|im_start|>assistant
    ...réponse du modèle...<|im_end|>

Le modèle « Qwen3-1.7B-Base » est un modèle *de base* (non instruct) : il ne
possède pas forcément de gabarit de conversation prêt à l'emploi. On lui en
fournit donc un explicitement (cf. `CHATML_TEMPLATE`), appliqué via le
tokenizer Hugging Face.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Consigne système : la "personnalité" et les règles de l'agent de triage.
# ---------------------------------------------------------------------------
# C'est la pièce la plus importante côté métier : elle cadre le comportement
# (rôle, niveaux de priorité, obligation de prudence, format de réponse).
SYSTEM_PROMPT = (
    "Tu es un assistant de triage médical déployé au service des urgences du "
    "Centre Hospitalier Saint-Aurélien (CHSA). Ton rôle est d'aider le personnel "
    "soignant à évaluer la priorité de prise en charge d'un patient à partir de "
    "ses symptômes, antécédents et constantes vitales.\n\n"
    "Tu dois TOUJOURS :\n"
    "1. Poser des questions de clarification si l'information est insuffisante ;\n"
    "2. Classer la situation selon l'un des trois niveaux suivants :\n"
    "   - URGENCE_VITALE : pronostic vital engagé, prise en charge immédiate ;\n"
    "   - URGENCE_MODEREE : à voir rapidement, sans engagement vital immédiat ;\n"
    "   - CONSULTATION_DIFFEREE : situation non urgente, soins différés possibles ;\n"
    "3. Justifier clairement ton évaluation en langage compréhensible ;\n"
    "4. Rester prudent : en cas de doute, surclasser la priorité plutôt que la "
    "sous-estimer, et rappeler que tu ne remplaces pas un médecin.\n\n"
    "Tu ne poses jamais de diagnostic définitif et tu ne prescris jamais de "
    "traitement médicamenteux."
)

# Marqueurs ChatML (réutilisés à l'inférence pour repérer la fin de réponse).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Gabarit Jinja installé sur le tokenizer pour garantir un formatage identique
# partout. Il reproduit le ChatML de Qwen. `add_generation_prompt` ajoute le
# marqueur d'ouverture de la réponse de l'assistant (utile en génération).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_messages(user_content: str, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Construit la liste de messages [système, utilisateur] standard.

    Format attendu par les tokenizers Hugging Face (`apply_chat_template`).
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def render_chatml(
    user_content: str,
    assistant_content: str | None = None,
    system: str = SYSTEM_PROMPT,
) -> str:
    """Rend une conversation complète au format ChatML, sans dépendre du tokenizer.

    Pratique pour fabriquer la colonne « text » des datasets SFT (entraînement)
    et pour des tests unitaires déterministes. Si `assistant_content` est fourni,
    la réponse de l'assistant est incluse (cas SFT) ; sinon on s'arrête au
    marqueur d'ouverture de l'assistant (cas génération / prompt DPO).
    """
    text = (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user_content}{IM_END}\n"
        f"{IM_START}assistant\n"
    )
    if assistant_content is not None:
        text += f"{assistant_content}{IM_END}\n"
    return text


def compose_clinical_input(
    instruction: str,
    input_context: str = "",
) -> str:
    """Assemble l'instruction et le contexte clinique en un message utilisateur.

    Beaucoup de jeux de données distinguent « instruction » et « input »
    (contexte). On les fusionne proprement en un seul message utilisateur.
    """
    instruction = instruction.strip()
    input_context = (input_context or "").strip()
    if input_context:
        return f"{instruction}\n\nContexte :\n{input_context}"
    return instruction

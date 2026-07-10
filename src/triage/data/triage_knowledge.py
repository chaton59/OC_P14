"""Base de connaissances de triage (règles cliniques) et générateur de vignettes.

POURQUOI CE MODULE ?
--------------------
Les corpus publics (MedQuAD, FrenchMedMCQA…) contiennent de la *connaissance
médicale générale*, mais pas d'exemples au format exact que notre agent doit
produire : « symptômes → niveau de priorité + justification ». Or un modèle
n'apprend bien que ce qu'on lui montre. On fabrique donc, à partir d'une table
de règles cliniques validées, des *vignettes de triage* synthétiques et
bilingues qui enseignent au modèle le comportement et le format attendus.

C'est une démarche légitime et courante (« data augmentation » à base de
règles). Ces données sont 100 % synthétiques : aucune donnée patient réelle,
donc aucun risque RGPD à la source. Chaque vignette est tracée comme telle.

ATTENTION : ces règles sont une simplification pédagogique destinée à un POC.
Elles ne constituent pas un protocole de triage opposable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from triage.config import MEDICAL_DISCLAIMER, TRIAGE_LEVELS


@dataclass
class Presentation:
    """Un tableau clinique type, associé à un niveau de priorité de triage."""

    key: str                      # identifiant court
    level: str                    # clé de TRIAGE_LEVELS (URGENCE_VITALE, …)
    symptoms_fr: list[str]        # symptômes en français
    symptoms_en: list[str]        # symptômes en anglais
    rationale_fr: str             # justification clinique (fr)
    rationale_en: str             # justification clinique (en)
    red_flags_fr: list[str] = field(default_factory=list)  # signes d'alerte (fr)
    red_flags_en: list[str] = field(default_factory=list)  # signes d'alerte (en)
    advice_fr: str = ""           # conduite à tenir (fr)
    advice_en: str = ""           # conduite à tenir (en)


# ---------------------------------------------------------------------------
# Table des tableaux cliniques. Volontairement variée sur les 3 niveaux.
# ---------------------------------------------------------------------------
PRESENTATIONS: list[Presentation] = [
    # ---------------------- URGENCE VITALE (niveau 1) ----------------------
    Presentation(
        key="douleur_thoracique",
        level="URGENCE_VITALE",
        symptoms_fr=["douleur thoracique constrictive", "essoufflement", "sueurs"],
        symptoms_en=["crushing chest pain", "shortness of breath", "sweating"],
        rationale_fr=(
            "Une douleur thoracique constrictive associée à un essoufflement et "
            "des sueurs évoque un syndrome coronarien aigu (infarctus possible). "
            "Le pronostic vital peut être engagé à très court terme."
        ),
        rationale_en=(
            "Crushing chest pain with breathlessness and sweating suggests an "
            "acute coronary syndrome (possible heart attack). It can be "
            "immediately life-threatening."
        ),
        red_flags_fr=["irradiation au bras gauche ou à la mâchoire", "malaise"],
        red_flags_en=["pain radiating to left arm or jaw", "feeling faint"],
        advice_fr="Alerter immédiatement l'équipe médicale et préparer un ECG.",
        advice_en="Alert the medical team immediately and prepare an ECG.",
    ),
    Presentation(
        key="avc",
        level="URGENCE_VITALE",
        symptoms_fr=["paralysie d'un côté du visage", "trouble de la parole", "faiblesse d'un bras"],
        symptoms_en=["facial drooping on one side", "slurred speech", "arm weakness"],
        rationale_fr=(
            "L'association déficit facial + trouble du langage + faiblesse d'un "
            "membre est très évocatrice d'un accident vasculaire cérébral (AVC). "
            "Chaque minute compte pour préserver le cerveau."
        ),
        rationale_en=(
            "Facial droop, speech difficulty and limb weakness together strongly "
            "suggest a stroke. Time is brain: every minute counts."
        ),
        red_flags_fr=["apparition brutale des symptômes", "heure de début connue"],
        red_flags_en=["sudden onset", "known time of onset"],
        advice_fr="Déclencher la filière AVC sans délai (imagerie cérébrale urgente).",
        advice_en="Activate the stroke pathway without delay (urgent brain imaging).",
    ),
    Presentation(
        key="detresse_respiratoire",
        level="URGENCE_VITALE",
        symptoms_fr=["difficulté respiratoire sévère", "lèvres bleutées", "incapacité à parler"],
        symptoms_en=["severe breathing difficulty", "bluish lips", "unable to speak"],
        rationale_fr=(
            "Une détresse respiratoire avec cyanose (lèvres bleutées) traduit une "
            "hypoxie : l'oxygénation des organes est compromise. C'est une urgence "
            "absolue."
        ),
        rationale_en=(
            "Respiratory distress with cyanosis (bluish lips) reflects hypoxia: "
            "organ oxygenation is compromised. This is an absolute emergency."
        ),
        red_flags_fr=["saturation en oxygène basse", "épuisement respiratoire"],
        red_flags_en=["low oxygen saturation", "respiratory exhaustion"],
        advice_fr="Oxygénothérapie immédiate et appel du réanimateur.",
        advice_en="Immediate oxygen therapy and call the intensivist.",
    ),
    Presentation(
        key="hemorragie",
        level="URGENCE_VITALE",
        symptoms_fr=["saignement abondant non maîtrisé", "pâleur", "vertiges"],
        symptoms_en=["uncontrolled heavy bleeding", "pallor", "dizziness"],
        rationale_fr=(
            "Une hémorragie abondante avec pâleur et vertiges fait craindre un "
            "choc hémorragique (perte importante de sang). Le risque vital est "
            "immédiat."
        ),
        rationale_en=(
            "Heavy bleeding with pallor and dizziness raises concern for "
            "hemorrhagic shock (major blood loss). The risk to life is immediate."
        ),
        red_flags_fr=["pouls rapide et filant", "tension artérielle basse"],
        red_flags_en=["rapid weak pulse", "low blood pressure"],
        advice_fr="Compression directe, pose de voie veineuse et remplissage urgent.",
        advice_en="Direct pressure, IV access and urgent fluid resuscitation.",
    ),
    Presentation(
        key="anaphylaxie",
        level="URGENCE_VITALE",
        symptoms_fr=["gonflement du visage et de la gorge", "urticaire généralisée", "gêne respiratoire"],
        symptoms_en=["facial and throat swelling", "widespread hives", "difficulty breathing"],
        rationale_fr=(
            "Un œdème du visage/de la gorge avec urticaire et gêne respiratoire "
            "après exposition à un allergène évoque une anaphylaxie : l'obstruction "
            "des voies aériennes peut survenir en quelques minutes."
        ),
        rationale_en=(
            "Facial/throat swelling with hives and breathing difficulty after "
            "allergen exposure suggests anaphylaxis: airway obstruction can occur "
            "within minutes."
        ),
        red_flags_fr=["sensation de gorge qui se ferme", "exposition allergénique récente"],
        red_flags_en=["sensation of throat closing", "recent allergen exposure"],
        advice_fr="Adrénaline intramusculaire sans délai et surveillance rapprochée.",
        advice_en="Intramuscular adrenaline without delay and close monitoring.",
    ),
    Presentation(
        key="trouble_conscience",
        level="URGENCE_VITALE",
        symptoms_fr=["perte de connaissance", "convulsions", "absence de réponse"],
        symptoms_en=["loss of consciousness", "seizures", "unresponsive"],
        rationale_fr=(
            "Une altération profonde de la conscience ou des convulsions "
            "persistantes met en jeu la protection des voies aériennes et peut "
            "refléter une atteinte cérébrale grave."
        ),
        rationale_en=(
            "A profound alteration of consciousness or persistent seizures "
            "threatens airway protection and may reflect serious brain injury."
        ),
        red_flags_fr=["score de Glasgow bas", "convulsions de plus de 5 minutes"],
        red_flags_en=["low Glasgow score", "seizure lasting over 5 minutes"],
        advice_fr="Mettre en position de sécurité, surveiller les voies aériennes, alerter la réanimation.",
        advice_en="Place in recovery position, protect the airway, alert intensive care.",
    ),

    # ---------------------- URGENCE MODÉRÉE (niveau 2) ----------------------
    Presentation(
        key="fracture_simple",
        level="URGENCE_MODEREE",
        symptoms_fr=["douleur vive à la cheville après une chute", "gonflement", "difficulté à marcher"],
        symptoms_en=["sharp ankle pain after a fall", "swelling", "difficulty walking"],
        rationale_fr=(
            "Une douleur localisée avec gonflement et impotence après un "
            "traumatisme évoque une possible fracture. La situation est douloureuse "
            "mais sans menace vitale immédiate."
        ),
        rationale_en=(
            "Localised pain with swelling and inability to bear weight after "
            "trauma suggests a possible fracture. Painful but not immediately "
            "life-threatening."
        ),
        red_flags_fr=["déformation visible", "perte de sensibilité"],
        red_flags_en=["visible deformity", "loss of sensation"],
        advice_fr="Immobiliser, antalgiques, radiographie à prévoir.",
        advice_en="Immobilise, give analgesia, plan an X-ray.",
    ),
    Presentation(
        key="douleur_abdominale",
        level="URGENCE_MODEREE",
        symptoms_fr=["douleur abdominale en fosse iliaque droite", "fièvre modérée", "nausées"],
        symptoms_en=["right lower abdominal pain", "moderate fever", "nausea"],
        rationale_fr=(
            "Une douleur de la fosse iliaque droite avec fièvre et nausées peut "
            "correspondre à une appendicite. Une évaluation rapide est nécessaire "
            "pour éviter une complication."
        ),
        rationale_en=(
            "Right lower-quadrant pain with fever and nausea may indicate "
            "appendicitis. Prompt assessment is needed to avoid complications."
        ),
        red_flags_fr=["douleur qui s'aggrave", "ventre dur"],
        red_flags_en=["worsening pain", "rigid abdomen"],
        advice_fr="Examen clinique, bilan biologique et avis chirurgical.",
        advice_en="Clinical exam, blood tests and surgical opinion.",
    ),
    Presentation(
        key="asthme_modere",
        level="URGENCE_MODEREE",
        symptoms_fr=["sifflements respiratoires", "toux", "gêne à l'effort"],
        symptoms_en=["wheezing", "cough", "breathlessness on exertion"],
        rationale_fr=(
            "Une crise d'asthme modérée (sifflements, gêne) répond souvent au "
            "traitement bronchodilatateur mais nécessite une surveillance pour "
            "détecter une aggravation."
        ),
        rationale_en=(
            "A moderate asthma attack (wheezing, breathlessness) often responds "
            "to bronchodilators but requires monitoring to detect worsening."
        ),
        red_flags_fr=["difficulté à finir ses phrases", "saturation qui baisse"],
        red_flags_en=["unable to finish sentences", "falling oxygen saturation"],
        advice_fr="Bronchodilatateurs, surveillance de la saturation.",
        advice_en="Bronchodilators, monitor oxygen saturation.",
    ),
    Presentation(
        key="deshydratation",
        level="URGENCE_MODEREE",
        symptoms_fr=["vomissements répétés", "diarrhée", "fatigue importante"],
        symptoms_en=["repeated vomiting", "diarrhea", "marked fatigue"],
        rationale_fr=(
            "Des vomissements et diarrhées répétés exposent à une déshydratation, "
            "surtout chez les personnes fragiles. Une réhydratation et un bilan "
            "sont à organiser sans urgence vitale immédiate."
        ),
        rationale_en=(
            "Repeated vomiting and diarrhea risk dehydration, especially in frail "
            "patients. Rehydration and work-up are needed without immediate "
            "life threat."
        ),
        red_flags_fr=["incapacité à boire", "somnolence inhabituelle"],
        red_flags_en=["unable to drink", "unusual drowsiness"],
        advice_fr="Réhydratation orale ou intraveineuse selon la tolérance.",
        advice_en="Oral or intravenous rehydration depending on tolerance.",
    ),
    Presentation(
        key="plaie_profonde",
        level="URGENCE_MODEREE",
        symptoms_fr=["plaie profonde à la main", "saignement contrôlé", "douleur"],
        symptoms_en=["deep hand wound", "controlled bleeding", "pain"],
        rationale_fr=(
            "Une plaie profonde dont le saignement est maîtrisé requiert un "
            "parage, parfois des points de suture et une vérification du statut "
            "vaccinal antitétanique, sans urgence vitale."
        ),
        rationale_en=(
            "A deep wound with controlled bleeding needs cleaning, sometimes "
            "stitches, and a tetanus status check, without a life threat."
        ),
        red_flags_fr=["atteinte d'un tendon", "corps étranger"],
        red_flags_en=["tendon involvement", "foreign body"],
        advice_fr="Nettoyage, suture si besoin, vérifier la vaccination antitétanique.",
        advice_en="Clean, suture if needed, check tetanus vaccination.",
    ),
    Presentation(
        key="cephalee_febrile",
        level="URGENCE_MODEREE",
        symptoms_fr=["mal de tête persistant", "fièvre", "sensibilité à la lumière"],
        symptoms_en=["persistent headache", "fever", "light sensitivity"],
        rationale_fr=(
            "Un mal de tête fébrile avec photophobie doit faire évoquer (et "
            "écarter) une méningite. Une évaluation médicale rapprochée est "
            "justifiée."
        ),
        rationale_en=(
            "A febrile headache with photophobia should raise (and rule out) "
            "meningitis. Prompt medical assessment is warranted."
        ),
        red_flags_fr=["raideur de la nuque", "éruption cutanée"],
        red_flags_en=["neck stiffness", "skin rash"],
        advice_fr="Examen neurologique ; en cas de raideur de nuque, urgence absolue.",
        advice_en="Neurological exam; if neck stiffness, treat as absolute emergency.",
    ),

    # ------------------- CONSULTATION DIFFÉRÉE (niveau 3) -------------------
    Presentation(
        key="rhume",
        level="CONSULTATION_DIFFEREE",
        symptoms_fr=["nez qui coule", "éternuements", "légère fatigue"],
        symptoms_en=["runny nose", "sneezing", "mild fatigue"],
        rationale_fr=(
            "Un rhume banal (rhinite) évolue spontanément favorablement. Aucun "
            "signe de gravité : une prise en charge symptomatique suffit."
        ),
        rationale_en=(
            "A common cold resolves on its own. No warning signs: symptomatic "
            "care is enough."
        ),
        red_flags_fr=["fièvre élevée persistante", "gêne respiratoire"],
        red_flags_en=["persistent high fever", "breathing difficulty"],
        advice_fr="Repos, hydratation, traitement symptomatique. Consulter si aggravation.",
        advice_en="Rest, hydration, symptomatic treatment. See a doctor if it worsens.",
    ),
    Presentation(
        key="lombalgie",
        level="CONSULTATION_DIFFEREE",
        symptoms_fr=["douleur lombaire après un effort", "raideur", "gêne aux mouvements"],
        symptoms_en=["lower back pain after exertion", "stiffness", "movement discomfort"],
        rationale_fr=(
            "Une lombalgie commune après un effort, sans signe neurologique, est "
            "bénigne dans la grande majorité des cas et relève d'un traitement "
            "symptomatique."
        ),
        rationale_en=(
            "Common low back pain after exertion, without neurological signs, is "
            "benign in the vast majority of cases and managed symptomatically."
        ),
        red_flags_fr=["perte de force dans les jambes", "troubles urinaires"],
        red_flags_en=["leg weakness", "urinary problems"],
        advice_fr="Antalgiques, maintien d'une activité douce. Consulter si signe neurologique.",
        advice_en="Analgesia, keep gentle activity. See a doctor if neurological signs.",
    ),
    Presentation(
        key="angine",
        level="CONSULTATION_DIFFEREE",
        symptoms_fr=["mal de gorge", "fièvre légère", "difficulté modérée à avaler"],
        symptoms_en=["sore throat", "mild fever", "moderate difficulty swallowing"],
        rationale_fr=(
            "Un mal de gorge avec fièvre légère évoque une angine, le plus souvent "
            "virale. Une consultation différée permet de juger de l'intérêt d'un "
            "test et d'un éventuel antibiotique."
        ),
        rationale_en=(
            "A sore throat with mild fever suggests tonsillitis, most often viral. "
            "A deferred consultation can assess the need for a test and possible "
            "antibiotics."
        ),
        red_flags_fr=["difficulté à respirer", "impossibilité totale d'avaler"],
        red_flags_en=["difficulty breathing", "complete inability to swallow"],
        advice_fr="Antalgiques, hydratation. Consulter le médecin traitant.",
        advice_en="Analgesia, hydration. See the general practitioner.",
    ),
    Presentation(
        key="eczema",
        level="CONSULTATION_DIFFEREE",
        symptoms_fr=["plaques rouges qui démangent", "peau sèche", "pas de fièvre"],
        symptoms_en=["itchy red patches", "dry skin", "no fever"],
        rationale_fr=(
            "Des lésions cutanées prurigineuses sans fièvre ni signe systémique "
            "évoquent un eczéma. La situation n'est pas urgente."
        ),
        rationale_en=(
            "Itchy skin lesions without fever or systemic signs suggest eczema. "
            "The situation is not urgent."
        ),
        red_flags_fr=["extension rapide", "signes d'infection (pus)"],
        red_flags_en=["rapid spread", "signs of infection (pus)"],
        advice_fr="Émollients, consultation dermatologique programmée.",
        advice_en="Emollients, scheduled dermatology consultation.",
    ),
    Presentation(
        key="conjonctivite",
        level="CONSULTATION_DIFFEREE",
        symptoms_fr=["œil rouge", "larmoiement", "sensation de grain de sable"],
        symptoms_en=["red eye", "watering", "gritty sensation"],
        rationale_fr=(
            "Un œil rouge larmoyant sans baisse de vision évoque une conjonctivite "
            "le plus souvent bénigne. Pas d'urgence vitale."
        ),
        rationale_en=(
            "A red watery eye without vision loss suggests usually benign "
            "conjunctivitis. No life threat."
        ),
        red_flags_fr=["baisse de la vision", "douleur oculaire intense"],
        red_flags_en=["vision loss", "severe eye pain"],
        advice_fr="Hygiène des mains, lavage oculaire. Consulter si baisse de vision.",
        advice_en="Hand hygiene, eye rinsing. See a doctor if vision decreases.",
    ),
    Presentation(
        key="entorse_legere",
        level="CONSULTATION_DIFFEREE",
        symptoms_fr=["douleur modérée au poignet", "léger gonflement", "mobilité conservée"],
        symptoms_en=["moderate wrist pain", "slight swelling", "preserved mobility"],
        rationale_fr=(
            "Une entorse légère, avec mobilité conservée et sans déformation, "
            "guérit habituellement avec un traitement simple."
        ),
        rationale_en=(
            "A mild sprain with preserved mobility and no deformity usually heals "
            "with simple treatment."
        ),
        red_flags_fr=["impossibilité de bouger l'articulation", "déformation"],
        red_flags_en=["inability to move the joint", "deformity"],
        advice_fr="Glace, repos, contention souple. Consulter si aggravation.",
        advice_en="Ice, rest, soft support. See a doctor if it worsens.",
    ),
]


# ---------------------------------------------------------------------------
# Fabrication des réponses structurées (le FORMAT que l'agent doit produire)
# ---------------------------------------------------------------------------
def format_triage_answer(p: Presentation, lang: str) -> str:
    """Compose une réponse de triage structurée et homogène pour une vignette.

    Le format est volontairement stable (mêmes intitulés) pour que :
      * le modèle apprenne un gabarit régulier, facile à reproduire ;
      * l'API puisse l'analyser (extraire le niveau) de façon fiable.
    """
    info = TRIAGE_LEVELS[p.level]
    if lang == "fr":
        red_flags = p.red_flags_fr or ["aucun signe d'alerte spécifique identifié"]
        return (
            f"Niveau de priorité : {p.level} ({info['label_fr']})\n"
            f"Délai recommandé : {info['delai']}\n\n"
            f"Analyse : {p.rationale_fr}\n\n"
            "Signes d'alerte à surveiller :\n"
            + "\n".join(f"- {rf}" for rf in red_flags)
            + f"\n\nRecommandation : {p.advice_fr}\n\n{MEDICAL_DISCLAIMER}"
        )
    # anglais
    red_flags = p.red_flags_en or ["no specific warning sign identified"]
    return (
        f"Priority level: {p.level} ({info['label_en']})\n"
        f"Recommended timeframe: {info['delai']}\n\n"
        f"Assessment: {p.rationale_en}\n\n"
        "Warning signs to monitor:\n"
        + "\n".join(f"- {rf}" for rf in red_flags)
        + f"\n\nRecommendation: {p.advice_en}\n\n{MEDICAL_DISCLAIMER}"
    )


# Variantes de formulation de la question patient (pour diversifier le dataset).
_QUESTION_TEMPLATES_FR = [
    "Un patient se présente aux urgences avec : {symptoms}. Quel est le niveau de priorité ?",
    "Voici les symptômes décrits : {symptoms}. Comment trier ce patient ?",
    "Patient {age} ans. Motif : {symptoms}. Évalue la priorité de prise en charge.",
    "J'ai {symptoms}. Est-ce urgent ?",
    "Symptômes relevés au triage : {symptoms}. Niveau d'urgence ?",
]
_QUESTION_TEMPLATES_EN = [
    "A patient presents to the ER with: {symptoms}. What is the priority level?",
    "Here are the reported symptoms: {symptoms}. How should this patient be triaged?",
    "Patient aged {age}. Complaint: {symptoms}. Assess the care priority.",
    "I have {symptoms}. Is this urgent?",
    "Symptoms recorded at triage: {symptoms}. Urgency level?",
]


def generate_triage_vignettes(
    n_per_presentation: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Génère des vignettes de triage variées sous forme de dictionnaires bruts.

    Pour chaque tableau clinique, on produit `n_per_presentation` variantes en
    combinant : la langue, le gabarit de question, un âge aléatoire et un
    sous-ensemble de symptômes. Cela crée de la diversité sans inventer de
    fausses données patient (tout est paramétrique et tracé comme synthétique).

    Renvoie des dicts contenant tout le nécessaire pour `build_sft` et `build_dpo`.
    """
    rng = random.Random(seed)
    vignettes: list[dict] = []

    for p in PRESENTATIONS:
        for i in range(n_per_presentation):
            lang = "fr" if i % 2 == 0 else "en"  # 50/50 fr/en → corpus bilingue
            symptoms = p.symptoms_fr if lang == "fr" else p.symptoms_en
            templates = _QUESTION_TEMPLATES_FR if lang == "fr" else _QUESTION_TEMPLATES_EN

            # On garde au moins 2 symptômes pour rester réaliste.
            k = rng.randint(2, len(symptoms))
            chosen_symptoms = rng.sample(symptoms, k)
            symptoms_str = ", ".join(chosen_symptoms)
            age = rng.randint(18, 88)

            question = rng.choice(templates).format(symptoms=symptoms_str, age=age)
            answer = format_triage_answer(p, lang)

            vignettes.append(
                {
                    "id": f"triage-{p.key}-{lang}-{i:03d}",
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "lang": lang,
                    "task_type": "triage",
                    "symptoms": chosen_symptoms,
                    "triage_level": p.level,
                    "presentation_key": p.key,
                    "source": "triage_synth",
                    "license": "CC0 (synthétique, généré par règles)",
                    "confidence": 0.9,
                }
            )
    rng.shuffle(vignettes)
    return vignettes


# Ordre de gravité croissant, pour distinguer sous-triage et sur-triage.
_SEVERITY_ORDER = ["CONSULTATION_DIFFEREE", "URGENCE_MODEREE", "URGENCE_VITALE"]

# Mauvais niveaux associés à chaque niveau correct, pour fabriquer le « rejeté ».
# Chaque niveau propose TROIS mauvaises cibles, alternées d'une vignette à
# l'autre (paramètre `variant`), avec une ASYMÉTRIE PRUDENTE ET BORNÉE :
#   * le SOUS-triage (l'erreur interdite en triage) est montré un peu plus
#     souvent comme « mauvaise réponse » que le sur-triage (~56 % vs ~44 %) ;
#   * mais CHAQUE niveau reste régulièrement rejeté (ratio max/min ≤ 2),
#     garanti par un test unitaire.
# ⚠️ BUG HISTORIQUE : avec un mapping fixe, URGENCE_VITALE n'apparaissait JAMAIS
# en « rejected » : le gradient DPO poussait systématiquement le modèle VERS
# l'urgence vitale → cas bénins classés graves. Ne jamais épargner un niveau.
_WRONG_LEVELS = {
    "URGENCE_VITALE": (
        "URGENCE_MODEREE", "CONSULTATION_DIFFEREE", "URGENCE_MODEREE",
    ),
    "URGENCE_MODEREE": (
        "CONSULTATION_DIFFEREE", "URGENCE_VITALE", "CONSULTATION_DIFFEREE",
    ),
    "CONSULTATION_DIFFEREE": (
        "URGENCE_MODEREE", "URGENCE_VITALE", "URGENCE_MODEREE",
    ),
}


def make_bad_triage_answer(p: Presentation, lang: str, variant: int = 0) -> str:
    """Fabrique une réponse de triage VOLONTAIREMENT mauvaise (pour le DPO).

    La mauvaise réponse reprend le MÊME gabarit (mêmes intitulés, longueur
    comparable) que la bonne : le DPO doit apprendre à rejeter le CONTENU
    (mauvais niveau, analyse erronée, conseil inadapté), pas un simple style
    « court = mauvais » (biais de longueur classique du DPO, corrigé ici).
    `variant` alterne la mauvaise cible (asymétrie prudente bornée, cf. table).
    """
    wrong_level = _WRONG_LEVELS[p.level][variant % 3]
    info = TRIAGE_LEVELS[wrong_level]
    is_undertriage = (
        _SEVERITY_ORDER.index(wrong_level) < _SEVERITY_ORDER.index(p.level)
    )

    if lang == "fr":
        if is_undertriage:
            analysis = (
                "Les symptômes décrits ne paraissent pas inquiétants à ce stade. "
                "Ce type de tableau est le plus souvent bénin et ne justifie pas "
                "de mobiliser l'équipe en priorité."
            )
            advice = ("Rentrez chez vous, reposez-vous et prenez un antalgique "
                      "si besoin ; inutile de consulter rapidement.")
        else:
            analysis = (
                "Les symptômes décrits sont très alarmants et font craindre une "
                "complication majeure imminente. Il ne faut prendre aucun risque, "
                "même en l'absence de signe de gravité objectif."
            )
            advice = ("Mobiliser immédiatement toute l'équipe et engager une "
                      "prise en charge maximale sans attendre l'évaluation.")
        return (
            f"Niveau de priorité : {wrong_level} ({info['label_fr']})\n"
            f"Délai recommandé : {info['delai']}\n\n"
            f"Analyse : {analysis}\n\n"
            "Signes d'alerte à surveiller :\n- aucun signe particulier à surveiller\n\n"
            f"Recommandation : {advice}\n\n{MEDICAL_DISCLAIMER}"
        )

    if is_undertriage:
        analysis = (
            "The reported symptoms do not look worrying at this stage. This kind "
            "of presentation is most often benign and does not justify "
            "prioritising the patient."
        )
        advice = ("Go home, rest and take a painkiller if needed; there is no "
                  "need to seek care quickly.")
    else:
        analysis = (
            "The reported symptoms are highly alarming and suggest an imminent "
            "major complication. No risk should be taken, even without any "
            "objective severity sign."
        )
        advice = ("Mobilise the whole team immediately and start maximal care "
                  "without waiting for the assessment.")
    return (
        f"Priority level: {wrong_level} ({info['label_en']})\n"
        f"Recommended timeframe: {info['delai']}\n\n"
        f"Assessment: {analysis}\n\n"
        "Warning signs to monitor:\n- nothing specific to monitor\n\n"
        f"Recommendation: {advice}\n\n{MEDICAL_DISCLAIMER}"
    )

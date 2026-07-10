# ☁️ Guide de déploiement cloud — pas à pas

> Objectif mission : « Endpoint de démonstration déployé sur le cloud,
> optimisé pour une inférence rapide grâce à vLLM. »
> Ce guide part de l'existant du dépôt (image GHCR construite par
> `.github/workflows/deploy.yml`, `docker-compose.yml`, config vLLM) et va
> jusqu'à l'URL publique. Tout est déjà prêt côté code : il ne manque que le
> compte cloud et ~30 minutes.

---

## 0. Prérequis (une seule fois)

1. **Pousser le dépôt sur GitHub** (voir réserve git de l'audit : le dépôt
   local n'a aucun commit) :

   ```bash
   git commit -m "POC agent de triage médical CHSA (SFT LoRA + DPO + API vLLM)"
   git remote add origin git@github.com:<votre-compte>/OC_P14.git
   git push -u origin master
   ```

2. Vérifier que les 2 workflows passent au vert dans l'onglet *Actions* :
   * `ci.yml` → lint + tests ;
   * `deploy.yml` → construit et publie `ghcr.io/<compte>/oc_p14-api:master`.

3. **Publier les poids du modèle fusionné** (3,3 Go — trop lourd pour git) sur
   le Hugging Face Hub, en dépôt privé :

   ```bash
   uv run huggingface-cli login
   uv run huggingface-cli upload <compte>/chsa-triage-qwen3-1.7b \
       models/qwen3-triage-merged --private
   ```

   > Alternative sans HF : `scp -r models/qwen3-triage-merged` vers la VM.

---

## 1. Option A — VM GPU « classique » (Scaleway / OVH / AWS…) — recommandé

C'est l'option la plus proche de la production hospitalière (SIH on-premise).
Une seule VM GPU suffit pour le POC : Qwen3-1.7B en bf16 tient dans ~6 Go de
VRAM ; toute instance L4 / T4 / A10 / RTX 4000 convient.

1. **Créer la VM GPU** (Ubuntu 22.04) chez le fournisseur choisi et installer
   Docker + le NVIDIA Container Toolkit :

   ```bash
   curl -fsSL https://get.docker.com | sh
   # NVIDIA Container Toolkit (expose le GPU aux conteneurs)
   distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   ```

2. **Récupérer le dépôt et les poids** :

   ```bash
   git clone https://github.com/<compte>/OC_P14.git && cd OC_P14
   # Poids depuis HF (dépôt privé → token) :
   pip install -U "huggingface_hub[cli]" && huggingface-cli login
   huggingface-cli download <compte>/chsa-triage-qwen3-1.7b \
       --local-dir models/qwen3-triage-merged
   ```

3. **Lancer la pile complète** (vLLM + passerelle API) :

   ```bash
   docker compose -f deploy/docker-compose.yml up -d --build
   # ou, pour utiliser l'image déjà publiée par la CI au lieu de rebuilder :
   #   remplacer `build:` par `image: ghcr.io/<compte>/oc_p14-api:master`
   ```

4. **Vérifier** :

   ```bash
   curl http://localhost:8080/health
   curl -X POST http://localhost:8080/triage -H 'Content-Type: application/json' \
        -d '{"message":"douleur thoracique et essoufflement depuis 1h"}'
   ```

5. **Exposer proprement** (ne JAMAIS ouvrir le port 8000 de vLLM, seulement le
   8080 de la passerelle — le compose lie déjà vLLM à 127.0.0.1) :

   * ouvrir le port 443 dans le groupe de sécurité du fournisseur ;
   * mettre un reverse-proxy TLS devant l'API, par exemple **Caddy**
     (2 lignes de config, certificat automatique) :

     ```
     triage.mondomaine.fr {
         reverse_proxy localhost:8080
     }
     ```

   * la démo est alors accessible sur `https://triage.mondomaine.fr/ui`.

---

## 2. Option B — RunPod / Vast.ai (GPU à l'heure, le moins cher pour une démo)

Idéal pour une soutenance : on loue le GPU 1-2 h.

1. Créer un pod GPU (template « Docker », par ex. RTX A4000/L4), image de base
   `vllm/vllm-openai:latest`, port HTTP 8000 exposé, et monter les poids
   (upload ou téléchargement HF au démarrage) :

   ```bash
   --model /workspace/qwen3-triage-merged --served-model-name chsa-triage --max-model-len 2048
   ```

2. Lancer la passerelle (sur le même pod ou n'importe quelle petite VM/CPU —
   elle n'a pas besoin de GPU) :

   ```bash
   docker run -d -p 8080:8080 \
     -e TRIAGE_INFERENCE_BACKEND=vllm \
     -e TRIAGE_VLLM_BASE_URL=http://<ip-du-pod>:8000/v1 \
     -e TRIAGE_VLLM_MODEL_NAME=chsa-triage \
     ghcr.io/<compte>/oc_p14-api:master
   ```

3. L'URL publique du pod (proxy RunPod) sert la démo : `…/ui`, `…/docs`.

---

## 3. Brancher le déploiement sur la CI/CD (optionnel, déjà préparé)

`deploy.yml` publie l'image à chaque push. Pour aller jusqu'au déploiement
automatique, décommenter la dernière étape du workflow et ajouter les secrets
GitHub (`Settings → Secrets and variables → Actions`) :

* `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` → l'étape SSH tire la
  nouvelle image et relance `docker compose` sur la VM.

---

## 4. Check-list go / no-go avant de partager l'URL

- [ ] `GET /health` → 200 et `backend: "vllm"` ;
- [ ] `POST /triage` cas vital → `URGENCE_VITALE` + red flags ;
- [ ] `POST /triage` cas bénin détaillé → PAS d'urgence vitale ;
- [ ] `GET /ui` → la page de chat répond (parcours complet avec relance) ;
- [ ] `GET /audit` → les interactions précédentes sont tracées ;
- [ ] latence p95 < 3 s sur 10 requêtes (`uv run triage-evaluate` la mesure) ;
- [ ] port 8000 (vLLM brut) NON accessible depuis l'extérieur ;
- [ ] HTTPS actif, URL de démo notée dans le rapport de soutenance.

## Sécurité / RGPD (rappels POC)

* Aucune donnée réelle de patient sur l'endpoint de démonstration.
* Le journal `logs/audit.log` reste DANS le conteneur/VM (ne pas l'exposer).
* L'API n'a pas d'authentification : ne pas laisser l'endpoint ouvert
  au-delà de la démo, ou ajouter une clé d'API au reverse-proxy.

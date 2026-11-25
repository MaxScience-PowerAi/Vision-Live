# Vision Live

Vision Live analyse en direct les images d’une webcam ou d’un iPhone connecté via DroidCam. L’application détecte et suit les objets, apprend de tes corrections et répond en français à des questions sur la scène.

- YOLOE détecte et segmente les objets. MobileCLIP2 affine les détections.
- La mémoire visuelle réutilise tes corrections et peut entraîner un adaptateur.
- Qwen3-VL décrit la scène. Whisper transcrit les questions et Kokoro ou Piper lit les réponses.
- Le mode surveillance applique des règles à des zones et peut enregistrer un événement et envoyer une alerte Telegram.

## Démarrage

La configuration d’origine utilise Linux, un environnement Conda nommé `ai_master`, Python 3.11 et un GPU NVIDIA avec une version de PyTorch compatible CUDA. Le lanceur d’origine et le notebook attendent le projet dans `~/Documents/Vision_Live`. Sur une nouvelle machine, clone le dépôt à cet emplacement.

Le projet utilise notamment Ultralytics, OpenCV, OpenCLIP, timm, PyAV, Transformers et bitsandbytes. Consulte les cellules de [Vision_Live.ipynb](Vision_Live.ipynb) pour les dépendances propres à la voix et à chaque fonctionnalité.

Place les poids dans `~/Modeles_IA/` selon le [guide des modèles](docs/modeles.md). Pour Qwen, copie `outils/charger_qwen.py` dans ce même dossier puis lance `bash telecharger_qwen.sh` depuis le dossier du projet. Le lanceur active le mode hors ligne de Hugging Face. Prépare les modèles et caches avant le premier lancement.

Active ton environnement Python, puis lance l’application depuis le dossier du projet.

```bash
python lancer_vision.py
```

Tu peux aussi ouvrir le notebook et exécuter sa cellule de lancement rapide. Dans la fenêtre vidéo, `F` ouvre la saisie texte, `V` lance la commande vocale, `I` demande une description de la scène et `Q` quitte.

Les captures, événements, journaux, modèles et réglages personnels restent sur la machine. Pour Telegram, suis le format décrit dans le notebook et conserve le jeton dans `config/telegram.json`.

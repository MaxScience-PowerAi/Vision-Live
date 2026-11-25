# Modeles_IA — les modèles communs à tous tes projets

Téléchargés une seule fois. N'importe quel projet de l'environnement `ai_master` peut les utiliser en
indiquant ce dossier (`Path.home() / "Modeles_IA"`).

| Fichier / dossier | Modèle | Taille | Utilisation |
|---|---|---|---|
| `yoloe-11m-seg.pt` | YOLOE-11m (Ultralytics) | 58 Mo | détection à vocabulaire ouvert : `YOLOE(str(MODELES / "yoloe-11m-seg.pt"))` |
| `mobileclip_blt.ts` | encodeur de texte de YOLOE | 572 Mo | Ultralytics le cherche dans le dossier courant → lancer `get_text_pe` depuis ce dossier |
| `yolo11m.pt` | YOLO11m classique (80 classes COCO) | 39 Mo | `YOLO(str(MODELES / "yolo11m.pt"))` |
| `visages/` | YuNet (détection) + SFace (reconnaissance), OpenCV | 38 Mo | `cv2.FaceDetectorYN.create(...)`, `cv2.FaceRecognizerSF.create(...)` |
| `qwen3-vl-4b-bnb4/` | Qwen3-VL-4B-Instruct, 4 bits (unsloth) | 4,1 Go | **passer par `charger_qwen.py`** (2 défauts du fichier y sont réparés) |
| `cache_textes.pt`, `cache_verif.pt` | caches de signatures texte (YOLOE, MobileCLIP2) | < 1 Mo | se reconstruisent tout seuls s'ils sont effacés |

Ailleurs (cache Hugging Face, `~/.cache/huggingface`, trouvés automatiquement) : Whisper small, MobileCLIP2-S2.

## Qwen3-VL-4B en 3 lignes
```python
import sys; from pathlib import Path; sys.path.insert(0, str(Path.home() / "Modeles_IA"))
from charger_qwen import charger, demander, decharger
qwen = charger()                                   # ~6 s, 4,3 Go de mémoire graphique
print(demander(qwen, image, "Que se passe-t-il sur cette image ?"))   # image OpenCV (BGR) ou PIL, ~5-7 s
```
Mesuré sur la Quadro RTX 3000 (5,6 Go utilisables) : pic de 4,46 Go seul, 5,59 Go avec tout Vision Live
(YOLOE + MobileCLIP2 + Whisper). La taille de l'image ne change presque pas la mémoire ; à 320 px il invente
des détails, garder 640 px.

"""Qwen3-VL-4B (4 bits) — chargement réparé + question sur une image. Utilisable par n'importe quel projet :

    import sys; sys.path.insert(0, str(Path.home() / "Modeles_IA"))
    from charger_qwen import charger, demander, decharger
    qwen = charger()
    print(demander(qwen, image_bgr_ou_pil, "Que se passe-t-il ?"))
    decharger(qwen)                      # rend la mémoire graphique

Le fichier publié par unsloth a deux défauts, réparés ici au chargement :
  1. 6 couches de la vision (deepstack_merger_list.*.linear_fc1/fc2) ne sont PAS compressées dans le fichier,
     mais transformers les prépare en 4 bits → plantage (AssertionError de bitsandbytes). On les remplace par
     de vraies couches avec leurs poids lus dans le fichier.
  2. Quelques couches restent en bfloat16 (la carte Quadro RTX 3000 ne le gère pas nativement) → converties en float16.
"""
import gc
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

QWEN = Path.home() / "Modeles_IA" / "qwen3-vl-4b-bnb4"


def charger(dossier=QWEN, device="cuda:0"):
    import bitsandbytes as bnb
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(dossier, device_map=device, dtype=torch.float16)
    with safe_open(str(Path(dossier) / "model.safetensors"), "pt", device="cpu") as fs:      # défaut n° 1
        for nom, mod in list(model.named_modules()):
            if isinstance(mod, bnb.nn.Linear4bit) and mod.weight.shape[1] != 1:
                w = fs.get_tensor(nom + ".weight")
                lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=mod.bias is not None, dtype=torch.float16, device=device)
                lin.weight.data.copy_(w)
                if mod.bias is not None:
                    lin.bias.data.copy_(fs.get_tensor(nom + ".bias"))
                parent, _, enfant = nom.rpartition(".")
                setattr(model.get_submodule(parent), enfant, lin)
    for mod in model.modules():                                                               # défaut n° 2
        if not isinstance(mod, bnb.nn.Linear4bit):
            for p in mod.parameters(recurse=False):
                if p.dtype == torch.bfloat16:            # conversion en passant par la RAM : pas de pic sur la carte
                    p.data = p.data.cpu()                # (le vocabulaire fait 742 Mo à lui seul)
                    p.data = p.data.half().to(device)
    torch.cuda.empty_cache()
    return {"model": model.eval(), "proc": AutoProcessor.from_pretrained(dossier), "device": device}


@torch.no_grad()
def demander(qwen, image, question, max_new_tokens=150, cote_max=640):
    """image : tableau OpenCV (BGR) ou image PIL. cote_max : plus petit = moins de mémoire et plus rapide."""
    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1].copy())
    k = cote_max / max(image.size)
    if k < 1:
        image = image.resize((round(image.width * k), round(image.height * k)))
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    entree = qwen["proc"].apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to(qwen["device"])
    sortie = qwen["model"].generate(**entree, max_new_tokens=max_new_tokens, do_sample=False)
    n = entree["input_ids"].shape[1]
    return qwen["proc"].batch_decode(sortie[:, n:], skip_special_tokens=True)[0].strip()


def decharger(qwen):
    qwen.clear()
    gc.collect()
    torch.cuda.empty_cache()

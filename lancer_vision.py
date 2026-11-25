"""VISION LIVE — lanceur en un clic.

Lit les cellules de code du notebook (la seule source du projet), les exécute dans l'ordre,
puis ouvre la fenêtre de vision. Si on modifie le notebook, ce lanceur suit automatiquement.

Utilisation :
  - double-clic sur l'icône « Vision Live » du bureau
  - ou dans le notebook : %run ~/Documents/Vision_Live/lancer_vision.py
  - ou dans un terminal : ~/anaconda3/envs/ai_master/bin/python ~/Documents/Vision_Live/lancer_vision.py
"""
import json
import os
import sys
from pathlib import Path

# Tous les modèles sont déjà sur le disque : on interdit les vérifications en ligne,
# pour que Vision Live démarre vite et marche même sans connexion.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

NOTEBOOK = Path.home() / "Documents" / "Vision_Live" / "Vision_Live.ipynb"
SAUVEGARDES = Path.home() / "Documents" / "Vision_Live" / "sauvegardes"


def trouver_notebook():
    """Le notebook de travail ; s'il a été déplacé ou supprimé, la sauvegarde la plus récente."""
    if NOTEBOOK.exists():
        return NOTEBOOK
    copies = sorted(SAUVEGARDES.glob("Vision_Live_v*.ipynb"), key=lambda p: p.stat().st_mtime)
    if copies:
        print(f"⚠ {NOTEBOOK} introuvable → utilisation de la sauvegarde {copies[-1].name}", flush=True)
        return copies[-1]
    sys.exit(f"Notebook introuvable : {NOTEBOOK} (et aucune sauvegarde dans {SAUVEGARDES})")


def cellules_a_executer(notebook):
    """Toutes les cellules de code, sauf : vides, installation (%pip, !...) et cellules qui appellent lancer()."""
    cellules = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for c in cellules:
        src = "".join(c["source"])
        lignes = [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
        if c["cell_type"] != "code" or not lignes:
            continue
        if any(l.startswith(("%", "!")) for l in lignes):    # magie Jupyter (installation)
            continue
        if lignes == ["lancer()"] or "%run" in src:          # cellules de lancement
            continue
        yield src


def main():
    notebook = trouver_notebook()
    espace = {"__name__": "vision_live"}
    for src in cellules_a_executer(notebook):
        titre = src.strip().splitlines()[0][:60]
        print(f"▶ {titre}", flush=True)
        exec(compile(src, f"{notebook.name} · {titre}", "exec"), espace)
    espace["lancer"]()
    return espace


if __name__ == "__main__":      # vrai aussi avec %run dans le notebook
    main()

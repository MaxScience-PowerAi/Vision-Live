#!/bin/bash
# Télécharge Qwen3-VL-4B (4 bits dynamique, ~4,3 Go) pour le niveau 4 de Vision Live.
# Reprend là où il s'est arrêté : on peut le relancer autant de fois que nécessaire.
# Sans xet (il bloquait à 0 octet), avec le téléchargement HTTP classique.

DOSSIER="$HOME/Documents/Vision_Live"
DEPOT="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
CIBLE="$HOME/Modeles_IA/qwen3-vl-4b-bnb4"
LOG="$DOSSIER/logs/telechargement_qwen.log"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$CIBLE" "$DOSSIER/logs"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate ai_master

for essai in 1 2 3 4 5 6 7 8 9 10; do
    echo "=== Essai $essai — $(date '+%H:%M:%S') ===" | tee -a "$LOG"
    if hf download "$DEPOT" --local-dir "$CIBLE" 2>&1 | tee -a "$LOG"; then
        echo "✅ Qwen3-VL-4B téléchargé dans $CIBLE" | tee -a "$LOG"
        du -sh "$CIBLE"
        exit 0
    fi
    echo "⚠️ Coupure, nouvel essai dans 30 s..." | tee -a "$LOG"
    sleep 30
done
echo "❌ Échec après 10 essais, voir $LOG" | tee -a "$LOG"
exit 1

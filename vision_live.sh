#!/usr/bin/env bash
# Lance VISION LIVE avec l'environnement ai_master (utilisé par l'icône du bureau).
LOG="$HOME/Documents/Vision_Live/logs/lanceur.log"
mkdir -p "$(dirname "$LOG")"
echo "===== $(date '+%d/%m/%Y %H:%M:%S') =====" >> "$LOG"
notify-send -i camera-web "Vision Live" "Démarrage... (chargement du modèle, ~15 s)" 2>/dev/null
cd "$HOME/Documents/Vision_Live" || exit 1
if ! "$HOME/anaconda3/envs/ai_master/bin/python" "$HOME/Documents/Vision_Live/lancer_vision.py" >> "$LOG" 2>&1; then
    notify-send -u critical -i dialog-error "Vision Live" "Erreur au lancement — voir $LOG" 2>/dev/null
fi

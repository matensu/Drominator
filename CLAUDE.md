# Projet Drone FPV — detection-v3

## Stack
- Python 3.11, PyTorch, Ultralytics YOLO, stable-baselines3
- Panda3D pour le rendu sim
- OpenCV pour la capture webcam

## Conventions
- Pas de wildcards imports
- Type hints sur les fonctions publiques
- Format Black, lint Ruff
- Tous les nouveaux scripts acceptent `--help`

## Règles dures
- Ne JAMAIS lancer un entraînement PPO ou YOLO sans confirmation explicite
- Ne JAMAIS toucher à `detection-v2` (branche du collègue)
- Les poids de modèles vont dans `ai/models/` (gitignoré)
- Tests rapides uniquement (quelques epochs/steps) pour valider que le pipeline tourne

## Structure
- `pipeline.py` / `test_webcam.py` — pipelines vision standalone
- `ai/` — tout le RL (env, train, viewer)
- `sim/` — rendu Panda3D et alternatives
- `ai/pipeline_utils.py` — fonctions partagées détection + depth
"""
Entraînement PPO — drone couloir infini.
Lance : python -m ai.train  (depuis la racine du projet)

Le modèle est sauvegardé dans ai/models/ toutes les 50 000 steps
et au Ctrl+C.
"""
import os
import sys

# Assure que la racine du projet est dans le path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

N_ENVS      = 8       # envs parallèles (pas de Panda3D → peut être élevé)
TOTAL_STEPS = 3_000_000
SAVE_FREQ   = 50_000
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")


def _make_env(seed: int):
    def _init():
        from ai.drone_env import DroneCorridorEnv
        env = DroneCorridorEnv(seed=seed)
        return Monitor(env)
    return _init


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = SubprocVecEnv([_make_env(i) for i in range(N_ENVS)])

    # Reprend l'entraînement si un modèle final existe
    final_path = os.path.join(MODEL_DIR, "drone_ppo_final.zip")
    if os.path.exists(final_path):
        print(f"[TRAIN] Reprise depuis {final_path}")
        model = PPO.load(final_path, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps        = 2048,
            batch_size     = 256,
            n_epochs       = 10,
            gamma          = 0.995,
            gae_lambda     = 0.95,
            learning_rate  = 3e-4,
            ent_coef       = 0.005,
            clip_range     = 0.2,
            policy_kwargs  = dict(net_arch=[256, 256, 128]),
            verbose        = 1,
            tensorboard_log= os.path.join(os.path.dirname(__file__), "logs"),
        )

    ckpt_cb = CheckpointCallback(
        save_freq  = SAVE_FREQ // N_ENVS,
        save_path  = MODEL_DIR,
        name_prefix= "drone_ppo",
        verbose    = 1,
    )

    print(f"[TRAIN] {N_ENVS} envs | {TOTAL_STEPS:,} steps cibles | Ctrl+C pour sauvegarder")
    try:
        model.learn(
            total_timesteps   = TOTAL_STEPS,
            callback          = ckpt_cb,
            progress_bar      = True,
            reset_num_timesteps= False,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrompu — sauvegarde en cours...")

    model.save(final_path.replace(".zip", ""))
    env.close()
    print(f"[TRAIN] Modèle sauvegardé → {final_path}")


if __name__ == "__main__":
    main()

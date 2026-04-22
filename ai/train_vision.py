"""
Entraînement visuel — le modèle reçoit l'image RGB 84×84 rendue par Panda3D,
exactement comme la caméra physique du drone.

Architecture :
  Observation : image RGB 84×84 (CnnPolicy / NatureCNN)
  Action      : [avancer, strafe, monter/descendre, yaw] ∈ [-1, 1]
  Reward      : +mètres avancés  /  -10 collision

Lance :
    python -m ai.train_vision

IMPORTANT : loadPrcFileData DOIT être appelé avant tout import ShowBase.
Ce fichier est auto-contenu (pas d'import sim/corridor_sim.py).
"""

# ── Panda3D config globale ─────────────────────────────────────────────────────
# Doit être exécutée avant ShowBase, y compris dans les sous-processus.
from panda3d.core import loadPrcFileData
loadPrcFileData("", """
window-type offscreen
win-size 84 84
audio-library-name null
sync-video 0
framebuffer-multisample 0
""")

import os, sys, math, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from panda3d.core import (
    Texture, GraphicsOutput,
    AmbientLight, DirectionalLight,
    LColor, Fog, NodePath,
)
from direct.showbase.ShowBase import ShowBase  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.corridor import (
    CORRIDOR_W, CORRIDOR_H, CHUNK_LEN,
    gen_chunk, items_to_aabbs,
    apply_action, check_collision,
)

W_OBS, H_OBS = 84, 84
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_vision")
N_ENVS    = 2        # chaque process a sa propre instance Panda3D
TOTAL_STEPS = 3_000_000
SAVE_FREQ   = 50_000


# ── Renderer Panda3D inline ────────────────────────────────────────────────────

class _CorridorRenderer(ShowBase):
    """Rendu offscreen 84×84 — génère exactement l'image vue par le drone."""

    def __init__(self, seed: int = 0):
        ShowBase.__init__(self)
        self.disableMouse()

        self._tex = Texture("col")
        self.win.addRenderTexture(self._tex, GraphicsOutput.RTMCopyRam)

        self.camLens.setFov(90)
        self.camLens.setNearFar(0.05, 80.0)

        fog = Fog("f")
        fog.setColor(0.07, 0.07, 0.09)
        fog.setExpDensity(0.022)
        self.render.setFog(fog)
        self.win.setClearColor(LColor(0.07, 0.07, 0.09, 1))

        self._setup_lights()
        self._build_walls()
        self._chunk_nodes: dict[int, list] = {}

    def sync(self, pos: list, yaw: float, chunks: dict):
        """Synchronise la caméra et les obstacles avec l'état de l'env."""
        env_cis = set(chunks.keys())
        sim_cis = set(self._chunk_nodes.keys())

        for ci in sim_cis - env_cis:
            for n in self._chunk_nodes[ci]:
                n.removeNode()
            del self._chunk_nodes[ci]

        for ci in env_cis - sim_cis:
            nodes = [self._box(ox, oy, oz, w, d, h, r, g, b)
                     for (ox, oy, oz, w, d, h, r, g, b) in chunks[ci]]
            self._chunk_nodes[ci] = nodes

        self.camera.setPos(*pos)
        self.camera.setHpr(yaw, 0, 0)

    def get_frame(self) -> np.ndarray:
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()
        raw = self._tex.getRamImageAs("RGB")
        if raw:
            arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(H_OBS, W_OBS, 3)
            return np.ascontiguousarray(np.flipud(arr[:, :, ::-1]))
        return np.zeros((H_OBS, W_OBS, 3), dtype=np.uint8)

    def _build_walls(self):
        L  = 4000.0
        hw = CORRIDOR_W / 2
        B  = self._box
        B(0, L/2, -0.05,          CORRIDOR_W+.4, L, 0.1,  0.35, 0.33, 0.30)
        B(0, L/2, CORRIDOR_H+.05, CORRIDOR_W+.4, L, 0.1,  0.25, 0.25, 0.28)
        B(-(hw+.1), L/2, CORRIDOR_H/2, 0.2, L, CORRIDOR_H+.2, 0.40, 0.38, 0.43)
        B( (hw+.1), L/2, CORRIDOR_H/2, 0.2, L, CORRIDOR_H+.2, 0.40, 0.38, 0.43)
        B(0, -0.1, CORRIDOR_H/2, CORRIDOR_W+.4, 0.2, CORRIDOR_H+.2, 0.48, 0.46, 0.50)

    def _setup_lights(self):
        amb = AmbientLight("a")
        amb.setColor(LColor(0.42, 0.42, 0.46, 1))
        self.render.setLight(self.render.attachNewNode(amb))
        sun = DirectionalLight("s")
        sun.setColor(LColor(0.72, 0.68, 0.58, 1))
        sn = self.render.attachNewNode(sun)
        sn.setHpr(35, -50, 0)
        self.render.setLight(sn)

    def _box(self, x, y, z, sx, sy, sz, r, g, b) -> NodePath:
        n = self.loader.loadModel("models/misc/rgbCube")
        n.setScale(sx, sy, sz)
        n.setPos(x, y, z)
        n.setColor(r, g, b, 1)
        n.reparentTo(self.render)
        return n


# ── Environnement visuel ────────────────────────────────────────────────────────

class DroneVisionEnv(gym.Env):
    """
    Observation = image RGB 84×84 rendue en temps réel par Panda3D.
    Le modèle apprend à naviguer depuis l'image brute, comme avec une vraie caméra.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0):
        super().__init__()
        self._seed    = seed
        self._renderer = _CorridorRenderer(seed)

        # CnnPolicy (NatureCNN) attend (H, W, C) uint8
        self.observation_space = spaces.Box(0, 255, (H_OBS, W_OBS, 3), np.uint8)
        self.action_space      = spaces.Box(-1.0, 1.0, (4,), np.float32)

        self._rng      = random.Random(seed)
        self._pos      = [0.0, 1.0, 1.5]
        self._yaw      = 0.0
        self._prev_y   = 1.0
        self._chunks:  dict = {}
        self._aabbs:   dict = {}
        self._done_ci: set  = set()

    # ── Gymnasium ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        import time as _t
        s = seed if seed is not None else int(_t.time() * 1e6) % (2**31)
        self._rng     = random.Random(s)
        self._pos     = [0.0, 1.0, 1.5]
        self._yaw     = 0.0
        self._prev_y  = 1.0
        self._chunks.clear()
        self._aabbs.clear()
        self._done_ci.clear()
        self._ensure_chunks()
        self._renderer.sync(self._pos, self._yaw, self._chunks)
        return self._renderer.get_frame(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._pos, self._yaw = apply_action(self._pos, self._yaw, action)
        self._ensure_chunks()
        self._cleanup_chunks()

        collision = check_collision(self._pos, self._aabbs)
        dy        = max(0.0, self._pos[1] - self._prev_y)
        self._prev_y = self._pos[1]

        reward = -10.0 if collision else dy

        self._renderer.sync(self._pos, self._yaw, self._chunks)
        obs = self._renderer.get_frame()

        return obs, float(reward), collision, False, {}

    def render(self):
        return self._renderer.get_frame()

    # ── Chunks ────────────────────────────────────────────────────────────────

    def _ensure_chunks(self):
        ci0 = max(0, int(self._pos[1] / CHUNK_LEN))
        for ci in range(ci0, ci0 + 4):
            if ci not in self._done_ci:
                items = gen_chunk(ci, self._rng)
                self._chunks[ci] = items
                self._aabbs[ci]  = items_to_aabbs(items)
                self._done_ci.add(ci)

    def _cleanup_chunks(self):
        ci_now = int(self._pos[1] / CHUNK_LEN)
        for ci in [c for c in list(self._chunks) if c < ci_now - 1]:
            del self._chunks[ci]
            del self._aabbs[ci]


# ── Entraînement ───────────────────────────────────────────────────────────────

def _make_env(seed: int):
    def _init():
        env = DroneVisionEnv(seed=seed)
        return Monitor(env)
    return _init


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # SubprocVecEnv : chaque process a sa propre instance Panda3D
    env = SubprocVecEnv([_make_env(i) for i in range(N_ENVS)])

    final_path = os.path.join(MODEL_DIR, "drone_vision_final.zip")
    if os.path.exists(final_path):
        print(f"[TRAIN] Reprise : {final_path}")
        model = PPO.load(final_path, env=env)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            n_steps       = 512,
            batch_size    = 64,
            n_epochs      = 4,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            learning_rate = 2.5e-4,
            ent_coef      = 0.01,
            clip_range    = 0.1,
            verbose       = 1,
            device        = "cuda",
            tensorboard_log = os.path.join(os.path.dirname(__file__), "logs_vision"),
        )

    ckpt_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ // N_ENVS,
        save_path   = MODEL_DIR,
        name_prefix = "drone_vision",
        verbose     = 1,
    )

    print(f"[TRAIN] Vision CNN | {N_ENVS} envs Panda3D | {TOTAL_STEPS:,} steps | Ctrl+C pour sauvegarder")
    try:
        model.learn(
            total_timesteps    = TOTAL_STEPS,
            callback           = ckpt_cb,
            progress_bar       = True,
            reset_num_timesteps= False,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrompu — sauvegarde…")

    model.save(final_path.replace(".zip", ""))
    env.close()
    print(f"[TRAIN] Modèle sauvegardé → {final_path}")


if __name__ == "__main__":
    main()

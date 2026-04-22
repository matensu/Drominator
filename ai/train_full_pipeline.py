"""
Entraînement sur le pipeline complet de test_webcam.py.

Le modèle voit exactement ce que produit le programme :
  - Frame Panda3D du couloir (caméra simulée)
  - Passée dans YOLO  → boîtes de détection dessinées
  - Passée dans Depth Anything V2 → colormap de profondeur en overlay
  = Image annotée 84×84, identique à ce que verrait le drone réel.

Lance : python -m ai.train_full_pipeline
"""

# ── Panda3D : doit être avant tout import ShowBase ────────────────────────────
from panda3d.core import loadPrcFileData
loadPrcFileData("", """
window-type offscreen
win-size 320 240
audio-library-name null
sync-video 0
framebuffer-multisample 0
""")

import os, sys, math, random
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from panda3d.core import (
    Texture, GraphicsOutput,
    AmbientLight, DirectionalLight,
    LColor, Fog, NodePath,
)
from direct.showbase.ShowBase import ShowBase  # noqa: E402

from ai.corridor import (
    CORRIDOR_W, CORRIDOR_H, CHUNK_LEN,
    gen_chunk, items_to_aabbs,
    apply_action, check_collision, DT,
)

# ── Fonctions pipeline (module isolé, n'importe pas test_webcam.py) ──────────
from ai.pipeline_utils import (
    load_depth_model, infer_depth, depth_colormap,
    estimate_distance, danger_color, compute_focal,
    detect_unknown_obstacles,
    CONF_THRESHOLD, INFER_SIZE, NMS_IOU, FOV_WIDE,
    MIN_BBOX_PX, MIN_UNKNOWN_AREA,
)
from ultralytics import YOLO

OBS_W, OBS_H  = 320, 240      # taille obs pour le CNN
RENDER_W      = 320          # résolution rendu Panda3D (YOLO a besoin de détails)
RENDER_H      = 240
DEPTH_EVERY   = 3            # depth Anything tous les N steps (lent ~30ms)
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "models_pipeline")
TOTAL_STEPS   = 2_000_000
SAVE_FREQ     = 20_000
_POOL_SIZE    = 80           # max obstacles simultanés (4 chunks × ~15)


# ── Renderer Panda3D inline ───────────────────────────────────────────────────

class _CorridorRenderer(ShowBase):
    W, H = RENDER_W, RENDER_H

    def __init__(self):
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

        # Pool pré-alloué — jamais créé/détruit après init
        self._pool = []
        for _ in range(_POOL_SIZE):
            n = self.loader.loadModel("models/misc/rgbCube")
            n.reparentTo(self.render)
            n.hide()
            self._pool.append(n)
        self._used = 0

    def sync(self, pos: list, yaw: float, chunks: dict):
        items = []
        for chunk_items in chunks.values():
            items.extend(chunk_items)

        n_visible = min(len(items), _POOL_SIZE)
        for i in range(n_visible):
            ox, oy, oz, w, d, h, r, g, b = items[i]
            n = self._pool[i]
            n.setPos(ox, oy, oz)
            n.setScale(w, d, h)
            n.setColor(r, g, b, 1)
            n.show()
        for i in range(n_visible, self._used):
            self._pool[i].hide()
        self._used = n_visible

        self.camera.setPos(*pos)
        self.camera.setHpr(yaw, 0, 0)

    def get_frame(self) -> np.ndarray:
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()
        raw = self._tex.getRamImageAs("RGB")
        if raw:
            arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(self.H, self.W, 3)
            return np.ascontiguousarray(np.flipud(arr[:, :, ::-1]))
        return np.zeros((self.H, self.W, 3), dtype=np.uint8)

    def _build_walls(self):
        L = 4000.0; hw = CORRIDOR_W / 2; B = self._box
        B(0, L/2, -0.05,           CORRIDOR_W+.4, L, 0.1,  0.35,0.33,0.30)
        B(0, L/2,  CORRIDOR_H+.05, CORRIDOR_W+.4, L, 0.1,  0.25,0.25,0.28)
        B(-(hw+.1), L/2, CORRIDOR_H/2, 0.2, L, CORRIDOR_H+.2, 0.40,0.38,0.43)
        B( (hw+.1), L/2, CORRIDOR_H/2, 0.2, L, CORRIDOR_H+.2, 0.40,0.38,0.43)
        B(0, -0.1,  CORRIDOR_H/2, CORRIDOR_W+.4, 0.2, CORRIDOR_H+.2, 0.48,0.46,0.50)

    def _setup_lights(self):
        amb = AmbientLight("a"); amb.setColor(LColor(.42,.42,.46,1))
        self.render.setLight(self.render.attachNewNode(amb))
        sun = DirectionalLight("s"); sun.setColor(LColor(.72,.68,.58,1))
        sn = self.render.attachNewNode(sun); sn.setHpr(35,-50,0)
        self.render.setLight(sn)

    def _box(self, x, y, z, sx, sy, sz, r, g, b) -> NodePath:
        n = self.loader.loadModel("models/misc/rgbCube")
        n.setScale(sx, sy, sz); n.setPos(x, y, z); n.setColor(r, g, b, 1)
        n.reparentTo(self.render); return n


# ── Environnement pipeline complet ───────────────────────────────────────────

class FullPipelineEnv(gym.Env):
    """
    Observation = image 84×84 produite par le pipeline test_webcam.py :
      frame Panda3D → YOLO → Depth Anything → image annotée.
    Identique à ce que voit le drone en conditions réelles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0):
        super().__init__()
        self._seed = seed

        # ── Modèles de perception ─────────────────────────────────────────
        yolo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "yolo11n.pt"
        )
        print("[ENV] Chargement YOLO...")
        self._yolo = YOLO(yolo_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ENV] Chargement Depth Anything V2 sur {device}...")
        self._depth_ok = load_depth_model(device)

        # ── Renderer Panda3D ──────────────────────────────────────────────
        self._renderer = _CorridorRenderer()

        # ── Espaces Gym ───────────────────────────────────────────────────
        self.observation_space = spaces.Box(0, 255, (OBS_H, OBS_W, 3), np.uint8)
        self.action_space      = spaces.Box(-1.0, 1.0, (4,), np.float32)

        # ── État interne ──────────────────────────────────────────────────
        self._rng      = random.Random(seed)
        self._pos      = [0.0, 1.0, 1.5]
        self._yaw      = 0.0
        self._max_y    = 1.0
        self._timer    = 5.0
        self._passed:  set  = set()
        self._chunks:  dict = {}
        self._aabbs:   dict = {}
        self._done_ci: set  = set()

        # Cache depth (recalculé tous les DEPTH_EVERY steps)
        self._depth_cache = None
        self._step_n      = 0

        # Focale caméra
        self._focal = compute_focal(RENDER_W, FOV_WIDE)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # Seed aléatoire à chaque épisode → couloir différent à chaque fois
        import time as _time
        s = seed if seed is not None else int(_time.time() * 1e6) % (2**31)
        self._rng      = random.Random(s)
        self._pos      = [0.0, 1.0, 1.5]
        self._yaw      = 0.0
        self._max_y    = 1.0
        self._timer    = 5.0
        self._passed   = set()
        self._depth_cache = None
        self._step_n   = 0
        self._chunks.clear(); self._aabbs.clear(); self._done_ci.clear()
        self._ensure_chunks()
        self._renderer.sync(self._pos, self._yaw, self._chunks)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._pos, self._yaw = apply_action(self._pos, self._yaw, action)
        self._ensure_chunks()
        self._cleanup_chunks()

        new_max = max(self._max_y, self._pos[1])
        reward  = max(0.0, new_max - self._max_y)
        self._max_y = new_max

        self._update_timer()
        self._timer -= DT

        collision = check_collision(self._pos, self._aabbs)
        time_up   = self._timer <= 0
        terminated = collision or time_up

        if collision:
            reward = -10.0

        self._renderer.sync(self._pos, self._yaw, self._chunks)
        obs = self._get_obs()

        return obs, float(reward), terminated, False, {}

    def render(self):
        return self._last_obs if hasattr(self, "_last_obs") else None

    def _update_timer(self):
        py = self._pos[1]
        for ci, aabbs in self._aabbs.items():
            for i, (x0, x1, y0, y1, z0, z1) in enumerate(aabbs):
                key = (ci, i)
                if key not in self._passed and py > (y0 + y1) / 2 + 0.5:
                    self._passed.add(key)
                    bonus = max(0.15, 1.0 - py / 150.0)
                    self._timer += bonus

    # ── Pipeline de perception (= test_webcam.py) ─────────────────────────────

    def _get_obs(self) -> np.ndarray:
        # 1. Rendu Panda3D — vue caméra brute du couloir
        frame = self._renderer.get_frame()       # 320×240 BGR
        annotated = frame.copy()
        fh, fw = frame.shape[:2]

        # 2. YOLO — détection des obstacles
        yolo_boxes  = []
        calib_pts   = []
        results = self._yolo(
            frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD,
            iou=NMS_IOU, verbose=False
        )[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls  = self._yolo.names[int(box.cls[0])]
            conf = float(box.conf[0])
            dist, _ = estimate_distance(cls, y2 - y1, fh, self._focal, self._focal)
            color = danger_color(dist)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cls} {conf:.0%}"
            if dist:
                lbl += f" ~{dist}m"
                calib_pts.append(((x1+x2)/2, (y1+y2)/2, dist))
            cv2.putText(annotated, lbl, (x1+2, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            yolo_boxes.append((x1, y1, x2, y2))

        # 3. Depth Anything V2 — recalculé tous les DEPTH_EVERY steps
        if self._depth_ok and self._step_n % DEPTH_EVERY == 0:
            self._depth_cache = infer_depth(frame)
        self._step_n += 1

        # 4. Overlay depth — même rendu que test_webcam.py
        if self._depth_cache is not None:
            depth_col = depth_colormap(self._depth_cache)   # BGR colormap
            # Obstacles inconnus détectés par depth
            unknown = detect_unknown_obstacles(
                self._depth_cache, yolo_boxes, frame.shape)
            for (x1, y1, x2, y2) in unknown:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (160, 100, 255), 1)
                cv2.putText(annotated, "?", (x1+2, max(y1-4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 100, 255), 1)
            # Blend depth 30% sur la frame (même visuel que test_webcam)
            annotated = cv2.addWeighted(annotated, 0.72, depth_col, 0.28, 0)

        # 5. Resize 84×84 → observation pour le CNN
        obs = cv2.resize(annotated, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
        self._last_obs = obs
        return obs

    # ── Gestion des chunks ────────────────────────────────────────────────────

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
            del self._chunks[ci]; del self._aabbs[ci]


# ── Callback stats reward ─────────────────────────────────────────────────────

from stable_baselines3.common.callbacks import BaseCallback

class RewardStatsCallback(BaseCallback):
    """
    Affiche toutes les PRINT_EVERY étapes :
      - reward min / max / moyenne  sur la fenêtre
      - meilleure reward vue depuis le début
      - barre ASCII proportionnelle à la moyenne
    """
    PRINT_EVERY = 1000
    BAR_WIDTH   = 30

    def __init__(self):
        super().__init__()
        self._ep_rewards:  list[float] = []
        self._cur_reward:  float       = 0.0
        self._best_ever:   float       = -float("inf")
        self._window:      list[float] = []   # rewards des PRINT_EVERY derniers épisodes

    def _on_step(self) -> bool:
        # Accumule la reward du step courant
        reward = float(self.locals["rewards"][0])
        done   = bool(self.locals["dones"][0])
        self._cur_reward += reward

        if done:
            self._window.append(self._cur_reward)
            if self._cur_reward > self._best_ever:
                self._best_ever = self._cur_reward
            self._cur_reward = 0.0

        if self.n_calls % self.PRINT_EVERY == 0 and self._window:
            mn  = min(self._window)
            mx  = max(self._window)
            avg = sum(self._window) / len(self._window)

            # Barre ASCII proportionnelle à la moyenne (0–50m = plein)
            filled = int(max(0.0, min(1.0, avg / 50.0)) * self.BAR_WIDTH)
            bar    = "█" * filled + "░" * (self.BAR_WIDTH - filled)

            print(
                f"\n{'─'*55}\n"
                f"  Steps     : {self.num_timesteps:>10,}\n"
                f"  Épisodes  : {len(self._window):>10,}\n"
                f"  Reward min: {mn:>10.1f} m\n"
                f"  Reward moy: {avg:>10.1f} m   [{bar}]\n"
                f"  Reward max: {mx:>10.1f} m\n"
                f"  Record    : {self._best_ever:>10.1f} m\n"
                f"{'─'*55}"
            )
            self._window.clear()   # repart à zéro pour la prochaine fenêtre

        return True


# ── Entraînement ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # n_envs=1 : YOLO + Depth partagent le GPU, pas de parallélisme possible
    env = DummyVecEnv([lambda: Monitor(FullPipelineEnv(seed=0))])

    final_path = os.path.join(MODEL_DIR, "drone_pipeline_final.zip")
    if os.path.exists(final_path):
        print(f"[TRAIN] Reprise : {final_path}")
        model = PPO.load(final_path, env=env)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            n_steps       = 256,
            batch_size    = 32,
            n_epochs      = 4,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            learning_rate = 2.5e-4,
            ent_coef      = 0.01,
            clip_range    = 0.1,
            verbose       = 1,
            device        = "cuda",
            tensorboard_log = os.path.join(os.path.dirname(__file__), "logs_pipeline"),
        )

    from stable_baselines3.common.callbacks import CallbackList
    ckpt_cb  = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = "drone_pipeline",
        verbose     = 1,
    )
    stats_cb = RewardStatsCallback()

    print("[TRAIN] Pipeline complet (YOLO + Depth Anything) | 1 env | Ctrl+C pour sauvegarder")
    print(f"[TRAIN] Observation : frame 84×84 annotée — même chose que test_webcam.py")
    try:
        model.learn(
            total_timesteps    = TOTAL_STEPS,
            callback           = CallbackList([ckpt_cb, stats_cb]),
            progress_bar       = True,
            reset_num_timesteps= False,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrompu — sauvegarde…")

    model.save(final_path.replace(".zip", ""))
    env.close()
    print(f"[TRAIN] Sauvegardé → {final_path}")


if __name__ == "__main__":
    main()

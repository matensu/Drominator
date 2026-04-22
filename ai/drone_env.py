"""
Environnement Gymnasium — drone couloir infini.
Observation : 48 raycasts analytiques + 5 variables d'état = 53 floats.
Action      : [avancer, strafe, monter/descendre, yaw] ∈ [-1, 1].
Reward      : +mètres_avancés  /  -10 collision.
Aucune dépendance Panda3D → peut tourner en n_envs parallèles.
"""
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ai.corridor import (
    CORRIDOR_W, CORRIDOR_H, CHUNK_LEN, N_RAYS,
    gen_chunk, items_to_aabbs,
    apply_action, check_collision, raycast,
)


class DroneCorridorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self._base_seed = seed

        # spaces
        obs_dim = N_RAYS + 5
        self.observation_space = spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (4,),      np.float32)

        # état interne
        self._rng:     random.Random   = random.Random(seed)
        self._pos:     list[float]     = [0.0, 1.0, 1.5]
        self._yaw:     float           = 0.0
        self._max_y:     float           = 1.0
        self._timer:     float           = 5.0
        self._passed:    set             = set()
        self._chunks:  dict[int, list]   = {}
        self._aabbs:   dict[int, list]   = {}
        self._done_ci: set[int]          = set()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        import time as _t
        s = seed if seed is not None else int(_t.time() * 1e6) % (2**31)
        self._rng    = random.Random(s)
        self._pos    = [0.0, 1.0, 1.5]
        self._yaw    = 0.0
        self._max_y  = 1.0
        self._timer  = 5.0
        self._passed = set()
        self._chunks.clear()
        self._aabbs.clear()
        self._done_ci.clear()
        self._ensure_chunks()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._pos, self._yaw = apply_action(self._pos, self._yaw, action)
        self._ensure_chunks()
        self._cleanup_chunks()

        # Reward : uniquement quand on atteint une nouvelle distance max
        new_max = max(self._max_y, self._pos[1])
        reward  = max(0.0, new_max - self._max_y)
        self._max_y = new_max

        # Bonus timer par obstacle dépassé (décroît avec la distance)
        self._update_timer()
        self._timer -= DT

        collision  = check_collision(self._pos, self._aabbs)
        time_up    = self._timer <= 0
        terminated = collision or time_up

        if collision:
            reward = -10.0

        return self._obs(), float(reward), terminated, False, {}

    def _update_timer(self):
        """Ajoute du temps pour chaque obstacle nouvellement dépassé."""
        py = self._pos[1]
        for ci, aabbs in self._aabbs.items():
            for i, (x0, x1, y0, y1, z0, z1) in enumerate(aabbs):
                key = (ci, i)
                if key not in self._passed and py > (y0 + y1) / 2 + 0.5:
                    self._passed.add(key)
                    # Bonus diminue avec la distance : 1s → 0.15s
                    bonus = max(0.15, 1.0 - py / 150.0)
                    self._timer += bonus

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        rays  = raycast(self._pos, self._yaw, self._aabbs)
        hr    = math.radians(self._yaw)
        state = np.array([
            self._pos[0] / (CORRIDOR_W / 2),          # x normalisé [-1,1]
            self._pos[2] / CORRIDOR_H,                 # z normalisé [0,1]
            math.sin(hr),                              # orientation sin
            math.cos(hr),                              # orientation cos
            min(self._pos[1] / 200.0, 1.0),           # progression (0→200m)
        ], dtype=np.float32)
        return np.concatenate([rays, state])

    # ── Génération du couloir ─────────────────────────────────────────────────

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

    # ── Accès état (pour la visualisation Panda3D) ────────────────────────────

    @property
    def pos(self)    -> list: return self._pos
    @property
    def yaw(self)    -> float: return self._yaw
    @property
    def chunks(self) -> dict:  return self._chunks
    @property
    def aabbs(self)  -> dict:  return self._aabbs

"""
Visualisation Panda3D du couloir infini.
Synchronisé avec DroneCorridorEnv pour afficher exactement ce que le modèle voit.

Approche pool : les NodePaths sont pré-alloués au démarrage et ne sont jamais
créés/détruits (show/hide + setPos/setScale à la place). Cela évite les
problèmes de synchronisation entre removeNode() et renderFrame().
"""
import numpy as np

from panda3d.core import (
    loadPrcFileData, Texture, GraphicsOutput,
    AmbientLight, DirectionalLight,
    LColor, NodePath, Fog,
)

loadPrcFileData("", """
window-type offscreen
win-size 640 480
audio-library-name null
sync-video 0
framebuffer-multisample 0
""")

from direct.showbase.ShowBase import ShowBase  # noqa: E402

from ai.corridor import CORRIDOR_W, CORRIDOR_H

# Nombre max d'obstacles simultanément visibles (4 chunks × ~15 obstacles)
_POOL_SIZE = 80


class CorridorSim(ShowBase):
    W, H = 640, 480

    def __init__(self, env):
        ShowBase.__init__(self)
        self.disableMouse()

        self._env = env

        # Texture couleur offscreen
        self._tex = Texture("col")
        self.win.addRenderTexture(self._tex, GraphicsOutput.RTMCopyRam)

        # Caméra
        self.camLens.setFov(90)
        self.camLens.setNearFar(0.05, 80.0)

        # Brouillard
        fog = Fog("fog")
        fog.setColor(0.08, 0.08, 0.10)
        fog.setExpDensity(0.018)
        self.render.setFog(fog)
        self.win.setClearColor(LColor(0.08, 0.08, 0.10, 1))

        self._setup_lights()
        self._build_static()

        # Pool de nodes pré-alloués — jamais créés/détruits après init
        self._pool: list[NodePath] = []
        for _ in range(_POOL_SIZE):
            n = self.loader.loadModel("models/misc/rgbCube")
            n.reparentTo(self.render)
            n.hide()
            self._pool.append(n)
        self._used = 0  # nombre de slots actifs lors du dernier sync

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def sync_and_render(self) -> np.ndarray:
        """Synchronise la scène avec l'état de l'env puis rend une frame BGR."""
        self._sync_obstacles()
        self._apply_camera()
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()

        raw = self._tex.getRamImageAs("RGB")
        if raw:
            arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(self.H, self.W, 3)
            return np.ascontiguousarray(np.flipud(arr[:, :, ::-1]))
        return np.zeros((self.H, self.W, 3), dtype=np.uint8)

    # ── Sync obstacles ────────────────────────────────────────────────────────

    def _sync_obstacles(self):
        """
        Aplatit tous les obstacles de l'env dans une liste, puis met à jour
        les nodes du pool en place. Aucune création/destruction de NodePath.
        """
        items = []
        for chunk_items in self._env.chunks.values():
            items.extend(chunk_items)

        n_visible = min(len(items), _POOL_SIZE)

        for i in range(n_visible):
            ox, oy, oz, w, d, h, r, g, b = items[i]
            node = self._pool[i]
            node.setPos(ox, oy, oz)
            node.setScale(w, d, h)
            node.setColor(r, g, b, 1)
            node.show()

        for i in range(n_visible, self._used):
            self._pool[i].hide()

        self._used = n_visible

    def clear_chunks(self):
        """Cache tous les obstacles (compatibilité avec le viewer)."""
        for i in range(self._used):
            self._pool[i].hide()
        self._used = 0

    def _apply_camera(self):
        p = self._env.pos
        self.camera.setPos(p[0], p[1], p[2])
        self.camera.setHpr(self._env.yaw, 0, 0)

    # ── Scène statique (murs permanents) ──────────────────────────────────────

    def _build_static(self):
        L  = 4000.0
        hw = CORRIDOR_W / 2
        B  = self._box
        B(0,        L/2,  -0.05,          CORRIDOR_W+.4, L,   0.1,           0.35, 0.33, 0.30)
        B(0,        L/2,  CORRIDOR_H+.05, CORRIDOR_W+.4, L,   0.1,           0.25, 0.25, 0.28)
        B(-(hw+.1), L/2,  CORRIDOR_H/2,   0.2,           L,   CORRIDOR_H+.2, 0.40, 0.38, 0.43)
        B( (hw+.1), L/2,  CORRIDOR_H/2,   0.2,           L,   CORRIDOR_H+.2, 0.40, 0.38, 0.43)
        B(0,       -0.1,  CORRIDOR_H/2,   CORRIDOR_W+.4, 0.2, CORRIDOR_H+.2, 0.48, 0.46, 0.50)

    def _setup_lights(self):
        amb = AmbientLight("amb")
        amb.setColor(LColor(0.42, 0.42, 0.46, 1))
        self.render.setLight(self.render.attachNewNode(amb))

        sun = DirectionalLight("sun")
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

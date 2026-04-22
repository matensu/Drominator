"""
Simulateur 3D Panda3D — rendu offscreen → numpy BGR.
Scène intérieure : salle, personnes, chaises, tables, caisses.
Contrôles : move(speed), strafe(speed), turn(dyaw, dpitch).
"""
import math
import numpy as np
import cv2

from panda3d.core import (
    loadPrcFileData, Texture, GraphicsOutput, PNMImage,
    AmbientLight, DirectionalLight, PointLight,
    LColor, Vec3, NodePath,
    AntialiasAttrib, TransparencyAttrib,
    Fog,
)

# ── DOIT être avant l'import de ShowBase ──────────────────────────────────────
loadPrcFileData("", """
window-type offscreen
win-size 640 480
audio-library-name null
framebuffer-multisample 1
multisamples 2
sync-video 0
""")

from direct.showbase.ShowBase import ShowBase  # noqa: E402


class PandaSim(ShowBase):
    W, H = 640, 480

    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()

        # ── Texture offscreen ─────────────────────────────────────────────
        self._tex = Texture("frame")
        self.win.addRenderTexture(self._tex, GraphicsOutput.RTMCopyRam)

        # ── Caméra ────────────────────────────────────────────────────────
        self._yaw   = 0.0
        self._pitch = 0.0
        self._pos   = Vec3(0, -4, 1.65)  # intérieur, regard vers +Y
        self.camLens.setFov(75)
        self._apply_camera()

        # ── Brouillard léger ──────────────────────────────────────────────
        fog = Fog("room_fog")
        fog.setColor(0.08, 0.08, 0.12)
        fog.setExpDensity(0.025)
        self.render.setFog(fog)
        self.win.setClearColor(LColor(0.08, 0.08, 0.12, 1))

        self._build_scene()

    # ── Contrôles ─────────────────────────────────────────────────────────────
    def move(self, speed: float):
        hr = math.radians(self._yaw)
        self._pos.x -= math.sin(hr) * speed
        self._pos.y += math.cos(hr) * speed
        self._clamp()
        self._apply_camera()

    def strafe(self, speed: float):
        hr = math.radians(self._yaw)
        self._pos.x += math.cos(hr) * speed
        self._pos.y += math.sin(hr) * speed
        self._clamp()
        self._apply_camera()

    def turn(self, dyaw: float, dpitch: float):
        self._yaw   += dyaw
        self._pitch  = max(-55.0, min(55.0, self._pitch + dpitch))
        self._apply_camera()

    def _clamp(self):
        self._pos.x = max(-10.2, min(10.2, self._pos.x))
        self._pos.y = max(-5.5,  min(9.2,  self._pos.y))

    def _apply_camera(self):
        self.camera.setPos(self._pos)
        self.camera.setHpr(self._yaw, self._pitch, 0)

    # ── Helpers scène ─────────────────────────────────────────────────────────
    def _box(self, x, y, z, sx, sy, sz, r, g, b):
        """Cube Panda3D mis à l'échelle et coloré."""
        n = self.loader.loadModel("models/misc/rgbCube")
        n.setScale(sx, sy, sz)
        n.setPos(x, y, z)
        n.setColor(r, g, b, 1)
        n.reparentTo(self.render)
        return n

    def _sphere(self, x, y, z, r_scale, r, g, b):
        n = self.loader.loadModel("models/misc/sphere")
        n.setScale(r_scale)
        n.setPos(x, y, z)
        n.setColor(r, g, b, 1)
        n.reparentTo(self.render)
        return n

    # ── Construction de la scène ──────────────────────────────────────────────
    def _build_scene(self):
        self.render.setAntialias(AntialiasAttrib.MAuto)

        # ── Éclairage ──────────────────────────────────────────────────────
        amb = AmbientLight("amb")
        amb.setColor(LColor(0.28, 0.27, 0.32, 1))
        self.render.setLight(self.render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(LColor(0.85, 0.78, 0.70, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(40, -55, 0)
        self.render.setLight(sun_np)

        for (lx, ly) in [(-4, 2), (4, -2), (0, -5)]:
            pl = PointLight(f"pl{lx}{ly}")
            pl.setColor(LColor(1.0, 0.88, 0.65, 1))
            pl.setAttenuation((0.6, 0, 0.04))
            pl_np = self.render.attachNewNode(pl)
            pl_np.setPos(lx, ly, 3.85)
            self.render.setLight(pl_np)

        B = self._box
        S = self._sphere

        # ── Salle (caméra à Y=-4, regarde vers +Y) ────────────────────────
        # Sol / Plafond
        B( 0,  4,  -0.05, 20, 28, 0.1,  0.42, 0.40, 0.38)
        B( 0,  4,   4.05, 20, 28, 0.1,  0.22, 0.22, 0.26)
        # Mur arrière (loin, +Y)
        B( 0,  9,   2.0,  20, 0.2, 4.2, 0.52, 0.50, 0.55)
        # Mur avant (derrière la caméra)
        B( 0, -5,   2.0,  20, 0.2, 4.2, 0.52, 0.50, 0.55)
        # Murs latéraux
        B(-10, 4,   2.0,  0.2, 28, 4.2, 0.45, 0.44, 0.48)
        B( 10, 4,   2.0,  0.2, 28, 4.2, 0.45, 0.44, 0.48)
        # Mur intérieur avec ouverture porte (Y=2)
        B(-7.2, 2,  2.0,  5.5, 0.25, 4.2, 0.48, 0.46, 0.52)
        B( 7.2, 2,  2.0,  5.5, 0.25, 4.2, 0.48, 0.46, 0.52)
        B( 0,   2,  3.6,  20,  0.25, 1.0, 0.48, 0.46, 0.52)

        # ── Personnes (devant la caméra, Y positif) ────────────────────────
        for (px, py) in [(-1.5, 2.5), (2.5, 4.5), (-4.0, 6.5)]:
            B(px, py,  0.9,  0.44, 0.28, 1.8,  0.20, 0.30, 0.72)
            S(px, py,  1.88, 0.18,             0.18, 0.25, 0.65)
            B(px-0.11, py, 0.35, 0.14, 0.25, 0.70, 0.15, 0.22, 0.60)
            B(px+0.11, py, 0.35, 0.14, 0.25, 0.70, 0.15, 0.22, 0.60)

        # ── Chaises ────────────────────────────────────────────────────────
        for (cx, cy) in [(-3.5, 3.5), (4, 2.5), (1.5, 6), (-2, 7.5)]:
            B(cx, cy,        0.45, 0.52, 0.52, 0.06, 0.32, 0.18, 0.10)
            B(cx, cy - 0.23, 0.73, 0.52, 0.06, 0.55, 0.28, 0.15, 0.08)
            for (lx, lz) in [(-0.2,-0.2),(-0.2,0.2),(0.2,-0.2),(0.2,0.2)]:
                B(cx+lx, cy+lz, 0.22, 0.05, 0.05, 0.44, 0.25, 0.12, 0.07)

        # ── Tables ─────────────────────────────────────────────────────────
        for (tx, ty) in [(-5.5, 5), (5.5, 7)]:
            B(tx, ty, 0.75, 1.4, 0.8, 0.06, 0.38, 0.22, 0.12)
            for (lx, lz) in [(-0.58,-0.32),(-0.58,0.32),(0.58,-0.32),(0.58,0.32)]:
                B(tx+lx, ty+lz, 0.37, 0.06, 0.06, 0.74, 0.30, 0.17, 0.09)

        # ── Caisses ────────────────────────────────────────────────────────
        B(-7.5, 3,  0.45, 0.9, 0.9, 0.9,  0.22, 0.50, 0.25)
        B(-7.5, 3,  1.25, 0.7, 0.7, 0.7,  0.20, 0.45, 0.22)
        B( 7.5, 5,  0.3,  1.1, 0.8, 0.6,  0.20, 0.45, 0.22)
        B( 7.5, 5,  0.9,  0.6, 0.6, 0.6,  0.20, 0.45, 0.22)

        # ── Grille au sol ──────────────────────────────────────────────────
        from panda3d.core import LineSegs
        ls = LineSegs()
        ls.setColor(0.35, 0.34, 0.36, 0.6)
        ls.setThickness(1)
        for i in range(-9, 10, 2):
            ls.moveTo(i, -5, 0.01); ls.drawTo(i, 9, 0.01)
        for j in range(-4, 10, 2):
            ls.moveTo(-9, j, 0.01); ls.drawTo(9, j, 0.01)
        self.render.attachNewNode(ls.create())

    # ── Rendu → numpy ─────────────────────────────────────────────────────────
    def get_frame(self) -> np.ndarray:
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()

        # Récupérer les pixels via PNMImage (compatible toutes versions)
        img = PNMImage()
        self.win.getDisplayRegion(0).getScreenshot(img)
        w, h = img.getXSize(), img.getYSize()

        # Lecture rapide via la texture GPU (RTMCopyRam)
        raw = self._tex.getRamImageAs("RGB")
        if raw:
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            if arr.size == self.H * self.W * 3:
                arr = arr.reshape(self.H, self.W, 3)
                bgr = arr[:, :, ::-1].copy()
                return np.ascontiguousarray(np.flipud(bgr))

        # Fallback : PNMImage vectorisé
        arr = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(h):
            row = img.getRow(y)
            for x in range(w):
                p = row.getPixel(x)
                arr[y, x] = [p.z, p.y, p.x]  # BGR
        return np.ascontiguousarray((arr * 255).astype(np.uint8))

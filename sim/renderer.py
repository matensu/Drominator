"""
Renderer 3D software minimaliste — numpy + OpenCV uniquement.
Projection perspective, rendu de polygones (painter's algorithm), éclairage Lambertien.
"""
import math
import numpy as np
import cv2

LIGHT = np.array([0.4, -0.9, 0.3], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)


class SoftRenderer:
    def __init__(self, W=640, H=480, fov_deg=75):
        self.W, self.H = W, H
        self.focal = W / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
        self.cx, self.cy = W / 2.0, H / 2.0

        # Camera
        self.pos   = np.array([0.0, 1.65, 8.0])
        self.yaw   = 0.0
        self.pitch = 0.0

        # Faces : (4×3 array de sommets world, couleur BGR)
        self._faces: list[tuple[np.ndarray, tuple]] = []
        self._build_scene()

    # ── Contrôles caméra ──────────────────────────────────────────────────────
    def move(self, speed: float):
        """Avancer (speed>0) ou reculer."""
        self.pos[0] += math.sin(self.yaw) * speed
        self.pos[2] -= math.cos(self.yaw) * speed
        self._clamp_pos()

    def strafe(self, speed: float):
        """Latéral droite (speed>0)."""
        self.pos[0] += math.cos(self.yaw) * speed
        self.pos[2] += math.sin(self.yaw) * speed
        self._clamp_pos()

    def turn(self, dyaw: float, dpitch: float):
        self.yaw   += dyaw
        self.pitch  = max(-1.2, min(1.2, self.pitch + dpitch))

    def _clamp_pos(self):
        self.pos[0] = max(-10.3, min(10.3, self.pos[0]))
        self.pos[2] = max(-5.5,  min(9.5,  self.pos[2]))
        self.pos[1] = 1.65

    # ── Construction de la scène ──────────────────────────────────────────────
    def _box(self, cx, cy, cz, sx, sy, sz, color):
        hw, hh, hd = sx/2, sy/2, sz/2
        V = np.array([
            [cx-hw, cy-hh, cz-hd], [cx+hw, cy-hh, cz-hd],
            [cx+hw, cy+hh, cz-hd], [cx-hw, cy+hh, cz-hd],
            [cx-hw, cy-hh, cz+hd], [cx+hw, cy-hh, cz+hd],
            [cx+hw, cy+hh, cz+hd], [cx-hw, cy+hh, cz+hd],
        ], dtype=np.float64)
        for fi in ([0,1,2,3],[5,4,7,6],[4,0,3,7],[1,5,6,2],[3,2,6,7],[4,5,1,0]):
            self._faces.append((V[fi], color))

    def _build_scene(self):
        # Sol / plafond / murs
        self._box( 0, -0.05,  0, 22, 0.1, 32, (90,  85,  80))
        self._box( 0,  4.05,  0, 22, 0.1, 32, (55,  52,  50))
        self._box( 0,  2.0,  -6, 22, 4.2, 0.2,(130,125,120))
        self._box(-11, 2.0,   0, 0.2, 4.2, 32,(110,105,100))
        self._box( 11, 2.0,   0, 0.2, 4.2, 32,(110,105,100))
        # Mur intérieur avec porte
        self._box(-7.2, 2.0, 2, 7.5, 4.2, 0.25,(115,110,105))
        self._box( 7.2, 2.0, 2, 7.5, 4.2, 0.25,(115,110,105))
        self._box( 0, 3.6,   2, 16,  1.0, 0.25,(115,110,105))
        # Personnes (corps + tête)
        for (px, pz) in [(-1.5, 3.5), (2.0, 1.0), (-3.5, -2.0)]:
            self._box(px, 0.9,   pz, 0.44, 1.8, 0.28, (50, 80, 200))  # corps
            self._box(px, 1.88, pz, 0.28, 0.28, 0.28, (40, 60, 160))  # tête
        # Chaises
        for (cx, cz) in [(-4, 5), (3.5, 4), (1, -3), (-2, -4.5)]:
            self._box(cx, 0.45, cz,  0.52, 0.06, 0.52, (40,  80, 130))
            self._box(cx, 0.73, cz-0.23, 0.52, 0.55, 0.06, (40, 80, 130))
        # Tables
        for (tx, tz) in [(-5, 0), (5, -3)]:
            self._box(tx, 0.75, tz, 1.4, 0.06, 0.8, (30, 100, 160))
        # Caisses
        self._box(-7,  0.45, 4,   0.9, 0.9, 0.9, (40, 140,  60))
        self._box(-7,  1.25, 4,   0.7, 0.7, 0.7, (40, 140,  60))
        self._box( 7.5, 0.3, 1,   1.1, 0.6, 0.8, (40, 140,  60))
        self._box( 7.8, 0.25,-1,  0.5, 0.5, 0.5, (40, 140,  60))

    # ── Rendu ─────────────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), (18, 15, 12), dtype=np.uint8)

        # Dégradé de fond (sol/ciel)
        horizon = self.H // 2
        for y in range(horizon):
            t = y / horizon
            img[y] = (int(25+t*15), int(20+t*12), int(18+t*10))

        # Matrices de rotation caméra
        cy, sy = math.cos(-self.yaw),   math.sin(-self.yaw)
        cp, sp = math.cos(-self.pitch), math.sin(-self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0],   [0, cp, -sp], [0, sp, cp]])
        R  = Rx @ Ry

        draw_list = []

        for (verts, color) in self._faces:
            cam = (verts - self.pos) @ R.T

            # Toute la face derrière la caméra → on saute
            if np.all(cam[:, 2] <= 0.1):
                continue

            avg_z = float(cam[:, 2].mean())
            if avg_z <= 0.1:
                continue

            # Back-face culling
            n_cam = np.cross(cam[1]-cam[0], cam[2]-cam[0])
            if n_cam[2] >= 0:
                continue

            # Projection perspective
            z = cam[:, 2].copy()
            z[z < 0.1] = 0.1
            px = (cam[:, 0] / z * self.focal + self.cx).astype(np.int32)
            py = (-cam[:, 1] / z * self.focal + self.cy).astype(np.int32)

            # Hors écran complet → skip
            if np.all(px < -10) or np.all(px > self.W+10):
                continue
            if np.all(py < -10) or np.all(py > self.H+10):
                continue

            # Éclairage Lambertien sur la normale monde
            wn = np.cross(verts[1]-verts[0], verts[2]-verts[0]).astype(np.float64)
            wn_norm = np.linalg.norm(wn)
            if wn_norm > 1e-9:
                wn /= wn_norm
                shade = 0.30 + max(0.0, float(np.dot(-wn, LIGHT))) * 0.70
            else:
                shade = 0.50

            # Brouillard progressif
            shade *= max(0.15, 1.0 - avg_z / 22.0)

            bgr = tuple(min(255, int(c * shade)) for c in color)
            pts = np.stack([px, py], axis=1)
            draw_list.append((avg_z, pts, bgr))

        # Painter's algorithm (arrière → avant)
        draw_list.sort(key=lambda x: -x[0])

        for (_, pts, bgr) in draw_list:
            p = pts.reshape(-1, 1, 2)
            cv2.fillPoly(img, [p], bgr)
            edge = tuple(min(255, c + 18) for c in bgr)
            cv2.polylines(img, [p], True, edge, 1, cv2.LINE_AA)

        # Overlay info
        cv2.putText(img, "SIM  WASD=move  <>^v=look",
                    (8, self.H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (160, 200, 160), 1)
        return img

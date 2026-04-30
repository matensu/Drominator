"""
Fake stereo depth — webcam ou simulateur Panda3D, pipeline VINS-Fusion.

Usage :
  python fake_stereo.py                 # webcam 0
  python fake_stereo.py --sim           # simulateur Panda3D
  python fake_stereo.py 1 90            # webcam 1, FOV 90°

Touches communes :
  q / Échap  → quitter
  + / -      → max_corners ±50
  r          → reset nuage 3D
  1/2/3      → vue face / dessus / côté
  wasd       → orbite caméra 3D

Touches simulateur (--sim) :
  ↑ ↓        → avancer / reculer
  ← →        → strafer gauche / droite
  e / c      → monter / descendre
  j / l      → tourner gauche / droite
"""

import sys
import os
import math
import collections
import numpy as np
import cv2

# Accès à pipeline_utils depuis le dossier parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ai.pipeline_utils import load_depth_model, infer_depth


# ─────────────────────────────────────────────────────────────────────────────
# Paire stéréo synthétique (visualisation uniquement)
# ─────────────────────────────────────────────────────────────────────────────

STRETCH_PCT  = 3
DEFAULT_FOV  = 120.0   # FOV typique caméra FPV grand angle (°)
DEFAULT_K1   = 0   # distorsion barillet typique caméra FPV

def make_stereo_pair(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w      = img.shape[:2]
    new_w     = w + int(w * STRETCH_PCT / 100)
    stretched = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
    return img, stretched[:, :w].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Tracker KLT (Shi-Tomasi + flux optique pyramidal)
# ─────────────────────────────────────────────────────────────────────────────

class KLTTracker:
    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self, max_corners: int = 300, min_dist: int = 25):
        self.max_corners = max_corners
        self.min_dist    = min_dist
        self.prev_gray   = None
        self.prev_pts    = None
        self._clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _enhance(self, g):
        return self._clahe.apply(g)

    def track(self, gray: np.ndarray) -> np.ndarray:
        """
        Retourne (N, 2) float32 — positions des features dans cette frame.
        """
        le = self._enhance(gray)

        # Suivi temporel
        tracked = np.empty((0, 1, 2), np.float32)
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts):
            fwd, sf, _  = cv2.calcOpticalFlowPyrLK(
                self._enhance(self.prev_gray), le, self.prev_pts, None, **self.LK)
            back, sb, _ = cv2.calcOpticalFlowPyrLK(le, self._enhance(self.prev_gray),
                                                    fwd, None, **self.LK)
            err  = np.abs(self.prev_pts - back).reshape(-1, 2).max(axis=1)
            ok   = (err < 1.0) & (sf.ravel() == 1) & (sb.ravel() == 1)
            tracked = fwd[ok]

        # Nouveaux coins dans les zones libres
        mask = np.full(gray.shape, 255, np.uint8)
        for pt in tracked.reshape(-1, 2).astype(int):
            cv2.circle(mask, tuple(pt), self.min_dist, 0, -1)
        needed  = max(0, self.max_corners - len(tracked))
        new_pts = cv2.goodFeaturesToTrack(le, maxCorners=needed, qualityLevel=0.01,
                                          minDistance=self.min_dist, mask=mask)
        new_pts = new_pts if new_pts is not None else np.empty((0, 1, 2), np.float32)

        all_pts = np.vstack([tracked, new_pts]) if len(tracked) else new_pts

        self.prev_gray = gray
        self.prev_pts  = all_pts
        return all_pts.reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion features 2D + depth map → points 3D
# ─────────────────────────────────────────────────────────────────────────────

def features_to_3d(pts: np.ndarray, depth_norm: np.ndarray,
                   K: np.ndarray, depth_scale: float = 5.0) -> np.ndarray:
    """
    pts        : (N, 2) positions pixel dans l'image gauche
    depth_norm : (H, W) float32 [0,1]  — 1 = proche, 0 = loin (Depth Anything)
    K          : matrice intrinsèque caméra
    depth_scale: profondeur max en unités arbitraires

    Retourne (N, 3) XYZ.
    """
    if len(pts) == 0:
        return np.empty((0, 3), np.float32)

    f  = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    h, w = depth_norm.shape

    ix = np.clip(pts[:, 0].astype(int), 0, w - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, h - 1)

    # Depth Anything: 1=proche → Z petit ; on veut Z grand pour objets lointains
    Z = (1.0 - depth_norm[iy, ix].astype(np.float32)) * depth_scale + 0.1

    X = (pts[:, 0] - cx) * Z / f
    Y = (pts[:, 1] - cy) * Z / f

    return np.stack([X, Y, Z], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Rendu 3D style VINS-Fusion
# ─────────────────────────────────────────────────────────────────────────────

class PointCloudRenderer:
    def __init__(self, canvas_w=1280, canvas_h=480, max_pts=8000):
        self.W    = canvas_w
        self.H    = canvas_h
        self.buf  = collections.deque(maxlen=max_pts)
        self.az   = 30.0
        self.el   = -20.0
        self.dist = 5.0
        self.f    = 200.0   # focale de rendu px

    def set_frame(self, pts3d: np.ndarray):
        valid = pts3d[np.isfinite(pts3d).all(axis=1) & (pts3d[:, 2] > 0)]
        self.buf.clear()
        if len(valid) == 0:
            return
        z_med = float(np.median(valid[:, 2]))
        if z_med > 0:
            valid = valid * (3.0 / z_med)
        for p in valid:
            self.buf.append(p)

    def reset(self):
        self.buf.clear()

    def _view_matrix(self):
        az = math.radians(self.az)
        el = math.radians(self.el)
        target = np.zeros(3)
        eye = target + np.array([
            self.dist * math.cos(el) * math.sin(az),
            self.dist * math.sin(el),
            self.dist * math.cos(el) * math.cos(az),
        ])
        fwd = target - eye;  fwd /= np.linalg.norm(fwd) + 1e-9
        up  = np.array([0., 1., 0.])
        right = np.cross(fwd, up);  right /= np.linalg.norm(right) + 1e-9
        up2   = np.cross(right, fwd)
        R = np.stack([right, -up2, fwd])
        return R, -R @ eye

    def _project(self, pts, R, t):
        p  = (R @ pts.T).T + t
        ok = p[:, 2] > 0.01
        px = np.full((len(pts), 2), -1.0)
        if ok.any():
            pv = p[ok]
            px[ok, 0] = self.f * pv[:, 0] / pv[:, 2] + self.W / 2
            px[ok, 1] = self.f * pv[:, 1] / pv[:, 2] + self.H / 2
        return px, p[:, 2]

    def render(self) -> np.ndarray:
        canvas = np.zeros((self.H, self.W, 3), np.uint8)

        if len(self.buf) < 2:
            cv2.putText(canvas, "En attente de points 3D...",
                        (self.W // 2 - 170, self.H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
            return canvas

        R, t = self._view_matrix()
        pts  = np.array(self.buf, np.float32)
        px, zb = self._project(pts, R, t)

        for i in np.argsort(-zb):
            x, y = int(px[i, 0]), int(px[i, 1])
            if 0 <= x < self.W and 0 <= y < self.H and zb[i] > 0:
                br = max(80, min(255, int(255 / (1 + zb[i] * 0.25))))
                cv2.circle(canvas, (x, y), 3, (0, br, 0), -1)

        # Marqueur rouge = caméra réelle
        cam_px, _ = self._project(np.zeros((1, 3), np.float32), R, t)
        cx, cy = int(cam_px[0, 0]), int(cam_px[0, 1])
        if 0 <= cx < self.W and 0 <= cy < self.H:
            cv2.ellipse(canvas, (cx, cy), (18, 10), 0, 0, 360, (0, 0, 200), -1)
            cv2.ellipse(canvas, (cx, cy), (18, 10), 0, 0, 360, (0, 0, 255),  1)

        def hud(msg, y):
            cv2.putText(canvas, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (50, 50, 50), 2, cv2.LINE_AA)
            cv2.putText(canvas, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (170, 170, 170), 1, cv2.LINE_AA)

        hud(f"points: {len(self.buf)}   az:{self.az:.0f}°  el:{self.el:.0f}°", 18)
        hud("wasd=orbite  1=face  2=dessus  3=côté  r=reset", 36)
        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Vue caméra gauche annotée
# ─────────────────────────────────────────────────────────────────────────────

def draw_left(img, pts, pts3d, n_total):
    out = img.copy()
    if len(pts3d) == 0 or len(pts) == 0:
        return out

    z     = pts3d[:, 2]
    valid = np.isfinite(z) & (z > 0)
    z_min = float(z[valid].min()) if valid.any() else 0
    z_max = float(z[valid].max()) if valid.any() else 1

    for i, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        if valid[i] and z_max > z_min:
            t_ = (z[i] - z_min) / (z_max - z_min)
            col = (0, int(255 * (1 - t_)), int(255 * t_))
        else:
            col = (80, 80, 80)
        cv2.circle(out, (x, y), 4, col, -1)

    def txt(msg, y):
        cv2.putText(out, msg, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, msg, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    txt(f"LEFT  feat:{n_total}", 22)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    use_sim   = "--sim" in sys.argv
    args      = [a for a in sys.argv[1:] if a != "--sim"]
    cam_index = int(args[0])   if len(args) > 0 else 0
    fov_deg   = float(args[1]) if len(args) > 1 else DEFAULT_FOV

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not load_depth_model(device):
        print("[ERREUR] Depth Anything V2 requis.")
        sys.exit(1)

    # ── Source de frames ──────────────────────────────────────────────────────
    sim = None
    cap = None
    if use_sim:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from sim.panda_sim import PandaSim
        sim = PandaSim()
        frame0 = sim.get_frame()
        print("=== Mode simulateur Panda3D ===")
        print("  ↑↓ avancer/reculer  ←→ strafer  e/c monter/descendre  j/l tourner")
    else:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"[ERREUR] Impossible d'ouvrir la caméra {cam_index}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame0 = cap.read()
        if not ret:
            sys.exit(1)
        print(f"=== Mode webcam {cam_index} ===")

    h, w   = frame0.shape[:2]

    def make_K(fov):
        f = (w / 2.0) / math.tan(math.radians(fov / 2.0))
        return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], np.float64)

    def apply_undistort(img, K_, k1):
        if abs(k1) < 1e-4:
            return img
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0])
        return cv2.undistort(img, K_, dist)

    K    = make_K(fov_deg)
    k1   = DEFAULT_K1   # distorsion radiale : <0 = barillet (FPV), >0 = coussinet

    tracker  = KLTTracker(max_corners=300)
    renderer = PointCloudRenderer(canvas_w=w * 2, canvas_h=h)

    depth_map   = np.zeros((h, w), np.float32)
    depth_every = 3
    frame_idx   = 0

    print(f"  FOV {fov_deg}°  k1={k1}  device {device}")
    print("  q=quitter  +/-=corners  r=reset  wasd=orbite  1/2/3=vues")
    print("  [ ]=FOV ±5°   o/p=distorsion k1 ±0.05")

    while True:
        # ── Acquisition frame ─────────────────────────────────────────────────
        if use_sim:
            frame = sim.get_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame = apply_undistort(frame, K, k1)
        left, right = make_stereo_pair(frame)
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        if frame_idx % depth_every == 0:
            depth_map = infer_depth(left)
        frame_idx += 1

        pts   = tracker.track(gray)
        pts3d = features_to_3d(pts, depth_map, K)
        renderer.set_frame(pts3d)

        # ── Vues ──────────────────────────────────────────────────────────────
        left_vis  = draw_left(left, pts, pts3d, len(pts))
        # HUD paramètres live
        def hud_img(img, msg, y):
            cv2.putText(img, msg, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, msg, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255,255,255), 1, cv2.LINE_AA)
        hud_img(left_vis, f"FOV:{fov_deg:.0f}°  k1:{k1:.2f}  [ ]=fov  o/p=dist", h - 8)

        right_vis = right.copy()
        cv2.putText(right_vis, f"RIGHT (stretch+crop {STRETCH_PCT}%)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        depth_color = cv2.applyColorMap(
            ((1.0 - depth_map) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        right_vis = cv2.addWeighted(right_vis, 0.6, depth_color, 0.4, 0)

        grid = np.vstack([np.hstack([left_vis, right_vis]), renderer.render()])
        cv2.imshow("VINS-style Depth", grid)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        # Orbite caméra 3D
        elif key == ord('w'):  renderer.el = min(renderer.el + 5, 85)
        elif key == ord('s'):  renderer.el = max(renderer.el - 5, -85)
        elif key == ord('a'):  renderer.az -= 5
        elif key == ord('d'):  renderer.az += 5
        elif key == ord('1'):  renderer.az, renderer.el = 0, 0
        elif key == ord('2'):  renderer.az, renderer.el = 0, -85
        elif key == ord('3'):  renderer.az, renderer.el = 90, -15
        elif key == ord('r'):  renderer.reset()
        elif key == ord('+'):  tracker.max_corners += 50
        elif key == ord('-'):  tracker.max_corners = max(50, tracker.max_corners - 50)
        elif key == ord(']'):
            fov_deg = min(fov_deg + 5, 170)
            K = make_K(fov_deg)
            print(f"  FOV → {fov_deg:.0f}°")
        elif key == ord('['):
            fov_deg = max(fov_deg - 5, 20)
            K = make_K(fov_deg)
            print(f"  FOV → {fov_deg:.0f}°")
        elif key == ord('p'):
            k1 = round(k1 - 0.05, 3)   # plus de correction barillet
            print(f"  k1 → {k1:.2f}")
        elif key == ord('o'):
            k1 = round(k1 + 0.05, 3)
            print(f"  k1 → {k1:.2f}")

        # Contrôles simulateur
        if use_sim:
            SPEED = 0.05
            if key == 82:   sim.move( SPEED)    # ↑
            elif key == 84: sim.move(-SPEED)    # ↓
            elif key == 81: sim.strafe(-SPEED)  # ←
            elif key == 83: sim.strafe( SPEED)  # →
            elif key == ord('e'): sim.turn(0,  2)   # monter
            elif key == ord('c'): sim.turn(0, -2)   # descendre
            elif key == ord('j'): sim.turn(-3, 0)   # tourner gauche
            elif key == ord('l'): sim.turn( 3, 0)   # tourner droite

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

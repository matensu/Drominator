"""
Vision combinée — toutes les vues en une seule fenêtre.

Usage :
  python vision.py                # webcam 0
  python vision.py --sim          # simulateur Panda3D
  python vision.py 2              # webcam index 2 (drone AV TO USB2.0)
  python vision.py 2 120          # webcam 2, FOV 120°

Disposition :
  ┌─────────────┬─────────────┬─────────────┐
  │  LEFT + KLT │  DEPTH map  │ RIGHT+obstas│
  ├─────────────┴─────────────┴─────────────┤
  │           Nuage 3D (VINS-style)         │
  └─────────────────────────────────────────┘

Touches :
  q / Échap  → quitter
  + / -      → max_corners ±50
  r          → reset nuage 3D
  1/2/3      → vue face / dessus / côté
  wasd       → orbite caméra 3D
  [ ]        → FOV ±5°
  o / p      → distorsion k1 ±0.05

Touches simulateur (--sim) :
  ↑ ↓        → avancer / reculer
  ← →        → strafer gauche / droite
  e / c      → monter / descendre
  j / l      → tourner gauche / droite
"""

import sys
import os
import math
import numpy as np
import cv2
import torch

# ── Imports internes ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from stereo.fake_stereo import (
    KLTTracker, PointCloudRenderer, features_to_3d, make_stereo_pair,
    draw_left, DEFAULT_FOV, DEFAULT_K1, STRETCH_PCT,
)
from ai.pipeline_utils import (
    load_depth_model, infer_depth, depth_colormap,
    detect_unknown_obstacles,
)


# ─────────────────────────────────────────────────────────────────────────────
# Panneau RIGHT — stretch+crop avec obstacles inconnus surlignés
# ─────────────────────────────────────────────────────────────────────────────

def draw_right(right_img: np.ndarray, depth_norm: np.ndarray,
               pts: np.ndarray) -> np.ndarray:
    yolo_boxes = []   # pas de détecteur externe — on passe liste vide
    unknown = detect_unknown_obstacles(depth_norm, yolo_boxes, right_img.shape)

    out = right_img.copy()
    depth_col = cv2.applyColorMap(
        ((1.0 - depth_norm) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(out, 0.55, depth_col, 0.45, 0)

    for (x1, y1, x2, y2) in unknown:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(out, "?", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1, cv2.LINE_AA)

    cv2.putText(out, f"RIGHT +depth +obstacles ({len(unknown)} det)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, f"RIGHT +depth +obstacles ({len(unknown)} det)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panneau DEPTH — Depth Anything colormap pur
# ─────────────────────────────────────────────────────────────────────────────

def draw_depth(depth_norm: np.ndarray, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    col = depth_colormap(depth_norm)
    cv2.putText(col, "DEPTH (Depth Anything V2)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(col, "DEPTH (Depth Anything V2)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return col


# ─────────────────────────────────────────────────────────────────────────────
# Assemblage de la grille finale
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(left_vis, depth_vis, right_vis, cloud_vis,
              target_w: int = 1280) -> np.ndarray:
    """
    Ligne 1 : LEFT | DEPTH | RIGHT  (chacun redimensionné à target_w//3)
    Ligne 2 : nuage 3D pleine largeur (target_w)
    """
    col_w = target_w // 3
    col_h = int(col_w * left_vis.shape[0] / left_vis.shape[1])

    def fit(img):
        return cv2.resize(img, (col_w, col_h), interpolation=cv2.INTER_AREA)

    row1 = np.hstack([fit(left_vis), fit(depth_vis), fit(right_vis)])

    cloud_h = col_h
    cloud_row = cv2.resize(cloud_vis, (target_w, cloud_h),
                           interpolation=cv2.INTER_AREA)

    return np.vstack([row1, cloud_row])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    use_sim   = "--sim" in sys.argv
    args      = [a for a in sys.argv[1:] if a != "--sim"]
    cam_index = int(args[0])   if len(args) > 0 else 0
    fov_deg   = float(args[1]) if len(args) > 1 else DEFAULT_FOV

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not load_depth_model(device):
        print("[ERREUR] Depth Anything V2 requis.")
        sys.exit(1)

    # ── Source de frames ──────────────────────────────────────────────────────
    sim = None
    cap = None
    if use_sim:
        from sim.panda_sim import PandaSim
        sim    = PandaSim()
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

    h, w = frame0.shape[:2]
    k1   = DEFAULT_K1

    def make_K(fov):
        f = (w / 2.0) / math.tan(math.radians(fov / 2.0))
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)

    def apply_undistort(img, K_, k1_):
        if abs(k1_) < 1e-4:
            return img
        dist = np.array([k1_, 0.0, 0.0, 0.0, 0.0])
        return cv2.undistort(img, K_, dist)

    K        = make_K(fov_deg)
    tracker  = KLTTracker(max_corners=300)
    renderer = PointCloudRenderer(canvas_w=w * 3, canvas_h=h)

    depth_map   = np.zeros((h, w), np.float32)
    depth_every = 3
    frame_idx   = 0

    print(f"  FOV {fov_deg}  k1={k1}  device={device}")
    print("  q=quitter  +/-=corners  r=reset  wasd=orbite  1/2/3=vues")
    print("  [ ]=FOV+-5  o/p=distorsion k1+-0.05")

    WIN = "Drominator Vision"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        # ── Acquisition ───────────────────────────────────────────────────────
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

        # ── Construire chaque panneau ─────────────────────────────────────────
        left_vis  = draw_left(left, pts, pts3d, len(pts))
        # HUD FOV / k1 en bas
        cv2.putText(left_vis,
                    f"FOV:{fov_deg:.0f}°  k1:{k1:.2f}  [ ]=fov  o/p=dist",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(left_vis,
                    f"FOV:{fov_deg:.0f}°  k1:{k1:.2f}  [ ]=fov  o/p=dist",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

        depth_vis = draw_depth(depth_map, frame.shape)
        right_vis = draw_right(right, depth_map, pts)
        cloud_vis = renderer.render()

        grid = make_grid(left_vis, depth_vis, right_vis, cloud_vis, target_w=w * 3)
        cv2.imshow(WIN, grid)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        elif key == ord('w'):  renderer.el = min(renderer.el + 5,  85)
        elif key == ord('s'):  renderer.el = max(renderer.el - 5, -85)
        elif key == ord('a'):  renderer.az -= 5
        elif key == ord('d'):  renderer.az += 5
        elif key == ord('1'):  renderer.az, renderer.el = 0,   0
        elif key == ord('2'):  renderer.az, renderer.el = 0,  -85
        elif key == ord('3'):  renderer.az, renderer.el = 90, -15
        elif key == ord('r'):  renderer.reset()
        elif key == ord('+'):  tracker.max_corners += 50
        elif key == ord('-'):  tracker.max_corners = max(50, tracker.max_corners - 50)
        elif key == ord(']'):
            fov_deg = min(fov_deg + 5, 170);  K = make_K(fov_deg)
            print(f"  FOV → {fov_deg:.0f}°")
        elif key == ord('['):
            fov_deg = max(fov_deg - 5, 20);   K = make_K(fov_deg)
            print(f"  FOV → {fov_deg:.0f}°")
        elif key == ord('p'):
            k1 = round(k1 - 0.05, 3);  print(f"  k1 → {k1:.2f}")
        elif key == ord('o'):
            k1 = round(k1 + 0.05, 3);  print(f"  k1 → {k1:.2f}")

        if use_sim:
            SPEED = 0.05
            if   key == 82:         sim.move( SPEED)
            elif key == 84:         sim.move(-SPEED)
            elif key == 81:         sim.strafe(-SPEED)
            elif key == 83:         sim.strafe( SPEED)
            elif key == ord('e'):   sim.turn(0,  2)
            elif key == ord('c'):   sim.turn(0, -2)
            elif key == ord('j'):   sim.turn(-3, 0)
            elif key == ord('l'):   sim.turn( 3, 0)

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

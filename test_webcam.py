import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import argparse
import sys
import cv2
import numpy as np
import threading
import torch
import time
from ultralytics import YOLO

from ai.pipeline_utils import (
    CONF_THRESHOLD, INFER_SIZE, NMS_IOU, FOV_WIDE, FOV_NARROW,
    MIN_BBOX_PX,
    DEPTH_MODEL_CHOICES,
    compute_focal, estimate_distance, danger_color,
    load_depth_model, infer_depth, depth_colormap, detect_unknown_obstacles,
    reset_depth_stream, depth_backend,
)

DEPTH_MAX_HZ = 12      # plafond depth inference (thread depth)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Drominator — détection v2 (YOLO + depth)")
    ap.add_argument("--sim", action="store_true", help="Rendu Panda3D au lieu de la webcam")
    ap.add_argument("--autopilot", action="store_true", help="Mode PPO + visualisation couloir")
    ap.add_argument(
        "--depth-model", default="auto", choices=DEPTH_MODEL_CHOICES,
        help="Backend depth — auto = VDA-Small → VDA-Base → DA-V2 (défaut: auto)",
    )
    ap.add_argument(
        "--depth-infer-every", type=int, default=3,
        help="N'exécute le forward depth qu'une frame sur K (défaut 3)",
    )
    ap.add_argument(
        "--benchmark", action="store_true",
        help="Mesure les FPS du backend depth sur 100 frames synthétiques puis quitte",
    )
    ap.add_argument(
        "--device", default="",
        help="Force 'cpu' / 'cuda' / 'cuda:0'. Défaut : auto-détect",
    )
    return ap.parse_args()


ARGS = _parse_args()
SIM_MODE = ARGS.sim
AUTOPILOT_MODE = ARGS.autopilot


def to_3d(cx, cy, dist, fw, fh, focal):
    return (round((cx - fw/2) * dist / focal, 2),
            round((cy - fh/2) * dist / focal, 2),
            round(dist, 2))

# ── Calibration distance via depth map ───────────────────────────────────────
def calibrate_depth_scale(calib_pts, depth_map):
    """
    Calibre l'échelle depth → mètres à partir des objets YOLO connus.
    calib_pts : liste de (cx, cy, dist_metric)
    Retourne alpha tel que dist_metric ≈ alpha * depth_norm
    """
    alphas = []
    for (cx, cy, dist) in calib_pts:
        r, c = int(cy), int(cx)
        if 0 <= r < depth_map.shape[0] and 0 <= c < depth_map.shape[1]:
            d = depth_map[r, c]
            if d > 0.05:
                alphas.append(dist / d)
    return float(np.median(alphas)) if alphas else None

def depth_dist(cx, cy, depth_map, scale):
    """Distance métrique d'un point via la depth map calibrée."""
    if scale is None:
        return None
    r, c = int(cy), int(cx)
    if 0 <= r < depth_map.shape[0] and 0 <= c < depth_map.shape[1]:
        d = depth_map[r, c]
        if d > 0.02:
            return round(scale * d, 1)
    return None

# ── Top-down map ──────────────────────────────────────────────────────────────
MAP_SIZE    = 220
MAP_RANGE_M = 20.0

def draw_topdown_map(objects_3d):
    m     = np.full((MAP_SIZE, MAP_SIZE, 3), 30, dtype=np.uint8)
    scale = MAP_SIZE / MAP_RANGE_M
    for d in range(5, int(MAP_RANGE_M), 5):
        r = int(d * scale)
        cv2.circle(m, (MAP_SIZE//2, MAP_SIZE-10), r, (60,60,60), 1)
        cv2.putText(m, f"{d}m", (MAP_SIZE//2+2, MAP_SIZE-10-r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
    for (x, z, color) in objects_3d:
        px = int(MAP_SIZE/2 + x*scale)
        py = int(MAP_SIZE-10 - z*scale)
        if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
            cv2.circle(m, (px, py), 7, color, -1)
    dx, dy = MAP_SIZE//2, MAP_SIZE-10
    pts = np.array([[dx, dy-12],[dx-7,dy+4],[dx+7,dy+4]], np.int32)
    cv2.fillPoly(m, [pts], (255,255,255))
    cv2.putText(m, "TOP VIEW", (4,14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    return m

# ── Depth : chargement + inférence importés depuis ai/pipeline_utils ─────────

# ── Mode Autopilot ────────────────────────────────────────────────────────────

def _run_autopilot():
    """Boucle autonome : modèle PPO + visualisation Panda3D couloir infini."""
    import os
    from stable_baselines3 import PPO
    from ai.drone_env    import DroneCorridorEnv
    from sim.corridor_sim import CorridorSim
    from ai.corridor      import raycast, N_RAYS

    model_path = os.path.join(os.path.dirname(__file__), "ai", "models", "drone_ppo_final.zip")
    if not os.path.exists(model_path):
        # Cherche le checkpoint le plus récent
        ckpt_dir = os.path.join(os.path.dirname(__file__), "ai", "models")
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("drone_ppo_") and f.endswith(".zip")]
        ) if os.path.isdir(ckpt_dir) else []
        if ckpts:
            model_path = os.path.join(ckpt_dir, ckpts[-1])
            print(f"[AUTO] Utilise checkpoint : {ckpts[-1]}")
        else:
            print("[AUTO] Aucun modèle trouvé dans ai/models/")
            print("[AUTO] Lance d'abord : python -m ai.train")
            print("[AUTO] Démo avec actions aléatoires (Ctrl+C pour quitter)")
            model_path = None

    env = DroneCorridorEnv(seed=42)
    obs, _ = env.reset()

    sim = CorridorSim(env)
    print("[AUTO] Couloir infini lancé — Échap pour quitter")

    model = PPO.load(model_path) if model_path else None

    cv2.namedWindow("Drominator — Autopilot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drominator — Autopilot", 960, 480)

    total_reward = 0.0
    episode      = 1
    steps        = 0
    t0           = time.perf_counter()
    fps_disp     = 0.0
    t_prev       = t0

    while True:
        # ── Inférence ──────────────────────────────────────────────────────
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        steps        += 1

        # ── Rendu ──────────────────────────────────────────────────────────
        frame = sim.sync_and_render()

        # ── Overlay ────────────────────────────────────────────────────────
        t_now    = time.perf_counter()
        fps_disp = 0.85 * fps_disp + 0.15 / max(t_now - t_prev, 1e-6)
        t_prev   = t_now
        pos      = env.pos
        rays     = obs[:N_RAYS]
        min_dist = float(rays.min()) * 9.0   # RAY_DIST = 9m

        info_lines = [
            (f"FPS {fps_disp:.0f}",                (0, 255, 0)),
            (f"Episode {episode}  step {steps}",   (200, 200, 200)),
            (f"Y = {pos[1]:.1f} m",                (0, 220, 255)),
            (f"Reward cumulé = {total_reward:.1f}",(255, 200, 0)),
            (f"Obstacle min  = {min_dist:.2f} m",  (255, 80, 80) if min_dist < 1.5 else (160, 255, 160)),
        ]
        for i, (txt, col) in enumerate(info_lines):
            cv2.putText(frame, txt, (10, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)

        # Visualisation des raycasts (barre en bas)
        bar_h = 18
        bar_w = frame.shape[1]
        bar   = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        n     = len(rays)
        for ri, v in enumerate(rays):
            x0 = int(ri / n * bar_w)
            x1 = int((ri + 1) / n * bar_w)
            col_val = int(v * 255)
            danger  = max(0, 255 - col_val * 3)
            cv2.rectangle(bar, (x0, 0), (x1, bar_h),
                          (0, col_val, danger), -1)
        display = np.vstack([frame, bar])

        cv2.imshow("Drominator — Autopilot", display)

        # ── Reset si collision ─────────────────────────────────────────────
        if terminated:
            print(f"[AUTO] Ep {episode:4d} | {pos[1]:.1f} m | reward {total_reward:.1f}")
            episode     += 1
            steps        = 0
            total_reward = 0.0
            obs, _       = env.reset()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if AUTOPILOT_MODE:
    _run_autopilot()
    sys.exit(0)


# ── État partagé entre threads ────────────────────────────────────────────────
class PipelineState:
    def __init__(self):
        self._lock     = threading.Lock()
        self.frame     = None
        self.yolo_pack = None   # (frame_copy, results)
        self.depth_map = None   # ndarray H×W float32
        self.running   = True

    def set_frame(self, f):
        with self._lock: self.frame = f

    def get_frame(self):
        with self._lock: return self.frame

    def set_yolo(self, frame, results):
        with self._lock: self.yolo_pack = (frame, results)

    def get_yolo(self):
        with self._lock: return self.yolo_pack

    def set_depth(self, d):
        with self._lock: self.depth_map = d

    def get_depth(self):
        with self._lock: return self.depth_map

# ── Threads ───────────────────────────────────────────────────────────────────
def _capture_loop(cap, st):
    while st.running:
        ret, f = cap.read()
        if ret and f is not None:
            st.set_frame(f.copy())
        else:
            time.sleep(0.001)

def _sim_capture_loop(renderer, st):
    while st.running:
        st.set_frame(renderer.get_frame())
        time.sleep(1 / 60)

def _yolo_loop(model, st):
    while st.running:
        frame = st.get_frame()
        if frame is None:
            time.sleep(0.001)
            continue
        res = model(frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD,
                    iou=NMS_IOU, verbose=False)[0]
        st.set_yolo(frame.copy(), res)

def _depth_loop(st, depth_ok):
    interval = 1.0 / DEPTH_MAX_HZ
    while st.running:
        t0 = time.perf_counter()
        if not depth_ok:
            time.sleep(0.1)
            continue
        frame = st.get_frame()
        if frame is not None:
            st.set_depth(infer_depth(frame))
        elapsed = time.perf_counter() - t0
        rem = interval - elapsed
        if rem > 0:
            time.sleep(rem)

# ── Init ──────────────────────────────────────────────────────────────────────
device = ARGS.device or ("cuda" if torch.cuda.is_available() else "cpu")

if ARGS.benchmark:
    depth_ok = load_depth_model(
        device=device, model=ARGS.depth_model, infer_every=ARGS.depth_infer_every,
    )
    if not depth_ok:
        sys.exit(2)
    print(f"[DEPTH] Backend actif : {depth_backend()}")
    import time as _bt
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(100)]
    _ = infer_depth(frames[0])
    reset_depth_stream()
    t0 = _bt.perf_counter()
    for f in frames:
        _ = infer_depth(f)
    dt = _bt.perf_counter() - t0
    fps = 100.0 / dt
    print(
        f"[BENCH] backend={depth_backend()} device={device} frames=100 "
        f"infer_every={ARGS.depth_infer_every} → {fps:.1f} FPS "
        f"({dt * 10:.1f} ms/frame)"
    )
    if fps < 10.0:
        print("[BENCH] ⚠ en-dessous du plancher 10 FPS")
    elif fps < 15.0:
        print("[BENCH] ⚠ en-dessous de la cible 15 FPS (plancher 10 OK)")
    else:
        print("[BENCH] ✓ au-dessus de la cible 15 FPS")
    sys.exit(0)

yolo = YOLO("models/yolo11n.pt")
yolo.to(device)
print(f"[IA]   YOLOv11n sur {device}")

depth_ok = load_depth_model(
    device=device,
    model=ARGS.depth_model,
    infer_every=ARGS.depth_infer_every,
)
if depth_ok:
    print(f"[DEPTH] Backend actif : {depth_backend()}")

renderer = None
cap      = None

if SIM_MODE:
    from sim.panda_sim import PandaSim
    renderer = PandaSim()
    frame_w, frame_h = 640, 480
    print("[CAM]  Mode simulation Panda3D (ZQSD + flèches)")
else:
    for idx in range(10):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c
            print(f"[CAM]  Caméra index {idx}")
            break
        c.release()
    if cap is None:
        raise RuntimeError("Aucune caméra trouvée")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

focal_wide   = compute_focal(frame_w, FOV_WIDE)
focal_narrow = compute_focal(frame_w, FOV_NARROW)
print(f"[CAM]  {frame_w}x{frame_h} | focal {focal_wide:.0f}/{focal_narrow:.0f}px")
print("[OK]   Détection v2 — 3 threads — Echap pour quitter\n")

# Démarrage des threads
st = PipelineState()

# En mode sim, get_frame() doit rester sur le main thread (contrainte Panda3D).
# On démarre seulement YOLO + depth en background.
if SIM_MODE:
    threads = [
        threading.Thread(target=_yolo_loop,  args=(yolo, st), daemon=True),
        threading.Thread(target=_depth_loop, args=(st, depth_ok), daemon=True),
    ]
else:
    threads = [
        threading.Thread(target=_capture_loop, args=(cap, st), daemon=True),
        threading.Thread(target=_yolo_loop,    args=(yolo, st), daemon=True),
        threading.Thread(target=_depth_loop,   args=(st, depth_ok), daemon=True),
    ]
for t in threads:
    t.start()

panel_sq = frame_h
cv2.namedWindow("Drominator — Detection v2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drominator — Detection v2",
                 min(frame_w + panel_sq*2, 1600), frame_h)

fps_display = 0.0
t_prev      = time.perf_counter()

# ── Boucle d'affichage ────────────────────────────────────────────────────────
while True:
    # Panda3D doit tourner sur le main thread
    if SIM_MODE and renderer is not None:
        st.set_frame(renderer.get_frame())

    yolo_pack = st.get_yolo()
    depth_map = st.get_depth()

    if yolo_pack is None:
        if cv2.waitKey(1) == 27:
            break
        continue

    frame, results = yolo_pack
    frame      = frame.copy()
    fh, fw     = frame.shape[:2]
    objects_3d = []
    yolo_boxes = []
    calib_pts  = []   # (cx, cy, dist_metric) pour calibrer la depth

    # ── YOLO ─────────────────────────────────────────────────────────────────
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls   = yolo.names[int(box.cls[0])]
        conf  = float(box.conf[0])
        cx    = (x1+x2)/2
        cy    = (y1+y2)/2
        dist, cfd = estimate_distance(cls, y2-y1, fh, focal_wide, focal_narrow)
        color = danger_color(dist)
        yolo_boxes.append((x1, y1, x2, y2))

        if dist is not None:
            calib_pts.append((cx, cy, dist))
            ox, oy, oz = to_3d(cx, cy, dist, fw, fh, focal_wide)
            objects_3d.append((ox, oz, color))
            coord_str = f"  [{ox:+.1f},{oy:+.1f},{oz:.1f}]m"
        else:
            coord_str = ""

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.circle(frame, (int(cx),int(cy)), 3, color, -1)
        dist_str = f" ~{dist}m ({cfd:.0%})" if dist else ""
        lbl = f"{cls} {conf:.0%}{dist_str}"
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, lbl, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        if coord_str:
            cv2.putText(frame, coord_str, (x1, y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── Depth ────────────────────────────────────────────────────────────────
    if depth_map is not None:
        scale   = calibrate_depth_scale(calib_pts, depth_map)
        unknown = detect_unknown_obstacles(depth_map, yolo_boxes, frame.shape)

        for (x1, y1, x2, y2) in unknown:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            dist   = depth_dist(cx, cy, depth_map, scale)
            if dist is None:
                dist, _ = estimate_distance("unknown", y2-y1, fh, focal_wide, focal_narrow)
            color = danger_color(dist)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            lbl = f"? ~{dist}m" if dist else "?"
            cv2.putText(frame, lbl, (x1+2, max(y1-4,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if dist is not None:
                ox,oy,oz = to_3d(cx, cy, dist, fw, fh, focal_wide)
                objects_3d.append((ox, oz, (160,100,255)))

        depth_panel = cv2.resize(depth_colormap(depth_map), (panel_sq, fh))
        label_depth = f"DEPTH  scale={'%.1f'%scale if scale else 'N/A'}"
        cv2.putText(depth_panel, label_depth, (6,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    else:
        depth_panel = np.zeros((fh, panel_sq, 3), dtype=np.uint8)
        cv2.putText(depth_panel, "DEPTH N/A", (10, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 1)

    # ── FPS ──────────────────────────────────────────────────────────────────
    t_now       = time.perf_counter()
    fps_display = 0.9*fps_display + 0.1/(t_now - t_prev)
    t_prev      = t_now
    cv2.putText(frame, f"FPS {fps_display:.0f}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    topdown  = cv2.resize(draw_topdown_map(objects_3d), (panel_sq, fh))
    combined = np.hstack([frame, depth_panel, topdown])
    cv2.imshow("Drominator — Detection v2", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if SIM_MODE and renderer is not None:
        SPD = 0.12
        if key == ord('z'): renderer.move( SPD)
        if key == ord('s'): renderer.move(-SPD)
        if key == ord('q'): renderer.strafe(-SPD)
        if key == ord('d'): renderer.strafe( SPD)
        if key == 81:       renderer.turn(-0.05, 0)   # flèche gauche
        if key == 83:       renderer.turn( 0.05, 0)   # flèche droite
        if key == 82:       renderer.turn(0,  0.04)   # flèche haut
        if key == 84:       renderer.turn(0, -0.04)   # flèche bas

st.running = False
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
print("[FIN] Détection v2 terminée.")

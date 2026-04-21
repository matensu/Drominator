import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import math
import numpy as np
import torch
import time
from ultralytics import YOLO

CONF_THRESHOLD = 0.5
INFER_SIZE     = 320

# FOV horizontal de la caméra en degrés
# Webcam standard : ~70°  |  FPV grand angle : 120-150°
# Plus cette valeur est précise, meilleure est l'estimation de distance
# FOVs utilisés pour la fusion pondérée
# FOV_WIDE  = grand angle  → précis de près  (caméra FPV typique)
# FOV_NARROW = angle étroit → précis de loin  (zoom/standard)
FOV_WIDE   = 120
FOV_NARROW = 60

# Seuil de pixels minimum pour qu'une estimation soit considérée fiable
MIN_BBOX_PX = 20

# Hauteurs réelles des objets COCO (mètres)
REAL_HEIGHT = {
    "person": 1.75, "bicycle": 1.1,  "car": 1.5,  "motorcycle": 1.2,
    "bus":    3.0,  "truck":   3.5,  "dog": 0.5,  "cat":        0.3,
    "chair":  0.9,  "bottle":  0.25, "laptop": 0.03, "tv": 0.6,
}
DEFAULT_HEIGHT = 1.0

# ── Caméra intrinsèque ────────────────────────────────────────────────────────
def compute_focal(frame_w: int, fov_deg: float) -> float:
    return frame_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

# ── 3D ───────────────────────────────────────────────────────────────────────
def to_3d(cx_px, cy_px, dist_m, frame_w, frame_h, focal_px):
    x = (cx_px - frame_w / 2.0) * dist_m / focal_px
    y = (cy_px - frame_h / 2.0) * dist_m / focal_px
    return round(x, 2), round(y, 2), round(dist_m, 2)

# ── Distance fusionnée ────────────────────────────────────────────────────────
def estimate_distance(label: str, bbox_h_px: int, frame_h: int,
                      focal_wide: float, focal_narrow: float) -> tuple[float, float] | tuple[None, None]:
    """
    Retourne (distance_m, confidence_0_to_1).

    Fusion pondérée entre deux estimations FOV :
      - FOV large  (focal_wide)   : fiable de PRÈS  (grande bbox)
      - FOV étroit (focal_narrow) : fiable de LOIN  (petite bbox)

    Le poids est basé sur la taille relative de la bbox dans le frame.
    Confiance globale = fonction du nombre de pixels disponibles.
    """
    if bbox_h_px < MIN_BBOX_PX:
        return None, None

    real_h = REAL_HEIGHT.get(label, DEFAULT_HEIGHT)

    dist_wide   = (real_h * focal_wide)   / bbox_h_px
    dist_narrow = (real_h * focal_narrow) / bbox_h_px

    # bbox_ratio : 0 = objet minuscule (loin), 1 = objet occupe tout le frame (très près)
    bbox_ratio = min(bbox_h_px / frame_h, 1.0)

    # Poids exclusifs : quand l'un est confiant, l'autre s'efface
    # SHARPNESS contrôle l'exclusivité : 1 = linéaire, 3 = quasi-exclusif, 5 = winner-takes-all
    SHARPNESS  = 5
    raw_wide   = bbox_ratio ** SHARPNESS          # confiant de près (grande bbox)
    raw_narrow = (1.0 - bbox_ratio) ** SHARPNESS  # confiant de loin (petite bbox)
    total      = raw_wide + raw_narrow

    w_wide   = raw_wide   / total
    w_narrow = raw_narrow / total

    dist_fused = w_wide * dist_wide + w_narrow * dist_narrow

    # Confiance globale : croît avec le carré de bbox_h (plus de pixels = moins d'erreur relative)
    confidence = min((bbox_h_px / MIN_BBOX_PX) ** 2 / 100.0, 1.0)
    confidence = round(confidence, 2)

    return round(dist_fused, 1), confidence

def danger_color(dist):
    if dist is None: return (180, 180, 180)
    if dist < 3:     return (0,   0,   255)   # rouge  < 3m
    if dist < 8:     return (0, 165,   255)   # orange < 8m
    return              (0, 210,    0)         # vert   > 8m

# ── Top-down map ──────────────────────────────────────────────────────────────
MAP_SIZE    = 220   # pixels
MAP_RANGE_M = 20.0  # mètres représentés sur la carte

def draw_topdown_map(objects_3d: list) -> np.ndarray:
    """
    objects_3d : liste de (x, z, color)
    Retourne une image MAP_SIZE x MAP_SIZE avec vue du dessus.
    Drone = triangle en bas au centre, obstacles = cercles colorés.
    """
    m = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    m[:] = (30, 30, 30)

    # Grille de référence (tous les 5m)
    scale = MAP_SIZE / MAP_RANGE_M
    for d in range(5, int(MAP_RANGE_M), 5):
        r = int(d * scale)
        cv2.circle(m, (MAP_SIZE // 2, MAP_SIZE - 10), r, (60, 60, 60), 1)
        cv2.putText(m, f"{d}m", (MAP_SIZE // 2 + 2, MAP_SIZE - 10 - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    # Obstacles
    for (x, z, color) in objects_3d:
        px = int(MAP_SIZE / 2 + x * scale)
        py = int(MAP_SIZE - 10 - z * scale)
        if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
            cv2.circle(m, (px, py), 7, color, -1)

    # Drone (triangle)
    drone_x = MAP_SIZE // 2
    drone_y = MAP_SIZE - 10
    pts = np.array([[drone_x, drone_y - 12],
                    [drone_x - 7, drone_y + 4],
                    [drone_x + 7, drone_y + 4]], np.int32)
    cv2.fillPoly(m, [pts], (255, 255, 255))

    cv2.putText(m, "TOP VIEW", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return m

# ── Init ──────────────────────────────────────────────────────────────────────
model = YOLO("models/yolo11n.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"[IA]  YOLOv11n sur {next(model.model.parameters()).device}")

cap = None
for idx in range(10):
    c = cv2.VideoCapture(idx)
    if c.isOpened():
        cap = c
        print(f"[CAM] Camera index {idx}")
        break
    c.release()
if cap is None:
    raise RuntimeError("Aucune camera trouvee")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
focal_wide   = compute_focal(frame_w, FOV_WIDE)
focal_narrow = compute_focal(frame_w, FOV_NARROW)
print(f"[CAM] Resolution {frame_w}x{frame_h} | FOV {FOV_WIDE}°/{FOV_NARROW}° | focal {focal_wide:.0f}/{focal_narrow:.0f}px")
print("[OK]  Test actif — Echap pour quitter\n")

cv2.namedWindow("Drominator — Test Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drominator — Test Webcam", 1180, 540)

fps_display = 0.0
t_prev = time.perf_counter()

# ── Boucle ────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    results  = model(frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
    objects_3d = []

    for box in results.boxes:
        x1, y1, x2, y2  = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls[0])]
        conf     = float(box.conf[0])

        bbox_cx = (x1 + x2) / 2
        bbox_cy = (y1 + y2) / 2
        dist, conf_dist = estimate_distance(cls_name, y2 - y1, frame_h, focal_wide, focal_narrow)
        color           = danger_color(dist)

        # Coordonnées 3D
        if dist is not None:
            ox, oy, oz = to_3d(bbox_cx, bbox_cy, dist, frame_w, frame_h, focal_wide)
            objects_3d.append((ox, oz, color))
            coord_str = f"  [{ox:+.1f}, {oy:+.1f}, {oz:.1f}]m"
        else:
            coord_str = ""

        # Dessin bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Point central
        cv2.circle(frame, (int(bbox_cx), int(bbox_cy)), 3, color, -1)

        dist_str = f" ~{dist}m ({conf_dist:.0%})" if dist is not None else ""
        label    = f"{cls_name} {conf:.0%}{dist_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Coordonnées 3D sous la bbox
        if coord_str:
            cv2.putText(frame, coord_str, (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # FPS
    t_now       = time.perf_counter()
    fps_display = 0.9 * fps_display + 0.1 * (1.0 / (t_now - t_prev))
    t_prev      = t_now
    cv2.putText(frame, f"FPS {fps_display:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Carte top-down collée à droite
    topdown = draw_topdown_map(objects_3d)
    topdown_resized = cv2.resize(topdown, (frame_h, frame_h))  # carré = hauteur frame
    combined = np.hstack([frame, topdown_resized])

    cv2.imshow("Drominator — Test Webcam", combined)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[FIN] Test termine.")

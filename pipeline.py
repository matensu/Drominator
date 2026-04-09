import cv2
import torch
import time
from ultralytics import YOLO

CAMERA_INDEX   = 4
CONF_THRESHOLD = 0.5
INFER_SIZE     = 320

# Hauteurs réelles de référence par classe COCO (en mètres)
# Utilisées pour estimer la distance via la taille dans l'image
REAL_HEIGHT = {
    "person": 1.75, "bicycle": 1.1, "car": 1.5, "motorcycle": 1.2,
    "bus": 3.0, "truck": 3.5, "dog": 0.5, "cat": 0.3,
    "tree": 4.0, "chair": 0.9, "bottle": 0.25,
}
DEFAULT_HEIGHT = 1.0   # fallback pour les classes inconnues
FOCAL_PX       = 600   # longueur focale estimée en pixels (caméra FPV typique)

def estimate_distance(label: str, bbox_h_px: int) -> float | None:
    if bbox_h_px <= 0:
        return None
    real_h = REAL_HEIGHT.get(label, DEFAULT_HEIGHT)
    return round((real_h * FOCAL_PX) / bbox_h_px, 1)

def danger_color(dist: float | None) -> tuple:
    if dist is None:
        return (200, 200, 200)
    if dist < 3:
        return (0, 0, 255)    # rouge  — DANGER
    if dist < 8:
        return (0, 165, 255)  # orange — ATTENTION
    return (0, 220, 0)        # vert   — OK

# ── Modèle ────────────────────────────────────────────────────────────────────
model = YOLO("models/yolo11n.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.model.parameters()).device
print(f"[IA]  YOLOv11n sur {device}")

# ── Caméra ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir /dev/video{CAMERA_INDEX}")

cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter.fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          60)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
print(f"[CAM] /dev/video{CAMERA_INDEX} ouvert (720x480 MJPG 60fps)")
print("[OK]  Pipeline actif — Echap pour quitter\n")

# ── Boucle principale ─────────────────────────────────────────────────────────
fps_display = 0.0
t_prev      = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    results = model(frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls[0])]
        conf     = float(box.conf[0])
        dist     = estimate_distance(cls_name, y2 - y1)
        color    = danger_color(dist)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label : "person 92% ~4.2m"
        dist_str = f" ~{dist}m" if dist is not None else ""
        label    = f"{cls_name} {conf:.0%}{dist_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # FPS overlay
    t_now       = time.perf_counter()
    fps_display = 0.9 * fps_display + 0.1 * (1.0 / (t_now - t_prev))
    t_prev      = t_now
    cv2.putText(frame, f"FPS {fps_display:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Drominator — FPV Vision", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[FIN] Pipeline arrete proprement.")

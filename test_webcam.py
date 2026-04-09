import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # force X11 sur Wayland

import cv2
import torch
import time
from ultralytics import YOLO

CONF_THRESHOLD = 0.5
INFER_SIZE     = 320

REAL_HEIGHT = {
    "person": 1.75, "bicycle": 1.1, "car": 1.5, "motorcycle": 1.2,
    "bus": 3.0, "truck": 3.5, "dog": 0.5, "cat": 0.3,
    "chair": 0.9, "bottle": 0.25,
}
DEFAULT_HEIGHT = 1.0
FOCAL_PX       = 600

def estimate_distance(label, bbox_h_px):
    if bbox_h_px <= 0:
        return None
    return round((REAL_HEIGHT.get(label, DEFAULT_HEIGHT) * FOCAL_PX) / bbox_h_px, 1)

def danger_color(dist):
    if dist is None:   return (200, 200, 200)
    if dist < 3:       return (0, 0, 255)
    if dist < 8:       return (0, 165, 255)
    return (0, 220, 0)

# ── Trouver une caméra disponible ─────────────────────────────────────────────
model = YOLO("models/yolo11n.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"[IA] YOLOv11n sur {next(model.model.parameters()).device}")

cap = None
for idx in range(10):
    c = cv2.VideoCapture(idx)
    if c.isOpened():
        cap = c
        print(f"[CAM] Camera trouvee sur index {idx}")
        break
    c.release()

if cap is None:
    raise RuntimeError("Aucune camera trouvee (index 0-9)")

# ── Boucle principale ─────────────────────────────────────────────────────────
fps_display = 0.0
t_prev = time.perf_counter()
print("[OK] Test actif — Echap pour quitter\n")

cv2.namedWindow("Drominator — Test Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drominator — Test Webcam", 960, 540)

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        dist_str = f" ~{dist}m" if dist is not None else ""
        label    = f"{cls_name} {conf:.0%}{dist_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    t_now       = time.perf_counter()
    fps_display = 0.9 * fps_display + 0.1 * (1.0 / (t_now - t_prev))
    t_prev      = t_now
    cv2.putText(frame, f"FPS {fps_display:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Drominator — Test Webcam", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[FIN] Test termine.")

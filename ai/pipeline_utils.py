"""
Fonctions de perception extraites de test_webcam.py.
Importable sans déclencher le lancement du programme principal.
"""
import math
import cv2
import numpy as np
import torch
from PIL import Image

# ── Constantes ────────────────────────────────────────────────────────────────
CONF_THRESHOLD    = 0.5
INFER_SIZE        = 320
NMS_IOU           = 0.8
FOV_WIDE          = 120
FOV_NARROW        = 60
MIN_BBOX_PX       = 20
MIN_UNKNOWN_AREA  = 1500
MAX_UNKNOWN_DEPTH = 0.55
DEPTH_CANNY_LOW   = 20
DEPTH_CANNY_HIGH  = 60

REAL_HEIGHT = {
    "person": 1.75, "bicycle": 1.1,  "car":  1.5,  "motorcycle": 1.2,
    "bus":    3.0,  "truck":   3.5,  "dog":  0.5,  "cat":        0.3,
    "chair":  0.9,  "bottle":  0.25, "laptop": 0.35, "tv":       0.6,
    "window": 1.5,  "door":    2.0,  "wall": 2.5,
}
DEFAULT_HEIGHT = 1.0

# ── Géométrie ──────────────────────────────────────────────────────────────────
def compute_focal(w, fov):
    return w / (2.0 * math.tan(math.radians(fov / 2.0)))

def estimate_distance(label, bbox_h_px, frame_h, focal_wide, focal_narrow):
    if bbox_h_px < MIN_BBOX_PX:
        return None, None
    real_h   = REAL_HEIGHT.get(label, DEFAULT_HEIGHT)
    d_wide   = (real_h * focal_wide)   / bbox_h_px
    d_narrow = (real_h * focal_narrow) / bbox_h_px
    ratio    = min(bbox_h_px / frame_h, 1.0)
    S        = 5
    rw, rn   = ratio**S, (1 - ratio)**S
    t        = rw + rn
    dist     = (rw / t) * d_wide + (rn / t) * d_narrow
    conf     = min((bbox_h_px / MIN_BBOX_PX)**2 / 100.0, 1.0)
    return round(dist, 1), round(conf, 2)

def danger_color(dist):
    if dist is None: return (180, 180, 180)
    if dist < 3:     return (0,   0,   255)
    if dist < 8:     return (0, 165,   255)
    return                  (0, 210,     0)

# ── Depth Anything V2 ─────────────────────────────────────────────────────────
_depth_model     = None
_depth_processor = None
_depth_device    = "cpu"

def load_depth_model(device="cpu"):
    global _depth_model, _depth_processor, _depth_device
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print("[DEPTH] Chargement Depth Anything V2-Small...")
        _depth_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf")
        _depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf")
        _depth_model.to(device).eval()
        _depth_device = device
        print(f"[DEPTH] Chargé sur {device}")
        return True
    except Exception as e:
        print(f"[DEPTH] Non disponible : {e}")
        return False

def infer_depth(frame_bgr):
    pil    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = _depth_processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(_depth_device) for k, v in inputs.items()}
    with torch.no_grad():
        raw = _depth_model(**inputs).predicted_depth.squeeze().cpu().float().numpy()
    d_min, d_max = raw.min(), raw.max()
    norm = (raw - d_min) / (d_max - d_min + 1e-8)
    return cv2.resize(norm, (frame_bgr.shape[1], frame_bgr.shape[0]),
                      interpolation=cv2.INTER_LINEAR)

def depth_colormap(depth_norm):
    return cv2.applyColorMap(
        ((1.0 - depth_norm) * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO)

def detect_unknown_obstacles(depth_norm, yolo_boxes, shape):
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    edges    = cv2.Canny(depth_u8, DEPTH_CANNY_LOW, DEPTH_CANNY_HIGH)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges    = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    unknown = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_UNKNOWN_AREA:
            continue
        x1, y1, wc, hc = cv2.boundingRect(cnt)
        x2, y2 = x1 + wc, y1 + hc
        if depth_norm[y1:y2, x1:x2].mean() > MAX_UNKNOWN_DEPTH:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if any(bx1-10 <= cx <= bx2+10 and by1-10 <= cy <= by2+10
               for (bx1, by1, bx2, by2) in yolo_boxes):
            continue
        unknown.append((x1, y1, x2, y2))
    return unknown

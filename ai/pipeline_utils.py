"""
Fonctions de perception extraites de test_webcam.py.

Importable sans déclencher de chargement lourd au module-level — tous les
imports de modèles (transformers, VDA) sont faits à l'intérieur des fonctions
``load_depth_model`` / ``_try_load_*``.

Depth backend — triple fallback :
  1. VDA via HF transformers (``AutoModelForDepthEstimation``)
     – Actuellement aucun model id VDA n'expose de config HF; ce tier échoue
       silencieusement et passe au suivant. Gardé comme crochet pour quand
       l'upstream ajoutera l'intégration.
  2. VDA via le repo officiel vendu sous ``vendor/Video-Depth-Anything``
     + poids cachés dans ``ai/models/metric_video_depth_anything_*.pth``.
     API streaming : cache de hidden states entre frames (INFER_LEN=32), reset
     via ``reset_depth_stream()``.
  3. Depth-Anything-V2-Small via HF transformers (comportement historique).

Conventions de sortie :
  * ``infer_depth(frame)`` → ``np.ndarray HxW float32`` normalisé ``[0, 1]``,
    disparité-like (plus haut = plus proche). Maintenu pour compat avec
    ``detect_unknown_obstacles`` et ``calibrate_depth_scale`` dans test_webcam.py.
  * ``infer_depth_meters(frame)`` → ``np.ndarray HxW float32`` en **mètres**,
    ou ``None`` si le backend courant ne produit pas de depth métrique
    (DA-V2). Disponible uniquement avec VDA-metric.

Inférence throttlée :
  Le paramètre ``infer_every`` passé à ``load_depth_model`` limite le coût en
  ne lançant un vrai forward qu'une frame sur K. Entre les appels, la dernière
  carte calculée est renvoyée telle quelle. ``reset_depth_stream()`` remet le
  compteur à zéro en même temps que le cache temporel.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

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

# ── Depth backends ────────────────────────────────────────────────────────────

DEPTH_MODEL_CHOICES = ("auto", "vda-small", "vda-base", "da-v2")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_VDA_VENDOR_DIR = _REPO_ROOT / "vendor" / "Video-Depth-Anything"
_DEFAULT_WEIGHTS_DIR = _THIS_DIR / "models"

_VDA_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
}
_VDA_CHECKPOINT_NAMES = {
    "vits": "metric_video_depth_anything_vits.pth",
    "vitb": "metric_video_depth_anything_vitb.pth",
}
_VDA_HF_REPOS = {
    "vits": "depth-anything/Metric-Video-Depth-Anything-Small",
    "vitb": "depth-anything/Metric-Video-Depth-Anything-Base",
}

# Module-level state — populated by load_depth_model().
_depth_model       = None
_depth_processor   = None            # DA-V2 HF processor
_depth_device      = "cpu"
_depth_backend: Optional[str] = None # "vda-small", "vda-base" ou "da-v2"
_depth_infer_every = 3
_depth_call_count  = 0
_last_depth_meters: Optional[np.ndarray] = None
_last_depth_norm:   Optional[np.ndarray] = None


def _vendor_vda_on_path() -> bool:
    if not _VDA_VENDOR_DIR.exists():
        return False
    p = str(_VDA_VENDOR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return True


def _ensure_vda_weights(variant: str, weights_dir: Path) -> Optional[Path]:
    ckpt_name = _VDA_CHECKPOINT_NAMES[variant]
    local_path = weights_dir / ckpt_name
    if local_path.exists():
        return local_path
    try:
        from huggingface_hub import hf_hub_download
        print(f"[DEPTH] Téléchargement {ckpt_name} depuis HF hub...")
        downloaded = hf_hub_download(
            repo_id=_VDA_HF_REPOS[variant],
            filename=ckpt_name,
            local_dir=str(weights_dir),
        )
        return Path(downloaded)
    except Exception as e:
        print(f"[DEPTH] Échec téléchargement VDA ({e})")
        print(
            f"[DEPTH] Téléchargement manuel : wget "
            f"https://huggingface.co/{_VDA_HF_REPOS[variant]}/resolve/main/{ckpt_name} "
            f"-O {local_path}"
        )
        return None


def _try_load_vda_transformers(variant: str, device: str) -> bool:
    """Tier 1 : VDA via HF transformers. Actuellement aucun model id VDA n'expose
    de ``config.json`` compatible AutoModelForDepthEstimation — donc renvoie
    False en pratique. Gardé comme crochet pour upstream."""
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(_VDA_HF_REPOS[variant])
    except Exception:
        return False
    # Si un jour AutoConfig réussit, ajouter ici le vrai chemin de chargement.
    return False


def _try_load_vda_repo(variant: str, device: str, weights_dir: Path) -> bool:
    """Tier 2 : VDA via repo vendorisé + classe streaming."""
    global _depth_model, _depth_device, _depth_backend
    if not _vendor_vda_on_path():
        print("[DEPTH] Repo VDA introuvable : vendor/Video-Depth-Anything")
        return False
    try:
        from video_depth_anything.video_depth_stream import VideoDepthAnything
    except Exception as e:
        print(f"[DEPTH] Import VDA streaming échoué : {e}")
        return False
    weights = _ensure_vda_weights(variant, weights_dir)
    if weights is None:
        return False
    try:
        model = VideoDepthAnything(**_VDA_CONFIGS[variant])
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[DEPTH] VDA state_dict : {len(missing)} manquant(s) / "
                f"{len(unexpected)} inattendu(s)"
            )
        model.to(device).eval()
    except Exception as e:
        print(f"[DEPTH] Init modèle VDA échouée : {e}")
        return False
    _depth_model = model
    _depth_device = device
    _depth_backend = "vda-small" if variant == "vits" else "vda-base"
    print(
        f"[DEPTH] Video-Depth-Anything Metric ({variant}) chargé sur {device} "
        f"— sortie en mètres"
    )
    return True


def _try_load_da_v2(device: str) -> bool:
    """Tier 3 : fallback historique sur Depth-Anything-V2-Small (HF transformers)."""
    global _depth_model, _depth_processor, _depth_device, _depth_backend
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print("[DEPTH] Fallback sur Depth-Anything-V2-Small...")
        _depth_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        _depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        _depth_model.to(device).eval()
        _depth_device = device
        _depth_backend = "da-v2"
        print(
            f"[DEPTH] Depth-Anything-V2-Small chargé sur {device} "
            f"— disparité relative normalisée par frame"
        )
        return True
    except Exception as e:
        print(f"[DEPTH] Fallback DA-V2 échoué : {e}")
        return False


def load_depth_model(
    device: str = "cpu",
    model: str = "auto",
    infer_every: int = 3,
    weights_dir: Optional[Path] = None,
) -> bool:
    """Charge un backend de depth estimation.

    Args:
        device: ``"cpu"`` / ``"cuda"`` / ``"cuda:0"``.
        model: ``"auto"`` (VDA-Small → VDA-Base → DA-V2), ``"vda-small"``,
               ``"vda-base"`` ou ``"da-v2"``.
        infer_every: n'exécute le forward qu'une frame sur K (cache la
                     précédente entre deux). ``1`` = aucune throttling.
        weights_dir: où cacher les poids VDA (défaut ``ai/models/``).

    Returns:
        ``True`` si au moins un backend a été chargé.
    """
    global _depth_infer_every, _depth_call_count
    global _last_depth_meters, _last_depth_norm
    _depth_infer_every = max(1, int(infer_every))
    _depth_call_count = 0
    _last_depth_meters = None
    _last_depth_norm = None

    weights_dir = Path(weights_dir) if weights_dir is not None else _DEFAULT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    if model not in DEPTH_MODEL_CHOICES:
        raise ValueError(
            f"--depth-model={model!r} invalide, choisir parmi {DEPTH_MODEL_CHOICES}"
        )

    if model == "auto":
        plan = ["vits", "vitb", "da-v2"]
    elif model == "vda-small":
        plan = ["vits"]
    elif model == "vda-base":
        plan = ["vitb"]
    else:  # da-v2
        plan = ["da-v2"]

    for step in plan:
        if step == "da-v2":
            if _try_load_da_v2(device):
                return True
            continue
        if _try_load_vda_transformers(step, device):
            return True
        if _try_load_vda_repo(step, device, weights_dir):
            return True

    print("[DEPTH] Tous les backends ont échoué")
    return False


def reset_depth_stream() -> None:
    """À appeler au début de chaque épisode : vide le cache temporel VDA et le
    compteur de throttling. Sans effet sur DA-V2 (stateless)."""
    global _depth_call_count, _last_depth_meters, _last_depth_norm
    _depth_call_count = 0
    _last_depth_meters = None
    _last_depth_norm = None
    if _depth_model is None or _depth_backend is None:
        return
    if _depth_backend.startswith("vda"):
        _depth_model.transform = None
        _depth_model.frame_cache_list = []
        _depth_model.frame_id_list = []
        _depth_model.id = -1


def depth_backend() -> Optional[str]:
    """Nom du backend courant (``"vda-small"``, ``"vda-base"`` ou ``"da-v2"``)."""
    return _depth_backend


def _infer_vda_meters(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    device_type = "cuda" if _depth_device.startswith("cuda") else "cpu"
    depth_m = _depth_model.infer_video_depth_one(
        frame_rgb, input_size=518, device=_depth_device, fp32=(device_type == "cpu"),
    )
    return depth_m.astype(np.float32)


def _infer_da_v2_raw(frame_bgr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = _depth_processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(_depth_device) for k, v in inputs.items()}
    with torch.no_grad():
        raw = _depth_model(**inputs).predicted_depth.squeeze().cpu().float().numpy()
    return raw.astype(np.float32)


def _meters_to_norm(depth_m: np.ndarray) -> np.ndarray:
    """Mètres → carte normalisée [0,1] disparité-like (plus haut = plus proche)."""
    disp = 1.0 / np.maximum(depth_m, 1e-3)
    lo, hi = float(disp.min()), float(disp.max())
    return ((disp - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _raw_to_norm(raw: np.ndarray) -> np.ndarray:
    lo, hi = float(raw.min()), float(raw.max())
    return ((raw - lo) / (hi - lo + 1e-8)).astype(np.float32)


def infer_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """Carte de depth HxW float32 normalisée [0,1], plus haut = plus proche.

    Appelée à chaque frame — throttlée par ``infer_every`` (cf. ``load_depth_model``).
    Pour obtenir les mètres bruts (VDA uniquement), utiliser ``infer_depth_meters``.
    """
    global _depth_call_count, _last_depth_meters, _last_depth_norm
    _depth_call_count += 1

    frame_h, frame_w = frame_bgr.shape[:2]
    should_infer = ((_depth_call_count - 1) % _depth_infer_every) == 0 or _last_depth_norm is None

    if not should_infer:
        if _last_depth_norm.shape == (frame_h, frame_w):
            return _last_depth_norm
        return cv2.resize(_last_depth_norm, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

    if _depth_backend in ("vda-small", "vda-base"):
        depth_m = _infer_vda_meters(frame_bgr)
        _last_depth_meters = depth_m
        norm = _meters_to_norm(depth_m)
    elif _depth_backend == "da-v2":
        raw = _infer_da_v2_raw(frame_bgr)
        _last_depth_meters = None
        norm = _raw_to_norm(raw)
    else:
        raise RuntimeError(
            "infer_depth() appelée avant load_depth_model() (backend non chargé)"
        )

    if norm.shape != (frame_h, frame_w):
        norm = cv2.resize(norm, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    _last_depth_norm = norm
    return norm


def infer_depth_meters(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Depth métrique HxW float32 en **mètres**, ou ``None`` si le backend
    courant ne produit pas de depth métrique (DA-V2 fallback).

    Respecte le throttling ``infer_every`` en partageant le cache avec
    ``infer_depth``.
    """
    if _depth_backend is None or _depth_backend == "da-v2":
        return None
    infer_depth(frame_bgr)  # met à jour _last_depth_meters si nécessaire
    return _last_depth_meters


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


# ── CLI / benchmark ───────────────────────────────────────────────────────────

def _benchmark(n_frames: int, frame_size: tuple[int, int]) -> float:
    """Exécute n_frames inférences sur frames aléatoires, retourne les FPS."""
    import time
    w, h = frame_size
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n_frames)]
    # warmup (1 forward pour initialiser la transform VDA + compile eventuels kernels)
    _ = infer_depth(frames[0])
    reset_depth_stream()
    t0 = time.perf_counter()
    for f in frames:
        _ = infer_depth(f)
    dt = time.perf_counter() - t0
    fps = n_frames / dt
    print(
        f"[BENCH] backend={_depth_backend} device={_depth_device} "
        f"frames={n_frames} infer_every={_depth_infer_every} "
        f"→ {fps:.1f} FPS ({dt * 1000 / n_frames:.1f} ms/frame)"
    )
    if fps < 10.0:
        print(f"[BENCH] ⚠ en-dessous du plancher 10 FPS — remonter au user")
    elif fps < 15.0:
        print(f"[BENCH] ⚠ en-dessous de la cible 15 FPS (plancher 10 OK)")
    else:
        print(f"[BENCH] ✓ au-dessus de la cible 15 FPS")
    return fps


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Smoke-test / benchmark du backend de depth estimation"
    )
    ap.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0")
    ap.add_argument(
        "--depth-model", default="auto", choices=DEPTH_MODEL_CHOICES,
        help="Force un backend (défaut: auto = VDA-Small → VDA-Base → DA-V2)",
    )
    ap.add_argument(
        "--depth-infer-every", type=int, default=3,
        help="Exécute le forward 1 frame sur K (défaut 3)",
    )
    ap.add_argument("--benchmark", action="store_true", help="Mesure FPS sur N frames")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument(
        "--frame-size", nargs=2, type=int, metavar=("W", "H"), default=[320, 240],
    )
    args = ap.parse_args()

    ok = load_depth_model(
        device=args.device, model=args.depth_model, infer_every=args.depth_infer_every,
    )
    if not ok:
        return 2

    if args.benchmark:
        _benchmark(args.n_frames, tuple(args.frame_size))
    else:
        print(f"[DEPTH] Backend actif : {_depth_backend} — prêt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

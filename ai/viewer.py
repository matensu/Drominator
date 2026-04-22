"""
Visualisation en direct de l'entraînement PPO.
Lance dans un terminal séparé pendant que l'entraînement tourne.
Recharge automatiquement le dernier checkpoint toutes les 30s.

Modes :
  (défaut)    → pipeline YOLO+Depth  — même image que le drone voit pendant l'entraînement
  --vision    → image Panda3D 84×84  — sans YOLO ni Depth
  --raycast   → raycasts analytiques  — le mode le plus rapide

Usage :
  python -m ai.viewer
  python -m ai.viewer --vision
  python -m ai.viewer --raycast
  python -m ai.viewer --model path/to/model.zip
"""
import os, sys, time, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RELOAD_SECS   = 30
_BASE         = os.path.dirname(__file__)

VISION_MODE   = "--vision"   in sys.argv
RAYCAST_MODE  = "--raycast"  in sys.argv
# défaut = pipeline

_model_arg = None
if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    if idx + 1 < len(sys.argv):
        _model_arg = sys.argv[idx + 1]

if   RAYCAST_MODE: MODEL_DIR, MODEL_PFX = os.path.join(_BASE, "models"),          "drone_ppo"
elif VISION_MODE:  MODEL_DIR, MODEL_PFX = os.path.join(_BASE, "models_vision"),   "drone_vision"
else:              MODEL_DIR, MODEL_PFX = os.path.join(_BASE, "models_pipeline"), "drone_pipeline"


def latest_model() -> str | None:
    if _model_arg:
        return _model_arg if os.path.exists(_model_arg) else None
    final = os.path.join(MODEL_DIR, f"{MODEL_PFX}_final.zip")
    if os.path.exists(final):
        return final
    ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, f"{MODEL_PFX}_*.zip")))
    return ckpts[-1] if ckpts else None


def load_model(path: str):
    from stable_baselines3 import PPO
    print(f"[VIEWER] Modèle chargé : {os.path.basename(path)}")
    return PPO.load(path, device="cpu")


def draw_rays(frame: np.ndarray, rays: np.ndarray) -> np.ndarray:
    W   = frame.shape[1]
    bar = np.zeros((22, W, 3), dtype=np.uint8)
    n   = len(rays)
    for i, v in enumerate(rays):
        x0 = int(i / n * W)
        x1 = int((i + 1) / n * W)
        g  = int(v * 255)
        r  = max(0, 255 - g * 3)
        cv2.rectangle(bar, (x0, 2), (x1, 20), (0, g, r), -1)
    cv2.putText(bar, "RAYCASTS (vert=loin  rouge=proche)",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return np.vstack([frame, bar])


def draw_minimap(env, W: int = 200, H: int = 200) -> np.ndarray:
    import math
    from ai.corridor import CORRIDOR_W, CORRIDOR_H
    img = np.zeros((H, W, 3), dtype=np.uint8)

    pos = env._pos
    view_x, view_y = 3.5, 8.0

    def to_px(wx, wy):
        px = int((wx - pos[0] + view_x) / (2 * view_x) * W)
        py = int((1 - (wy - pos[1] + view_y * 0.3) / view_y) * H)
        return px, py

    lwall, _ = to_px(-CORRIDOR_W / 2, pos[1])
    rwall, _ = to_px( CORRIDOR_W / 2, pos[1])
    cv2.line(img, (lwall, 0), (lwall, H), (80, 80, 80), 2)
    cv2.line(img, (rwall, 0), (rwall, H), (80, 80, 80), 2)

    for aabbs in env._aabbs.values():
        for (x0, x1, y0, y1, z0, z1) in aabbs:
            oy = (y0 + y1) / 2
            if abs(oy - pos[1]) > view_y:
                continue
            col = (60, 160, 60) if z1 > CORRIDOR_H * 0.5 else (60, 100, 200)
            cv2.rectangle(img, to_px(x0, y0), to_px(x1, y1), col, -1)

    dp = to_px(pos[0], pos[1])
    cv2.circle(img, dp, 5, (0, 255, 255), -1)

    hr = math.radians(env._yaw)
    dx = int(-math.sin(hr) * 14)
    dy = int(-math.cos(hr) * 14)
    cv2.arrowedLine(img, dp, (dp[0] + dx, dp[1] - dy), (0, 255, 255), 2, tipLength=0.4)
    cv2.putText(img, "TOP VIEW", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    return img


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Chargement de l'environnement ─────────────────────────────────────────
    if RAYCAST_MODE:
        from ai.drone_env     import DroneCorridorEnv
        from sim.corridor_sim import CorridorSim
        env = DroneCorridorEnv(seed=0)
        obs, _ = env.reset()
        sim = CorridorSim(env)
        is_image_obs = False
        print("[VIEWER] Mode RAYCAST — raycasts analytiques + Panda3D")

    elif VISION_MODE:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "window-type offscreen\nwin-size 84 84\naudio-library-name null\nsync-video 0\n")
        from ai.train_vision import DroneVisionEnv
        env = DroneVisionEnv(seed=0)
        obs, _ = env.reset()
        sim = None
        is_image_obs = True
        print("[VIEWER] Mode VISION — image Panda3D 84×84")

    else:
        # ── Mode par défaut : pipeline complet YOLO + Depth (= ce que le drone voit) ──
        from ai.train_full_pipeline import FullPipelineEnv
        env = FullPipelineEnv(seed=0)
        obs, _ = env.reset()
        sim = None
        is_image_obs = True
        print("[VIEWER] Mode PIPELINE — image annotée YOLO+Depth (= observation du drone)")

    cv2.namedWindow("Drominator — Live Training Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drominator — Live Training Viewer", 1040, 560)

    model        = None
    model_path   = None
    last_reload  = 0.0
    episode      = 1
    steps        = 0
    total_reward = 0.0
    best_dist    = 0.0
    fps_disp     = 0.0
    t_prev       = time.perf_counter()

    print(f"[VIEWER] Attend un modèle dans {MODEL_DIR} …")

    while True:
        now = time.perf_counter()

        # ── Rechargement automatique du modèle ────────────────────────────────
        if now - last_reload > RELOAD_SECS:
            mp = latest_model()
            if mp and mp != model_path:
                try:
                    model      = load_model(mp)
                    model_path = mp
                except Exception as e:
                    print(f"[VIEWER] Erreur chargement : {e}")
            last_reload = now

        # ── Inférence ─────────────────────────────────────────────────────────
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        steps        += 1
        best_dist     = max(best_dist, env._pos[1])

        # ── Rendu principal ───────────────────────────────────────────────────
        if sim is not None:
            # mode raycast : rendu Panda3D via CorridorSim
            frame = sim.sync_and_render()
        else:
            # mode image : obs EST déjà l'image, on upscale pour l'affichage
            frame = cv2.resize(obs, (640, 480), interpolation=cv2.INTER_NEAREST)

        fps_disp = 0.9 * fps_disp + 0.1 / max(now - t_prev, 1e-6)
        t_prev   = now

        # ── Overlays texte ────────────────────────────────────────────────────
        model_name = os.path.basename(model_path) if model_path else "ALÉATOIRE"
        status_col = (0, 255, 0) if model_path else (0, 100, 255)
        pos        = env._pos

        if RAYCAST_MODE:
            from ai.corridor import N_RAYS, RAY_DIST
            rays  = obs[:N_RAYS]
            min_d = float(rays.min()) * RAY_DIST
            obs_label = f"Obstacle : {min_d:.2f} m"
            obs_col   = (0,255,100) if min_d > 2.0 else (0,160,255) if min_d > 1.0 else (0,60,255)
        else:
            rays      = np.array([])
            min_d     = 0.0
            obs_label = "Image 84x84 annotée"
            obs_col   = (180, 180, 180)

        lines = [
            (f"FPS {fps_disp:.0f}",                 (0, 255, 0)),
            (f"Modele : {model_name}",               status_col),
            (f"Episode {episode}  |  step {steps}",  (200, 200, 200)),
            (f"Distance : {pos[1]:.1f} m",           (0, 220, 255)),
            (f"Reward   : {total_reward:.1f}",       (255, 210, 0)),
            (f"Record   : {best_dist:.1f} m",        (255, 150, 255)),
            (obs_label,                              obs_col),
        ]
        for i, (txt, col) in enumerate(lines):
            cv2.putText(frame, txt, (8, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        secs_left = max(0, int(RELOAD_SECS - (now - last_reload)))
        cv2.putText(frame, f"Reload dans {secs_left}s",
                    (8, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # ── Minimap + assemblage ──────────────────────────────────────────────
        minimap  = draw_minimap(env)
        minimap  = cv2.resize(minimap, (200, frame.shape[0]))
        combined = np.hstack([frame, minimap])

        if RAYCAST_MODE and len(rays) > 0:
            combined = draw_rays(combined, rays)

        cv2.imshow("Drominator — Live Training Viewer", combined)

        # ── Reset si épisode terminé ──────────────────────────────────────────
        if terminated:
            print(f"[VIEWER] Ep {episode:4d} | {pos[1]:.1f} m | reward {total_reward:.1f}")
            episode     += 1
            steps        = 0
            total_reward = 0.0
            obs, _       = env.reset()
            if sim is not None:
                sim.clear_chunks()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Géométrie partagée du couloir infini.
Utilisé par drone_env.py (entraînement) ET corridor_sim.py (visualisation).
"""
import math
import random
import numpy as np

CORRIDOR_W = 5.0    # largeur totale (X : -2.5 → +2.5)
CORRIDOR_H = 3.0    # hauteur totale (Z :  0.0 → +3.0)
CHUNK_LEN  = 20.0   # longueur d'un segment en Y
DRONE_R    = 0.22   # rayon de collision
N_RAYS     = 48     # 6 élévations × 8 azimuths
RAY_DIST   = 9.0    # portée max (m)
DT         = 1/30.
MAX_SPD    = 6.0    # m/s
MAX_YAW    = 90.0   # deg/s


# ── Génération des obstacles ───────────────────────────────────────────────────

def gen_chunk(ci: int, rng: random.Random) -> list[tuple]:
    """
    Génère les obstacles du chunk ci.
    Retourne une liste de tuples (ox, oy, oz, w, d, h, r, g, b).
    """
    y0      = ci * CHUNK_LEN
    y_start = y0 + (4.0 if ci == 0 else 2.0)
    y_end   = y0 + CHUNK_LEN - 2.0
    hw      = CORRIDOR_W / 2
    items   = []

    for _ in range(rng.randint(4, 9)):
        oy = rng.uniform(y_start, y_end)
        t  = rng.choice(['pillar', 'wall', 'crate', 'bar', 'arch'])

        if t == 'pillar':
            w  = rng.uniform(0.25, 0.65)
            d  = rng.uniform(0.25, 0.55)
            h  = rng.uniform(1.0,  CORRIDOR_H)
            ox = rng.uniform(-hw + 0.5, hw - 0.5)
            oz = h / 2
            c  = (rng.uniform(.45,.75), rng.uniform(.35,.60), rng.uniform(.30,.55))
            items.append((ox, oy, oz, w, d, h, *c))

        elif t == 'wall':
            w  = rng.uniform(1.5, hw * 1.3)
            d  = rng.uniform(0.15, 0.38)
            h  = rng.uniform(0.6, CORRIDOR_H * 0.90)
            ox = rng.choice([-1, 1]) * (hw - w / 2)
            oz = h / 2
            c  = (rng.uniform(.3,.6), rng.uniform(.4,.7), rng.uniform(.3,.6))
            items.append((ox, oy, oz, w, d, h, *c))

        elif t == 'crate':
            s  = rng.uniform(0.35, 1.10)
            ox = rng.uniform(-hw + 0.5, hw - 0.5)
            oz = s / 2
            c  = (rng.uniform(.15,.35), rng.uniform(.38,.58), rng.uniform(.15,.32))
            items.append((ox, oy, oz, s, s, s, *c))

        elif t == 'bar':
            w  = CORRIDOR_W * rng.uniform(0.50, 0.85)
            d  = rng.uniform(0.10, 0.26)
            h  = rng.uniform(0.12, 0.36)
            ox = rng.uniform(-0.5, 0.5)
            oz = rng.uniform(0.30, CORRIDOR_H - 0.20)
            c  = (rng.uniform(.55,.85), rng.uniform(.38,.58), rng.uniform(.18,.38))
            items.append((ox, oy, oz, w, d, h, *c))

        elif t == 'arch':
            pw  = rng.uniform(0.18, 0.42)
            pd  = rng.uniform(0.18, 0.38)
            h   = rng.uniform(CORRIDOR_H * 0.55, CORRIDOR_H)
            oz  = h / 2
            gap = rng.uniform(0.85, 1.45)
            c   = (rng.uniform(.5,.8), rng.uniform(.4,.65), rng.uniform(.3,.5))
            for ox in [-(gap / 2 + pw / 2), gap / 2 + pw / 2]:
                items.append((ox, oy, oz, pw, pd, h, *c))

    return items


def items_to_aabbs(items: list[tuple]) -> list[tuple]:
    """Convertit (ox,oy,oz,w,d,h,...) → (x0,x1,y0,y1,z0,z1)."""
    return [
        (ox - w/2, ox + w/2,
         oy - d/2, oy + d/2,
         oz - h/2, oz + h/2)
        for (ox, oy, oz, w, d, h, *_) in items
    ]


# ── Physique drone ─────────────────────────────────────────────────────────────

def apply_action(pos: list, yaw: float, action) -> tuple[list, float]:
    """
    Applique action=[fwd,strafe,vert,yaw_rate] ∈ [-1,1].
    Retourne (new_pos, new_yaw). Clamp corridor inclus.
    """
    fwd    = float(action[0]) * MAX_SPD
    strafe = float(action[1]) * MAX_SPD
    vert   = float(action[2]) * MAX_SPD
    dyaw   = float(action[3]) * MAX_YAW

    hr = math.radians(yaw)
    nx = pos[0] + (-math.sin(hr) * fwd + math.cos(hr) * strafe) * DT
    ny = pos[1] + ( math.cos(hr) * fwd + math.sin(hr) * strafe) * DT
    nz = pos[2] + vert * DT
    ny_yaw = yaw + dyaw * DT

    hw = CORRIDOR_W / 2 - DRONE_R
    nx = max(-hw, min(hw, nx))
    nz = max(DRONE_R, min(CORRIDOR_H - DRONE_R, nz))

    return [nx, ny, nz], ny_yaw


# ── Collision ─────────────────────────────────────────────────────────────────

def check_collision(pos: list, aabb_map: dict) -> bool:
    px, py, pz = pos

    # Mur gauche / droite
    if abs(px) >= CORRIDOR_W / 2 - DRONE_R:
        return True

    # Sol / plafond
    if pz <= DRONE_R or pz >= CORRIDOR_H - DRONE_R:
        return True

    # Aller en Y négatif = reculer derrière le départ
    if py <= 0:
        return True

    # Obstacles
    for aabbs in aabb_map.values():
        for (x0, x1, y0, y1, z0, z1) in aabbs:
            if (x0 - DRONE_R < px < x1 + DRONE_R and
                    y0 - DRONE_R < py < y1 + DRONE_R and
                    z0 - DRONE_R < pz < z1 + DRONE_R):
                return True

    return False


# ── Raycast ───────────────────────────────────────────────────────────────────

def raycast(pos: list, yaw: float, aabb_map: dict) -> np.ndarray:
    """Retourne N_RAYS distances normalisées [0,1]."""
    px, py, pz = pos
    hr    = math.radians(yaw)
    fwd   = np.array([-math.sin(hr),  math.cos(hr), 0.0])
    right = np.array([ math.cos(hr),  math.sin(hr), 0.0])
    up    = np.array([0.0, 0.0, 1.0])

    out = np.ones(N_RAYS, dtype=np.float32)
    idx = 0
    for vi in range(6):                       # élévations : -30° → +30°
        vp = math.radians(-30 + vi * 12)
        cv, sv = math.cos(vp), math.sin(vp)
        for hi in range(8):                   # azimuths : -70° → +70°
            ha = math.radians(-70 + hi * 20)
            d  = cv * (math.cos(ha) * fwd + math.sin(ha) * right) + sv * up
            d  = d / np.linalg.norm(d)
            out[idx] = _ray_dist(px, py, pz, d, aabb_map) / RAY_DIST
            idx += 1
    return np.clip(out, 0.0, 1.0)


def _ray_dist(ox, oy, oz, d, aabb_map) -> float:
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    hw = CORRIDOR_W / 2
    tmin = RAY_DIST

    # Murs du couloir
    for axis, val in [('x', -hw), ('x', hw), ('z', 0.), ('z', CORRIDOR_H), ('y', 0.)]:
        if   axis == 'x' and abs(dx) > 1e-9: t = (val - ox) / dx
        elif axis == 'z' and abs(dz) > 1e-9: t = (val - oz) / dz
        elif axis == 'y' and abs(dy) > 1e-9: t = (val - oy) / dy
        else: continue
        if 0 < t < tmin:
            tmin = t

    # Obstacles
    for aabbs in aabb_map.values():
        for bb in aabbs:
            t = _ray_aabb(ox, oy, oz, dx, dy, dz, bb)
            if t is not None and t < tmin:
                tmin = t

    return tmin


def _ray_aabb(ox, oy, oz, dx, dy, dz, bb) -> float | None:
    """Slab method — retourne la distance d'entrée ou None."""
    x0, x1, y0, y1, z0, z1 = bb
    tmin, tmax = -1e9, 1e9

    for o, dv, lo, hi in [(ox, dx, x0, x1), (oy, dy, y0, y1), (oz, dz, z0, z1)]:
        if abs(dv) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / dv, (hi - o) / dv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None

    t = tmin if tmin > 1e-4 else tmax
    return t if t > 1e-4 else None

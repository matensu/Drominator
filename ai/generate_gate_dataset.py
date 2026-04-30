"""
Génération d'un dataset YOLO de "gates" (portails rectangulaires traversables)
en réutilisant le rendu Panda3D du couloir infini.

Le script ne modifie ni `ai/corridor.py` ni `sim/corridor_sim.py` : il étend
`CorridorSim` via une sous-classe locale qui ajoute le support des gates,
la randomisation des lumières/murs et une caméra pilotable librement.
"""
from __future__ import annotations

import argparse
import colorsys
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from panda3d.core import (  # noqa: E402
    AmbientLight, DirectionalLight, LColor, LPoint3, NodePath,
)

from ai.corridor import CORRIDOR_W, CORRIDOR_H, CHUNK_LEN  # noqa: E402
from ai.drone_env import DroneCorridorEnv  # noqa: E402
from sim.corridor_sim import CorridorSim  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Extension locale de CorridorSim
# ─────────────────────────────────────────────────────────────────────────────


class GateCorridorSim(CorridorSim):
    """CorridorSim + gate (4 barres) + randomisation + caméra libre."""

    def __init__(self, env):
        self._static_nodes: list[NodePath] = []
        self._amb: AmbientLight | None = None
        self._amb_np: NodePath | None = None
        self._sun: DirectionalLight | None = None
        self._sun_np: NodePath | None = None

        super().__init__(env)

        self._gate_bars: list[NodePath] = []
        for _ in range(4):
            n = self.loader.loadModel("models/misc/rgbCube")
            n.reparentTo(self.render)
            n.hide()
            self._gate_bars.append(n)
        self._gate_aabb: tuple[float, float, float, float, float, float] | None = None

    def _box(self, x, y, z, sx, sy, sz, r, g, b):
        n = super()._box(x, y, z, sx, sy, sz, r, g, b)
        self._static_nodes.append(n)
        return n

    def _setup_lights(self):
        self._amb = AmbientLight("amb")
        self._amb.setColor(LColor(0.42, 0.42, 0.46, 1))
        self._amb_np = self.render.attachNewNode(self._amb)
        self.render.setLight(self._amb_np)

        self._sun = DirectionalLight("sun")
        self._sun.setColor(LColor(0.72, 0.68, 0.58, 1))
        self._sun_np = self.render.attachNewNode(self._sun)
        self._sun_np.setHpr(35, -50, 0)
        self.render.setLight(self._sun_np)

    # ── Domain randomization ─────────────────────────────────────────────────

    def randomize_lights(self, rng: random.Random) -> None:
        self._amb.setColor(LColor(
            rng.uniform(0.2, 0.6),
            rng.uniform(0.2, 0.6),
            rng.uniform(0.2, 0.6),
            1,
        ))
        self._sun.setColor(LColor(
            rng.uniform(0.5, 1.0),
            rng.uniform(0.5, 1.0),
            rng.uniform(0.5, 1.0),
            1,
        ))
        self._sun_np.setHpr(
            rng.uniform(0, 360),
            rng.uniform(-80, -10),
            0,
        )

    def randomize_walls(self, rng: random.Random) -> None:
        for n in self._static_nodes:
            n.setColor(
                rng.uniform(0.2, 0.6),
                rng.uniform(0.2, 0.6),
                rng.uniform(0.2, 0.6),
                1,
            )

    # ── Gate ─────────────────────────────────────────────────────────────────

    def place_gate(
        self,
        gx: float, gy: float, gz: float,
        gw: float, gh: float, gd: float,
        thickness: float,
        color: tuple[float, float, float],
    ) -> None:
        r, g, b = color
        inner_h = max(0.05, gh - 2 * thickness)

        top, bottom, left, right = self._gate_bars

        top.setPos(gx, gy, gz + gh / 2 - thickness / 2)
        top.setScale(gw, gd, thickness)

        bottom.setPos(gx, gy, gz - gh / 2 + thickness / 2)
        bottom.setScale(gw, gd, thickness)

        left.setPos(gx - gw / 2 + thickness / 2, gy, gz)
        left.setScale(thickness, gd, inner_h)

        right.setPos(gx + gw / 2 - thickness / 2, gy, gz)
        right.setScale(thickness, gd, inner_h)

        for bar in self._gate_bars:
            bar.setColor(r, g, b, 1)
            bar.show()

        self._gate_aabb = (gx, gy, gz, gw, gd, gh)

    def hide_gate(self) -> None:
        for bar in self._gate_bars:
            bar.hide()
        self._gate_aabb = None

    # ── Rendu avec caméra explicite ──────────────────────────────────────────

    def render_to_array(
        self,
        cam_pos: tuple[float, float, float],
        cam_hpr: tuple[float, float, float],
    ) -> np.ndarray:
        self._sync_obstacles()
        self.camera.setPos(*cam_pos)
        self.camera.setHpr(*cam_hpr)
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()
        raw = self._tex.getRamImageAs("RGB")
        if raw:
            arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(self.H, self.W, 3)
            return np.ascontiguousarray(np.flipud(arr[:, :, ::-1]))
        return np.zeros((self.H, self.W, 3), dtype=np.uint8)

    # ── Projection 3D → 2D ───────────────────────────────────────────────────

    def _fov_tan_half(self) -> tuple[float, float]:
        fov = self.camLens.getFov()
        return (
            math.tan(math.radians(fov[0]) / 2),
            math.tan(math.radians(fov[1]) / 2),
        )

    def _project(self, world_pt: tuple[float, float, float]) -> tuple[float, float] | None:
        """Projection manuelle, robuste hors-frustum (tant que devant la caméra)."""
        cam_rel = self.cam.getRelativePoint(self.render, LPoint3(*world_pt))
        x, y, z = cam_rel.x, cam_rel.y, cam_rel.z
        if y <= 0.01:
            return None
        tx, ty = self._fov_tan_half()
        ndc_x = (x / y) / tx
        ndc_y = (z / y) / ty
        u = (ndc_x + 1.0) * 0.5 * self.W
        v = (1.0 - ndc_y) * 0.5 * self.H
        return (u, v)

    def gate_bbox_2d(self) -> tuple[float, float, float, float] | None:
        """Bbox pixel (x0, y0, x1, y1), clippé à l'image. None si rejeté."""
        if self._gate_aabb is None:
            return None
        gx, gy, gz, gw, gd, gh = self._gate_aabb

        pts: list[tuple[float, float]] = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    wp = (gx + sx * gw / 2, gy + sy * gd / 2, gz + sz * gh / 2)
                    px = self._project(wp)
                    if px is None:
                        return None
                    pts.append(px)

        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        x0, x1 = min(us), max(us)
        y0, y1 = min(vs), max(vs)

        raw_area = max(1e-6, (x1 - x0) * (y1 - y0))

        cx0 = max(0.0, min(float(self.W), x0))
        cx1 = max(0.0, min(float(self.W), x1))
        cy0 = max(0.0, min(float(self.H), y0))
        cy1 = max(0.0, min(float(self.H), y1))

        clipped_area = max(0.0, (cx1 - cx0) * (cy1 - cy0))
        if clipped_area < 100:
            return None
        if clipped_area / raw_area < 0.3:
            return None

        return (cx0, cy0, cx1, cy1)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────


def _point_in_any_aabb(pt, aabb_map, margin: float = 0.3) -> bool:
    px, py, pz = pt
    for aabbs in aabb_map.values():
        for (x0, x1, y0, y1, z0, z1) in aabbs:
            if (x0 - margin < px < x1 + margin and
                    y0 - margin < py < y1 + margin and
                    z0 - margin < pz < z1 + margin):
                return True
    return False


def _sample_gate_params(rng: random.Random):
    gate_w = rng.uniform(1.0, 2.2)
    gate_h = rng.uniform(1.0, 2.2)
    gate_d = rng.uniform(0.15, 0.35)
    thickness = rng.uniform(0.08, 0.18)
    thickness = min(thickness, gate_h / 2 - 0.1, gate_w / 2 - 0.1)
    thickness = max(0.05, thickness)

    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.5, 1.0)
    val = rng.uniform(0.6, 1.0)
    color = colorsys.hsv_to_rgb(hue, sat, val)
    return gate_w, gate_h, gate_d, thickness, color


def _try_gate_frame(sim: GateCorridorSim, env: DroneCorridorEnv,
                    rng: random.Random, max_attempts: int = 20):
    hw = CORRIDOR_W / 2
    for _ in range(max_attempts):
        ci = rng.randint(1, 3)
        y_lo = ci * CHUNK_LEN + 2.0
        y_hi = (ci + 1) * CHUNK_LEN - 2.0

        gw, gh, gd, thickness, color = _sample_gate_params(rng)
        gx = rng.uniform(-(hw - gw / 2 - 0.1), hw - gw / 2 - 0.1)
        gy = rng.uniform(y_lo, y_hi)
        gz = rng.uniform(gh / 2 + 0.05, CORRIDOR_H - gh / 2 - 0.05)

        if _point_in_any_aabb((gx, gy, gz), env.aabbs, margin=0.0):
            continue

        dist = rng.uniform(0.5, 15.0)
        cam_y = gy - dist
        if cam_y < 0.5:
            continue
        cam_x = gx + rng.uniform(-1.0, 1.0)
        cam_z = gz + rng.uniform(-0.6, 0.6)
        cam_x = max(-hw + 0.25, min(hw - 0.25, cam_x))
        cam_z = max(0.35, min(CORRIDOR_H - 0.35, cam_z))

        if _point_in_any_aabb((cam_x, cam_y, cam_z), env.aabbs):
            continue

        dx = gx - cam_x
        dy = gy - cam_y
        dz = gz - cam_z
        base_yaw = math.degrees(math.atan2(-dx, dy))
        horiz = math.sqrt(dx * dx + dy * dy)
        base_pitch = math.degrees(math.atan2(dz, horiz))

        yaw = base_yaw + rng.uniform(-15.0, 15.0)
        pitch = base_pitch + rng.uniform(-15.0, 15.0)
        roll = rng.uniform(-15.0, 15.0)

        sim.place_gate(gx, gy, gz, gw, gh, gd, thickness, color)
        img = sim.render_to_array((cam_x, cam_y, cam_z), (yaw, pitch, roll))
        bbox = sim.gate_bbox_2d()
        if bbox is None:
            continue
        return img, bbox

    return None


def _try_empty_frame(sim: GateCorridorSim, env: DroneCorridorEnv,
                     rng: random.Random, max_attempts: int = 20):
    sim.hide_gate()
    hw = CORRIDOR_W / 2
    for _ in range(max_attempts):
        cam_y = rng.uniform(1.0, 4 * CHUNK_LEN - 5.0)
        cam_x = rng.uniform(-hw + 0.3, hw - 0.3)
        cam_z = rng.uniform(0.5, CORRIDOR_H - 0.3)
        if _point_in_any_aabb((cam_x, cam_y, cam_z), env.aabbs):
            continue
        yaw = rng.uniform(-45.0, 45.0)
        pitch = rng.uniform(-15.0, 15.0)
        roll = rng.uniform(-15.0, 15.0)
        return sim.render_to_array((cam_x, cam_y, cam_z), (yaw, pitch, roll))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


class GateDatasetBuilder:
    EMPTY_FRACTION = 0.15

    def __init__(self, output_dir: Path, base_seed: int):
        self.output_dir = Path(output_dir)
        self.env = DroneCorridorEnv(seed=base_seed)
        self.sim = GateCorridorSim(self.env)

    def _reset_scene(self, scene_seed: int) -> None:
        self.env.reset(seed=scene_seed)

    def _write_label(self, path: Path, bbox) -> None:
        if bbox is None:
            path.write_text("")
            return
        x0, y0, x1, y1 = bbox
        W, H = self.sim.W, self.sim.H
        cx = (x0 + x1) / 2.0 / W
        cy = (y0 + y1) / 2.0 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def generate_split(self, split: str, n: int, split_seed: int) -> None:
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(split_seed)
        written = 0
        attempts = 0
        max_global = max(100, n * 10)

        while written < n and attempts < max_global:
            attempts += 1
            self._reset_scene(rng.randint(0, 2**31 - 1))
            self.sim.randomize_lights(rng)
            self.sim.randomize_walls(rng)

            if rng.random() < self.EMPTY_FRACTION:
                img = _try_empty_frame(self.sim, self.env, rng)
                if img is None:
                    continue
                bbox = None
            else:
                res = _try_gate_frame(self.sim, self.env, rng)
                if res is None:
                    continue
                img, bbox = res

            fid = f"{written:08d}"
            cv2.imwrite(str(img_dir / f"{fid}.jpg"), img)
            self._write_label(lbl_dir / f"{fid}.txt", bbox)
            written += 1

            if written % 100 == 0 or written == n:
                print(f"  [{split}] {written}/{n}")

        if written < n:
            print(f"  [{split}] WARNING: only wrote {written}/{n} frames "
                  f"after {attempts} attempts")

    def generate_preview(self, n: int, preview_seed: int) -> None:
        preview_dir = self.output_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(preview_seed)
        written = 0
        attempts = 0
        max_global = max(50, n * 10)

        while written < n and attempts < max_global:
            attempts += 1
            self._reset_scene(rng.randint(0, 2**31 - 1))
            self.sim.randomize_lights(rng)
            self.sim.randomize_walls(rng)

            res = _try_gate_frame(self.sim, self.env, rng)
            if res is None:
                continue
            img, bbox = res
            x0, y0, x1, y1 = (int(round(v)) for v in bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img, "gate", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(preview_dir / f"{written:04d}.jpg"), img)
            written += 1

        print(f"  [preview] {written} previews in {preview_dir}")

    def write_yaml(self) -> None:
        yaml_path = self.output_dir / "data.yaml"
        content = (
            f"path: {self.output_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"names:\n"
            f"  0: gate\n"
        )
        yaml_path.write_text(content)
        print(f"Wrote {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate a YOLO gate-detection dataset from the Panda3D corridor sim",
    )
    ap.add_argument("--n-train", type=int, default=8000, help="number of train frames")
    ap.add_argument("--n-val", type=int, default=1000, help="number of val frames")
    ap.add_argument("--n-test", type=int, default=500, help="number of test frames")
    ap.add_argument("--output-dir", type=Path, default=Path("datasets/gates"),
                    help="dataset output directory")
    ap.add_argument("--seed", type=int, default=42, help="base random seed")
    ap.add_argument("--preview", type=int, default=0,
                    help="also generate N preview images with drawn bboxes")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder = GateDatasetBuilder(args.output_dir, args.seed)

    print(f"Generating train split ({args.n_train} frames)...")
    builder.generate_split("train", args.n_train, args.seed)

    print(f"Generating val split ({args.n_val} frames)...")
    builder.generate_split("val", args.n_val, args.seed + 10_000)

    print(f"Generating test split ({args.n_test} frames)...")
    builder.generate_split("test", args.n_test, args.seed + 20_000)

    builder.write_yaml()

    if args.preview > 0:
        print(f"Generating {args.preview} preview images...")
        builder.generate_preview(args.preview, args.seed + 30_000)

    print("Done.")


if __name__ == "__main__":
    main()

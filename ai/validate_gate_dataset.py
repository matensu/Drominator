"""
Validation du dataset YOLO de gates avant fine-tuning.

Exécute une batterie de checks (arborescence, format des labels, sanity check
Ultralytics, distribution statistique, preview visuel) et produit un rapport
console structuré. Exit 1 si un check critique échoue.

Usage:
    python ai/validate_gate_dataset.py
    python ai/validate_gate_dataset.py --dataset-dir datasets/gates --strict
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
CRITICAL_LABEL_RATIO = 0.01  # >1% mal formés => fail


# ----- Helpers d'affichage -----------------------------------------------

def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _line(marker: str, msg: str) -> None:
    print(f"  [{marker}] {msg}")


# ----- Structures de résultat --------------------------------------------

@dataclass
class CheckReport:
    critical_fail: bool = False
    warnings: int = 0

    def fail(self, msg: str) -> None:
        self.critical_fail = True
        _line("FAIL", msg)

    def warn(self, msg: str) -> None:
        self.warnings += 1
        _line("WARN", msg)

    def ok(self, msg: str) -> None:
        _line("PASS", msg)

    def info(self, msg: str) -> None:
        _line("INFO", msg)


# ----- Découverte des fichiers -------------------------------------------

def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _list_labels(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".txt")


# ----- 1) Volumes et arborescence ----------------------------------------

@dataclass
class SplitInventory:
    split: str
    images: list[Path]
    labels: list[Path]
    images_dir: Path
    labels_dir: Path
    matched: list[tuple[Path, Path]] = field(default_factory=list)
    missing_label: list[Path] = field(default_factory=list)
    missing_image: list[Path] = field(default_factory=list)
    empty_labels: int = 0


def check_inventory(dataset_dir: Path, report: CheckReport) -> dict[str, SplitInventory]:
    _section("1. Volumes et arborescence")
    inventories: dict[str, SplitInventory] = {}

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        if not images_dir.is_dir():
            report.fail(f"Dossier images manquant : {images_dir}")
            continue
        if not labels_dir.is_dir():
            report.fail(f"Dossier labels manquant : {labels_dir}")
            continue

        images = _list_images(images_dir)
        labels = _list_labels(labels_dir)
        inv = SplitInventory(
            split=split,
            images=images,
            labels=labels,
            images_dir=images_dir,
            labels_dir=labels_dir,
        )

        label_by_stem = {p.stem: p for p in labels}
        image_by_stem = {p.stem: p for p in images}

        for img in images:
            label = label_by_stem.get(img.stem)
            if label is None:
                inv.missing_label.append(img)
            else:
                inv.matched.append((img, label))

        for lab in labels:
            if lab.stem not in image_by_stem:
                inv.missing_image.append(lab)

        for _, lab in inv.matched:
            try:
                if lab.stat().st_size == 0:
                    inv.empty_labels += 1
                    continue
                with lab.open("r") as fh:
                    has_content = any(line.strip() for line in fh)
                if not has_content:
                    inv.empty_labels += 1
            except OSError as exc:
                report.fail(f"Impossible de lire {lab}: {exc}")

        inventories[split] = inv

    if not inventories:
        report.fail("Aucun split trouvé.")
        return inventories

    header = f"  {'split':<8} {'images':>8} {'labels':>8} {'matched':>8} {'no_label':>9} {'no_img':>7} {'empty':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for split, inv in inventories.items():
        print(
            f"  {split:<8} {len(inv.images):>8} {len(inv.labels):>8} "
            f"{len(inv.matched):>8} {len(inv.missing_label):>9} "
            f"{len(inv.missing_image):>7} {inv.empty_labels:>7}"
        )

    for inv in inventories.values():
        if inv.missing_label:
            report.fail(
                f"[{inv.split}] {len(inv.missing_label)} image(s) sans label "
                f"(ex: {inv.missing_label[0].name})"
            )
        if inv.missing_image:
            report.warn(
                f"[{inv.split}] {len(inv.missing_image)} label(s) orphelin(s) "
                f"(ex: {inv.missing_image[0].name})"
            )
        if not inv.images:
            report.fail(f"[{inv.split}] aucune image.")
        elif inv.matched and not inv.missing_label:
            report.ok(f"[{inv.split}] {len(inv.matched)} paires image/label cohérentes.")

    return inventories


# ----- 2) Format des labels ----------------------------------------------

@dataclass
class LabelIssue:
    file: Path
    line_no: int
    line: str
    reason: str


def _validate_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[LabelIssue]]:
    """Return (parsed_rows, issues)."""
    rows: list[tuple[int, float, float, float, float]] = []
    issues: list[LabelIssue] = []
    try:
        text = path.read_text()
    except OSError as exc:
        issues.append(LabelIssue(path, 0, "", f"unreadable: {exc}"))
        return rows, issues

    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(LabelIssue(path, i, raw, f"expected 5 fields, got {len(parts)}"))
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(parts[k]) for k in range(1, 5))
        except ValueError:
            issues.append(LabelIssue(path, i, raw, "non-numeric field(s)"))
            continue
        if cls != 0:
            issues.append(LabelIssue(path, i, raw, f"class_id={cls} (expected 0)"))
            continue
        for name, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= value <= 1.0):
                issues.append(LabelIssue(path, i, raw, f"{name}={value:.4f} out of [0,1]"))
                break
        else:
            rows.append((cls, cx, cy, w, h))
    return rows, issues


def check_label_format(
    inventories: dict[str, SplitInventory], report: CheckReport
) -> dict[str, list[list[tuple[int, float, float, float, float]]]]:
    _section("2. Format des labels")
    parsed_per_split: dict[str, list[list[tuple[int, float, float, float, float]]]] = {}
    all_issues: list[LabelIssue] = []
    total_files = 0
    bad_files = 0

    for split, inv in inventories.items():
        parsed_split: list[list[tuple[int, float, float, float, float]]] = []
        for _, lab in inv.matched:
            total_files += 1
            rows, issues = _validate_label_file(lab)
            parsed_split.append(rows)
            if issues:
                bad_files += 1
                all_issues.extend(issues)
        parsed_per_split[split] = parsed_split

    if total_files == 0:
        report.fail("Aucun label à valider.")
        return parsed_per_split

    bad_ratio = bad_files / total_files
    print(f"  Total fichiers labels analysés : {total_files}")
    print(f"  Fichiers avec au moins un problème : {bad_files} ({bad_ratio*100:.2f}%)")
    print(f"  Total lignes problématiques : {len(all_issues)}")

    if all_issues:
        print("\n  Premiers problèmes (max 20) :")
        for issue in all_issues[:20]:
            line_preview = issue.line if len(issue.line) <= 60 else issue.line[:57] + "..."
            print(f"    {issue.file.name}:{issue.line_no} — {issue.reason} | {line_preview!r}")

    if bad_ratio > CRITICAL_LABEL_RATIO:
        report.fail(
            f"{bad_ratio*100:.2f}% de fichiers mal formés > seuil critique "
            f"{CRITICAL_LABEL_RATIO*100:.1f}%"
        )
    elif bad_files > 0:
        report.warn(f"{bad_files} fichier(s) avec problème(s) de format mais sous le seuil critique.")
    else:
        report.ok("Tous les labels sont au bon format YOLO.")

    return parsed_per_split


# ----- 3) Sanity check Ultralytics ---------------------------------------

def check_ultralytics(dataset_dir: Path, report: CheckReport) -> None:
    _section("3. Sanity check Ultralytics (check_det_dataset)")
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        report.fail(f"data.yaml introuvable: {yaml_path}")
        return
    try:
        from ultralytics.data.utils import check_det_dataset
    except ImportError as exc:
        report.fail(f"Ultralytics non installé: {exc}")
        return
    try:
        info = check_det_dataset(str(yaml_path))
    except Exception as exc:  # noqa: BLE001
        report.fail(f"check_det_dataset a échoué: {exc}")
        return

    print("  Résumé renvoyé par Ultralytics :")
    for key in ("path", "train", "val", "test", "names", "nc"):
        if key in info:
            print(f"    {key}: {info[key]}")
    report.ok("check_det_dataset OK.")


# ----- 4) Distribution statistique sur le train --------------------------

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


def check_distribution(
    parsed_per_split: dict[str, list[list[tuple[int, float, float, float, float]]]],
    report: CheckReport,
) -> None:
    _section("4. Distribution statistique (train)")
    train_rows = parsed_per_split.get("train")
    if not train_rows:
        report.warn("Pas de split train à analyser.")
        return

    n_total = len(train_rows)
    n_zero = sum(1 for r in train_rows if not r)
    n_one = sum(1 for r in train_rows if len(r) == 1)
    n_multi = sum(1 for r in train_rows if len(r) >= 2)

    print(f"  Images train : {n_total}")
    print(f"    sans gate    : {n_zero} ({n_zero/n_total*100:.1f}%)")
    print(f"    1 gate       : {n_one} ({n_one/n_total*100:.1f}%)")
    print(f"    2+ gates     : {n_multi} ({n_multi/n_total*100:.1f}%)")

    areas = [w * h for rows in train_rows for (_, _, _, w, h) in rows]
    xs = [cx for rows in train_rows for (_, cx, _, _, _) in rows]
    ys = [cy for rows in train_rows for (_, _, cy, _, _) in rows]

    if not areas:
        report.warn("Aucune bbox dans le train — impossible de calculer la distribution.")
        return

    def _stats(vals: list[float], label: str) -> None:
        print(
            f"  {label:<10} min={min(vals):.4f} p10={_percentile(vals,10):.4f} "
            f"med={statistics.median(vals):.4f} p90={_percentile(vals,90):.4f} "
            f"max={max(vals):.4f}"
        )

    _stats(areas, "area")
    _stats(xs, "cx")
    _stats(ys, "cy")
    print(f"  std cx : {statistics.pstdev(xs):.4f}    std cy : {statistics.pstdev(ys):.4f}")
    print(f"  std area : {statistics.pstdev(areas):.6f}")

    zero_ratio = n_zero / n_total
    if zero_ratio == 0:
        report.warn("0% d'images sans gate — pas de négatifs dans le train.")
    elif zero_ratio > 0.5:
        report.warn(f"{zero_ratio*100:.1f}% d'images sans gate — beaucoup de négatifs.")

    if statistics.pstdev(areas) < 1e-4:
        report.warn("Variance des tailles de bbox quasi nulle — toutes les bbox ont la même taille.")

    med_x = statistics.median(xs)
    if med_x < 0.3 or med_x > 0.7:
        report.warn(f"Distribution X déséquilibrée (médiane cx = {med_x:.3f}).")
    else:
        report.ok(f"Distribution cx centrée (médiane = {med_x:.3f}).")


# ----- 5) Validation visuelle --------------------------------------------

def _draw_yolo_boxes(img: np.ndarray, rows: Iterable[tuple[int, float, float, float, float]]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for _, cx, cy, bw, bh in rows:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def _build_montage(images: list[np.ndarray], tile_width: int = 320) -> np.ndarray:
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    tiles: list[np.ndarray] = []
    tile_h = 0
    for img in images:
        h, w = img.shape[:2]
        new_h = max(1, int(round(h * tile_width / w)))
        tiles.append(cv2.resize(img, (tile_width, new_h)))
        tile_h = max(tile_h, new_h)

    padded: list[np.ndarray] = []
    for tile in tiles:
        h = tile.shape[0]
        if h < tile_h:
            pad = np.zeros((tile_h - h, tile_width, 3), dtype=tile.dtype)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    while len(padded) < rows * cols:
        padded.append(np.zeros((tile_h, tile_width, 3), dtype=padded[0].dtype))

    grid_rows = [np.hstack(padded[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(grid_rows)


def make_preview(
    dataset_dir: Path,
    inventories: dict[str, SplitInventory],
    n_preview: int,
    seed: int,
    report: CheckReport,
) -> Path | None:
    _section("5. Validation visuelle (preview)")
    train = inventories.get("train")
    if not train or not train.matched:
        report.warn("Aucun couple image/label dans train — pas de preview généré.")
        return None

    preview_dir = dataset_dir / "_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sample_size = min(n_preview, len(train.matched))
    sample = rng.sample(train.matched, sample_size)

    written = 0
    annotated_imgs: list[np.ndarray] = []
    for img_path, lab_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            report.warn(f"Impossible de lire {img_path}")
            continue
        rows, _ = _validate_label_file(lab_path)
        annotated = _draw_yolo_boxes(img, rows)
        out_path = preview_dir / f"{img_path.stem}_annotated.jpg"
        if not cv2.imwrite(str(out_path), annotated):
            report.warn(f"Échec écriture preview {out_path}")
            continue
        written += 1
        annotated_imgs.append(annotated)

    if written == 0:
        report.fail("Aucun preview écrit.")
        return None

    report.ok(f"{written} previews écrits dans {preview_dir}")

    montage_path = preview_dir / "montage.jpg"
    try:
        montage = _build_montage(annotated_imgs)
        if cv2.imwrite(str(montage_path), montage):
            report.ok(f"Montage grille généré : {montage_path.name}")
        else:
            report.warn(f"Échec écriture montage {montage_path}")
            montage_path = None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        report.warn(f"Échec construction montage : {exc}")
        montage_path = None  # type: ignore[assignment]

    if montage_path is not None:
        print(f"\n  Ouvre directement le montage : {montage_path.resolve()}")
    print(f"  Dossier preview : {preview_dir.resolve()}")
    return preview_dir


# ----- main ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/gates"),
        help="Racine du dataset (contient data.yaml, images/, labels/).",
    )
    ap.add_argument("--n-preview", type=int, default=12, help="Nombre d'images preview.")
    ap.add_argument("--seed", type=int, default=42, help="Seed du sampling preview.")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 même si seuls des warnings sont remontés.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir.resolve()
    report = CheckReport()

    print(f"Validation du dataset : {dataset_dir}")
    if not dataset_dir.is_dir():
        _line("FAIL", f"Dataset directory introuvable: {dataset_dir}")
        return 1

    inventories = check_inventory(dataset_dir, report)
    parsed = check_label_format(inventories, report) if inventories else {}
    check_ultralytics(dataset_dir, report)
    if parsed:
        check_distribution(parsed, report)
    make_preview(dataset_dir, inventories, args.n_preview, args.seed, report)

    _section("Résumé")
    print(f"  Warnings : {report.warnings}")
    print(f"  Critical fail : {report.critical_fail}")

    if report.critical_fail:
        _line("FAIL", "Au moins un check critique a échoué.")
        return 1
    if args.strict and report.warnings > 0:
        _line("FAIL", "Mode --strict : warnings considérés comme bloquants.")
        return 1
    _line("PASS", "Dataset OK pour le fine-tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

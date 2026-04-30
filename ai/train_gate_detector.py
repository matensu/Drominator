"""
Fine-tune YOLOv11n on the simulated gate dataset produced by
`ai/generate_gate_dataset.py`.

Two-phase fine-tuning:
  * Phase 1 (warmup): freeze the first N layers for a few epochs so the
    randomly-initialized detection head adapts to the gate class without
    corrupting the COCO backbone.
  * Phase 2 (full): unfreeze everything and train at the same low LR.

Artifacts (all under ``--project`` / ``--run-name``):
  * ``ai/models/<run-name>_warmup/`` – Ultralytics run dir, phase 1
  * ``ai/models/<run-name>/``        – Ultralytics run dir, phase 2
  * ``ai/models/yolo_gate.pt``       – final best weights (copied)
  * ``ai/models/gate_detector_training.png`` – combined training curves
  * ``ai/models/gate_detector_report.txt``   – test-set metrics

All hyperparameters are CLI-overridable; run with ``--help`` for details.
The script never launches training under ``--dry-run``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/gates/data.yaml"),
        help="Path to the YOLO data.yaml file",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pre-trained checkpoint to start from (COCO weights by default)",
    )
    ap.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size")
    ap.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 = auto, requires CUDA; falls back to 16 on CPU)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience (epochs without val improvement)",
    )
    ap.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    ap.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of leading layers to freeze during warmup phase",
    )
    ap.add_argument(
        "--freeze-epochs",
        type=int,
        default=5,
        help="Epochs to run with frozen layers before unfreezing",
    )
    ap.add_argument("--mosaic", type=float, default=0.5)
    ap.add_argument("--mixup", type=float, default=0.1)
    ap.add_argument("--hsv-v", type=float, default=0.4)
    ap.add_argument(
        "--project",
        type=Path,
        default=Path("ai/models"),
        help="Ultralytics 'project' directory (run dirs are created inside)",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default="gate_detector",
        help="Base name for the run dirs (warmup suffix is added automatically)",
    )
    ap.add_argument(
        "--output-weights",
        type=Path,
        default=Path("ai/models/yolo_gate.pt"),
        help="Where to copy the final best.pt",
    )
    ap.add_argument("--device", type=str, default="", help="CUDA device id, 'cpu', or empty for auto")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers (lower on small VMs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-train-max",
        type=int,
        default=None,
        help="If set, cap training images (applied as Ultralytics 'fraction')",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args, validate dataset, print plan, and exit without training",
    )
    return ap.parse_args()


def _count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for p in dir_path.iterdir() if p.suffix.lower() in exts)


def _resolve_batch(batch: int, device: str) -> int:
    if batch != -1:
        return batch
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if device.lower() == "cpu" or not has_cuda:
        print("[TRAIN] batch=-1 (auto) requires CUDA; falling back to batch=16")
        return 16
    return -1


def _compute_fraction(data_yaml: Path, n_train_max: int | None) -> float | None:
    if n_train_max is None:
        return None
    try:
        import yaml  # PyYAML ships with ultralytics
    except ImportError:
        print("[TRAIN] PyYAML not available — cannot resolve --n-train-max, sending fraction=1.0")
        return 1.0
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train_rel = cfg.get("train", "images/train")
    total = _count_images(root / train_rel)
    if total == 0:
        print(f"[TRAIN] Could not count training images at {root / train_rel}; using fraction=1.0")
        return 1.0
    frac = max(1e-6, min(1.0, n_train_max / total))
    print(f"[TRAIN] --n-train-max={n_train_max} / {total} training images → fraction={frac:.4f}")
    return frac


def _effective_freeze_epochs(args: argparse.Namespace) -> int:
    """Clamp warmup epochs to leave at least 1 epoch for the main phase."""
    if args.freeze <= 0 or args.freeze_epochs <= 0:
        return 0
    return max(0, min(args.freeze_epochs, args.epochs - 1))


def _print_plan(args: argparse.Namespace, batch: int, fraction: float | None) -> None:
    eff_freeze = _effective_freeze_epochs(args)
    if eff_freeze != args.freeze_epochs:
        print(
            f"[TRAIN] warmup clamped: --freeze-epochs={args.freeze_epochs} but "
            f"--epochs={args.epochs} → using {eff_freeze} warmup epoch(s)"
        )
    print("─" * 60)
    print("Planned training config")
    print("─" * 60)
    print(f"  data         : {args.data}")
    print(f"  model        : {args.model}")
    print(f"  epochs       : {args.epochs}  (warmup {eff_freeze} + main {args.epochs - eff_freeze})")
    print(f"  imgsz        : {args.imgsz}")
    print(f"  batch        : {batch}")
    print(f"  patience     : {args.patience}")
    print(f"  lr0          : {args.lr0}")
    print(f"  freeze       : {args.freeze} (warmup only)")
    print(f"  mosaic/mixup : {args.mosaic} / {args.mixup}")
    print(f"  hsv_v        : {args.hsv_v}")
    print(f"  device       : {args.device or 'auto'}")
    print(f"  workers      : {args.workers}")
    print(f"  seed         : {args.seed}")
    print(f"  fraction     : {fraction if fraction is not None else '1.0 (no cap)'}")
    print(f"  project      : {args.project}")
    print(f"  run name     : {args.run_name}  (+ {args.run_name}_warmup)")
    print(f"  output weights: {args.output_weights}")
    print("─" * 60)


def _validate_data_yaml(data_yaml: Path) -> bool:
    """Print the data.yaml content and basic split-dir checks. Returns True if usable."""
    if not data_yaml.exists():
        print(f"[TRAIN] WARNING: {data_yaml} not found — run ai/generate_gate_dataset.py first")
        return False
    print(f"[TRAIN] data.yaml contents ({data_yaml}):")
    for line in data_yaml.read_text().splitlines():
        print(f"    {line}")
    try:
        import yaml

        cfg = yaml.safe_load(data_yaml.read_text())
    except Exception as e:
        print(f"[TRAIN] Could not parse {data_yaml}: {e}")
        return False
    root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel is None:
            continue
        n = _count_images(root / rel)
        print(f"[TRAIN] {split:>5}: {n} images in {root / rel}")
    return True


def _train_phase(
    model_path: str | Path,
    args: argparse.Namespace,
    *,
    name: str,
    epochs: int,
    freeze: int,
    batch: int,
    fraction: float | None,
) -> Path:
    """Run one Ultralytics training phase and return the resulting run dir."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    train_kwargs = dict(
        data=str(args.data),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        patience=args.patience,
        lr0=args.lr0,
        freeze=freeze,
        mosaic=args.mosaic,
        mixup=args.mixup,
        hsv_v=args.hsv_v,
        project=str(args.project),
        name=name,
        seed=args.seed,
        device=args.device if args.device else None,
        workers=args.workers,
        exist_ok=True,
        verbose=True,
    )
    if fraction is not None:
        train_kwargs["fraction"] = fraction
    results = model.train(**train_kwargs)
    run_dir = Path(getattr(results, "save_dir", args.project / name))
    return run_dir


def _plot_training_curves(warmup_dir: Path, main_dir: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frames = []
    offset = 0
    for phase, d in (("warmup", warmup_dir), ("main", main_dir)):
        csv = d / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        df["phase"] = phase
        df["global_epoch"] = df["epoch"] + offset
        frames.append(df)
        offset += int(df["epoch"].max()) + 1 if len(df) else 0

    if not frames:
        print(f"[TRAIN] No results.csv found; skipping plot")
        return

    full = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = full["global_epoch"]

    def _plot(ax, cols, title):
        for c in cols:
            if c in full.columns:
                ax.plot(x, full[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    _plot(axes[0, 0], ["train/box_loss", "train/cls_loss", "train/dfl_loss"], "Train losses")
    _plot(axes[0, 1], ["val/box_loss", "val/cls_loss", "val/dfl_loss"], "Val losses")
    _plot(axes[1, 0], ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], "Val mAP")
    _plot(axes[1, 1], ["metrics/precision(B)", "metrics/recall(B)"], "Val precision / recall")

    freeze_split = full[full["phase"] == "warmup"]["global_epoch"].max()
    if pd.notna(freeze_split):
        for ax in axes.flat:
            ax.axvline(freeze_split + 0.5, color="red", linestyle="--", alpha=0.5, label="unfreeze")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Gate detector fine-tuning")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[TRAIN] Training curves → {out_png}")


def _evaluate_test(weights: Path, data_yaml: Path, imgsz: int, device: str, report_path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        device=device if device else None,
        verbose=True,
    )
    box = metrics.box
    lines = [
        "Gate detector — test-set evaluation",
        f"weights : {weights}",
        f"data    : {data_yaml}",
        "",
        f"mAP50       : {box.map50:.4f}",
        f"mAP50-95    : {box.map:.4f}",
        f"precision   : {float(box.mp):.4f}",
        f"recall      : {float(box.mr):.4f}",
    ]
    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"[TRAIN] Report → {report_path}")


def main() -> int:
    args = parse_args()

    data_ok = _validate_data_yaml(args.data)
    batch = _resolve_batch(args.batch, args.device)
    fraction = _compute_fraction(args.data, args.n_train_max) if data_ok else None
    _print_plan(args, batch, fraction)

    if args.dry_run:
        print("[TRAIN] --dry-run: exiting before training.")
        return 0
    if not data_ok:
        print(f"[TRAIN] Cannot train: {args.data} is missing or unreadable.", file=sys.stderr)
        return 2

    args.project.mkdir(parents=True, exist_ok=True)
    warmup_name = f"{args.run_name}_warmup"
    eff_freeze = _effective_freeze_epochs(args)

    if eff_freeze > 0:
        print(f"[TRAIN] Phase 1 (warmup): freeze={args.freeze}, epochs={eff_freeze}")
        warmup_dir = _train_phase(
            args.model, args,
            name=warmup_name,
            epochs=eff_freeze,
            freeze=args.freeze,
            batch=batch,
            fraction=fraction,
        )
        phase2_start = warmup_dir / "weights" / "best.pt"
        if not phase2_start.exists():
            phase2_start = warmup_dir / "weights" / "last.pt"
    else:
        warmup_dir = None
        phase2_start = Path(args.model)

    main_epochs = args.epochs - eff_freeze
    print(f"[TRAIN] Phase 2 (full): freeze=0, epochs={main_epochs}")
    main_dir = _train_phase(
        phase2_start, args,
        name=args.run_name,
        epochs=main_epochs,
        freeze=0,
        batch=batch,
        fraction=fraction,
    )

    best = main_dir / "weights" / "best.pt"
    if not best.exists():
        best = main_dir / "weights" / "last.pt"
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output_weights)
    print(f"[TRAIN] Final weights → {args.output_weights}")

    plot_path = args.project / "gate_detector_training.png"
    _plot_training_curves(warmup_dir or main_dir, main_dir, plot_path)

    report_path = args.project / "gate_detector_report.txt"
    _evaluate_test(args.output_weights, args.data, args.imgsz, args.device, report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

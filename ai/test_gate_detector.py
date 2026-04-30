"""
Quick visual validation of the fine-tuned gate detector.

Sources:
  * path to an image file (jpg/png/...)
  * path to a video file (mp4/avi/...)
  * an integer webcam index (e.g. ``--source 0``)

Examples:
    python ai/test_gate_detector.py --source datasets/gates/preview/0000.jpg
    python ai/test_gate_detector.py --source 0                 # webcam
    python ai/test_gate_detector.py --source gate.mp4 --save   # saves annotated video
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("ai/models/yolo_gate.pt"),
        help="Path to the fine-tuned YOLO checkpoint",
    )
    ap.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, video path, or webcam index (e.g. '0')",
    )
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", type=str, default="", help="CUDA device id, 'cpu', or empty for auto")
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output next to the source (or in runs/ for webcam)",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live window (useful on headless servers)",
    )
    return ap.parse_args()


def _is_webcam(source: str) -> bool:
    return source.isdigit()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _run_image(model, path: Path, args: argparse.Namespace) -> int:
    import cv2

    results = model.predict(
        source=str(path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        verbose=False,
    )
    if not results:
        print("[TEST] No result returned")
        return 1
    annotated = results[0].plot()
    n_det = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"[TEST] {path}: {n_det} detection(s)")

    if args.save:
        out = path.with_name(f"{path.stem}_detected{path.suffix}")
        cv2.imwrite(str(out), annotated)
        print(f"[TEST] Saved → {out}")

    if not args.no_show:
        cv2.imshow("gate detector", annotated)
        print("[TEST] Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _run_stream(model, source, args: argparse.Namespace, *, title: str, save_path: Path | None) -> int:
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[TEST] Failed to open source: {source}", file=sys.stderr)
        return 2

    writer = None
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
        print(f"[TEST] Writing annotated video → {save_path}")

    last = time.time()
    n_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                verbose=False,
            )
            annotated = results[0].plot() if results else frame

            n_frames += 1
            now = time.time()
            if now - last >= 1.0:
                fps = n_frames / (now - last)
                n_det = len(results[0].boxes) if results and results[0].boxes is not None else 0
                print(f"[TEST] {fps:5.1f} fps — {n_det} detection(s)")
                n_frames = 0
                last = now

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow(title, annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or ESC
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"[TEST] Weights not found: {args.weights}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    if _is_webcam(args.source):
        save_path = Path("runs/gate_webcam.mp4") if args.save else None
        return _run_stream(model, int(args.source), args, title="webcam", save_path=save_path)

    path = Path(args.source)
    if not path.exists():
        print(f"[TEST] Source not found: {path}", file=sys.stderr)
        return 2

    if _is_image(path):
        return _run_image(model, path, args)

    save_path = path.with_name(f"{path.stem}_detected.mp4") if args.save else None
    return _run_stream(model, str(path), args, title=path.name, save_path=save_path)


if __name__ == "__main__":
    raise SystemExit(main())

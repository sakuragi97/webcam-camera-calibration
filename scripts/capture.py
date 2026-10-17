"""Capture calibration frames from the webcam.

Run from the project root:
    python scripts/capture.py --out data/webcam/calib_1920x1080 --width 1920 --height 1080

Keys: SPACE = save frame (only if the board is detected, unless --force), q / ESC = quit.
The preview shows the resolution, fps, whether the board is detected, and a 3x3 grid
that turns green where saved frames already had corners (coverage).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `calib` importable
from calib.camera_io import open_camera                         # noqa: E402

BACKENDS = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


def parse_args():
    ap = argparse.ArgumentParser(description="Capture checkerboard frames for calibration")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", choices=BACKENDS, default="any")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", required=True, help="output folder, e.g. data/webcam/calib_1920x1080")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--force", action="store_true", help="save even when no board is detected")
    return ap.parse_args()


def detect_fast(gray, pattern, max_side=640):
    """Quick yes/no detection on a small copy. Returns corners in FULL-resolution pixels or None."""
    s = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, pattern, flags=flags)
    return corners.reshape(-1, 2) / s if found else None


def draw_overlay(view, coverage, corners, pattern, fps, saved):
    h, w = view.shape[:2]
    # coverage grid: green = a saved frame had corners in this cell
    for i in range(3):
        for j in range(3):
            x0, y0, x1, y1 = j * w // 3, i * h // 3, (j + 1) * w // 3, (i + 1) * h // 3
            color = (0, 180, 0) if coverage[i, j] else (0, 0, 200)
            cv2.rectangle(view, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), color, 2)
            cv2.putText(view, str(coverage[i, j]), (x0 + 12, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if corners is not None:
        cv2.drawChessboardCorners(view, pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
    status = f"{w}x{h}  {fps:4.1f} fps  board: {'YES' if corners is not None else 'no'}  saved: {saved}"
    cv2.putText(view, status, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def update_coverage(coverage, corners, w, h):
    cx = np.clip((corners[:, 0] / w * 3).astype(int), 0, 2)
    cy = np.clip((corners[:, 1] / h * 3).astype(int), 0, 2)
    for i, j in set(zip(cy, cx)):
        coverage[i, j] += 1


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pattern = (args.cols, args.rows)

    cap = open_camera(args.camera, args.width, args.height, BACKENDS[args.backend])
    coverage = np.zeros((3, 3), int)
    saved = 0
    fps, t_prev = 0.0, time.perf_counter()
    window = "capture  |  SPACE = save, q = quit"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame could not be read, stopping.")
                break
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)      # smoothed fps
            t_prev = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_fast(gray, pattern)

            view = frame.copy()                                    # never draw on the frame we save
            draw_overlay(view, coverage, corners, pattern, fps, saved)
            scale = min(1.0, 1280 / view.shape[1])                 # keep the window on screen
            cv2.imshow(window, cv2.resize(view, None, fx=scale, fy=scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if corners is None and not args.force:
                    print("No board detected: frame not saved (use --force to save anyway).")
                    continue
                h, w = frame.shape[:2]
                name = out / f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{saved:03d}.png"
                cv2.imwrite(str(name), frame)                      # full resolution, PNG = lossless
                saved += 1
                if corners is not None:
                    update_coverage(coverage, corners, w, h)
                print(f"saved {name.name}  ({w}x{h})")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"{saved} frame(s) saved to {out}")


if __name__ == "__main__":
    main()

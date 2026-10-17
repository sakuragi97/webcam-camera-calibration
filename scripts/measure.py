"""Measure distances in mm on the board plane.

Run from the project root:
    python scripts/measure.py data/webcam/apps_1920x1080/frame_xxx.png --calib calibration.yaml --square 8.24

Click points: every 2 clicks = one distance, printed in mm. r = reset, q / ESC = quit.
For the card: click the middle of the left and right edge (width), then top and bottom (height).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `calib` importable
from calib.board import object_points                          # noqa: E402
from calib.detect import find_corners                          # noqa: E402
from calib.geometry import image_to_board_plane                # noqa: E402
from calib.io_yaml import load_calibration                     # noqa: E402

PATTERN = (9, 6)
DISPLAY_SCALE = 0.6        # window is shown smaller; clicks are scaled back to full resolution


def main():
    ap = argparse.ArgumentParser(description="Click two points, get their distance in mm")
    ap.add_argument("image")
    ap.add_argument("--calib", default="calibration.yaml")
    ap.add_argument("--square", type=float, default=8.24, help="square size in mm")
    args = ap.parse_args()

    # 1) load image and calibration (the resolution must match)
    K, dist, size = load_calibration(args.calib)
    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert (img.shape[1], img.shape[0]) == size, f"image is {img.shape[1]}x{img.shape[0]}, calibration {size}"

    # 2) find the board and its pose (board -> camera) with K fixed
    corners = find_corners(gray, PATTERN)
    if corners is None:
        sys.exit("board not found in this image")
    ok, rvec, tvec = cv2.solvePnP(object_points(*PATTERN, args.square), corners, K, dist)

    # 3) mouse clicks -> board plane (mm); every pair of clicks is one measurement
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x / DISPLAY_SCALE, y / DISPLAY_SCALE))       # back to full-resolution pixels
        if len(clicks) % 2 == 0:
            XY = image_to_board_plane(np.array(clicks[-2:]), K, dist, rvec, tvec)
            print(f"distance {len(clicks) // 2}: {np.linalg.norm(XY[1] - XY[0]):.2f} mm")

    cv2.namedWindow("measure")
    cv2.setMouseCallback("measure", on_mouse)

    # 4) draw the board and the clicked points/lines until the user quits
    while True:
        view = img.copy()
        cv2.drawChessboardCorners(view, PATTERN, corners, True)
        for i, p in enumerate(clicks):
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            if i % 2 == 1:
                q = clicks[i - 1]
                cv2.line(view, (int(q[0]), int(q[1])), (int(p[0]), int(p[1])), (0, 0, 255), 2)
        cv2.imshow("measure", cv2.resize(view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            clicks.clear()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

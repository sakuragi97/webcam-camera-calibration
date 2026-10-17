"""Live undistortion.

Run from the project root:
    python scripts/undistort_live.py --calib calibration.yaml

Shows the original (left) and the undistorted image (right) with the fps. q / ESC = quit.
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
from calib.io_yaml import load_calibration                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Show original and undistorted webcam image")
    ap.add_argument("--calib", default="calibration.yaml")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    # 1) load K and dist; open the camera at the SAME resolution as the calibration
    K, dist, (w, h) = load_calibration(args.calib)
    cap = open_camera(args.camera, w, h)

    # 2) compute the undistortion maps ONCE: for every output pixel, where to read in the input
    #    alpha = 0 -> crop to valid pixels only (no black borders)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)

    t_last, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 3) per frame: just a lookup with the precomputed maps (fast)
        undist = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # 4) fps as a smoothed average
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now

        # 5) side by side, half size so it fits on the screen
        view = np.hstack([frame, undist])
        view = cv2.resize(view, None, fx=0.5, fy=0.5)
        cv2.putText(view, f"original | undistorted   {fps:.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("undistort", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

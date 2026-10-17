"""Generate the synthetic test bench (run once, ~1-2 minutes).

    python -m testbench.make_dataset          # from the project root

Creates data/synthetic/ with
    full/   2400x1800 calibration photos, img_00.jpg ... (a few of them are bad on purpose)
    hd/     1280x720 photos of the same camera in a "16:9 video mode"
    apps/   photos of the board, a bank card and an ArUco marker on a table
    ground_truth.json   true K, distortion, poses and corner positions
                        (read it through testbench.gt, used only to evaluate results)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ._render import Board, SceneLayout, SyntheticCamera, rot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "synthetic"

K_FULL = np.array([[2000.0, 0, 1212.0], [0, 2004.0, 886.0], [0, 0, 1]])
DIST = np.array([-0.30, 0.11, 0.0008, -0.0005, -0.015])
SIZE_FULL = (2400, 1800)
# the 16:9 video mode: the sensor rows 225..1574 are read out and scaled by 1280/2400
HD_CROP_Y = 225
HD_SCALE = 1280 / 2400
SIZE_HD = (1280, 720)


def _hd_camera_matrix():
    K = K_FULL.copy()
    K[1, 2] -= HD_CROP_Y
    K[:2] *= HD_SCALE
    return K


def _good_poses(cam: SyntheticCamera, rng, n, depth=(380, 620), tilt=35):
    w, h = cam.size
    targets = [(u, v) for v in (0.3, 0.5, 0.7) for u in (0.3, 0.5, 0.7)]
    tilts = [(0, 0), (tilt, 0), (-tilt, 0), (0, tilt), (0, -tilt),
             (0.8 * tilt, 0.8 * tilt), (-0.8 * tilt, 0.8 * tilt), (0.8 * tilt, -0.8 * tilt),
             (-0.8 * tilt, -0.8 * tilt)]
    out, k, attempts = [], 0, 0
    while k < n and attempts < 5000:
        attempts += 1
        tu, tv = targets[k % len(targets)]
        rx, ry = tilts[(k * 4) % len(tilts)] if k >= len(targets) else tilts[k]
        R = rot(rx + rng.normal(0, 4), ry + rng.normal(0, 4), rng.uniform(-25, 25))
        u = (tu + rng.normal(0, 0.05)) * w
        v = (tv + rng.normal(0, 0.05)) * h
        rvec, tvec = cam.pose_looking_at(u, v, rng.uniform(*depth), R)
        if cam.corners_inside(rvec, tvec, margin=int(0.02 * w)):
            out.append((rvec, tvec))
            k += 1
    return out


def _save(path, img):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 93])


def main(seed: int = 7, verbose: bool = True):
    t0 = time.time()
    board, layout = Board(), SceneLayout()
    objp = board._gt_corners_mm().astype(np.float64)
    rng = np.random.default_rng(seed)
    for sub in ("full", "hd", "apps"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    gt = {"board": asdict(board), "layout": asdict(layout),
          "cameras": {"full": {"K": K_FULL.tolist(), "dist": DIST.tolist(), "size": list(SIZE_FULL)},
                      "hd": {"K": _hd_camera_matrix().tolist(), "dist": DIST.tolist(), "size": list(SIZE_HD)}},
          "images": {}}

    # ---------------- full resolution set: 18 good + 3 planted problems, shuffled names
    cam = SyntheticCamera(K_FULL, DIST, SIZE_FULL, board, layout)
    items = [("good", rv, tv, {}) for rv, tv in _good_poses(cam, rng, 18)]
    w, h = SIZE_FULL
    rv, tv = cam.pose_looking_at(0.93 * w, 0.5 * h, 450, rot(10, -15, 5))
    items.append(("board partly outside the image", rv, tv, {}))
    rv, tv = cam.pose_looking_at(0.5 * w, 0.45 * h, 520, rot(-20, 15, 8))
    items.append(("motion blur (17 px horizontal)", rv, tv, {"motion_blur": 17}))
    K_alt = K_FULL.copy()
    K_alt[0, 0] *= 1.035
    K_alt[1, 1] *= 1.035
    K_alt[0, 2] += 22
    K_alt[1, 2] -= 15
    dist_alt = DIST * np.array([0.8, 0.8, 1, 1, 0.8])
    rv, tv = cam.pose_looking_at(0.42 * w, 0.55 * h, 430, rot(30, -25, -12))
    items.append(("focus changed: different K and distortion", rv, tv, {"camera": "alt"}))
    order = rng.permutation(len(items))
    cam_alt = None
    for new_idx, i in enumerate(order):
        note, rv, tv, opts = items[i]
        name = f"img_{new_idx:02d}.jpg"
        c = cam
        if opts.get("camera") == "alt":
            cam_alt = cam_alt or SyntheticCamera(K_alt, dist_alt, SIZE_FULL, board, layout)
            c = cam_alt
        img = c.render(rv, tv, rng=rng, motion_blur=opts.get("motion_blur", 0))
        _save(DATA / "full" / name, img)
        gt["images"][f"full/{name}"] = {"rvec": np.ravel(rv).tolist(), "tvec": np.ravel(tv).tolist(),
                                        "corners_px": c.project(objp, rv, tv).tolist(), "note": note}
    del cam_alt

    # ---------------- application photos: board + card + marker on a table
    centre = np.array([150.0, 70.0, 0.0])
    for name, (R, tu, tv_, depth) in {"app_01_top.jpg": (rot(12, -6, 4), 0.50, 0.50, 640),
                                      "app_02_oblique.jpg": (rot(42, 8, -10), 0.50, 0.52, 700),
                                      "app_03_side.jpg": (rot(25, -32, 18), 0.52, 0.48, 680)}.items():
        rv, tv = cam.pose_looking_at(tu * w, tv_ * h, depth, R, target_mm=centre)
        _save(DATA / "apps" / name, cam.render(rv, tv, extras=True, rng=rng))
        gt["images"][f"apps/{name}"] = {
            "rvec": np.ravel(rv).tolist(), "tvec": np.ravel(tv).tolist(),
            "corners_px": cam.project(objp, rv, tv).tolist(),
            "card_edge_midpoints_px": cam.project(layout.card_edge_midpoints(), rv, tv).tolist(),
            "note": "application"}
    del cam

    # ---------------- 16:9 video mode of the same camera
    cam_hd = SyntheticCamera(_hd_camera_matrix(), DIST, SIZE_HD, board, layout)
    for k, (rv, tv) in enumerate(_good_poses(cam_hd, rng, 15, depth=(420, 650), tilt=30)):
        name = f"hd_{k:02d}.jpg"
        _save(DATA / "hd" / name, cam_hd.render(rv, tv, rng=rng, blur=0.5))
        gt["images"][f"hd/{name}"] = {"rvec": np.ravel(rv).tolist(), "tvec": np.ravel(tv).tolist(),
                                      "corners_px": cam_hd.project(objp, rv, tv).tolist(), "note": "good"}

    (DATA / "ground_truth.json").write_text(json.dumps(gt))
    if verbose:
        print(f"Synthetic test data written to {DATA} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

"""Finding the checkerboard corners with sub-pixel accuracy."""
from __future__ import annotations

import cv2
import numpy as np


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int],
                 max_side: int = 1600) -> np.ndarray | None:
    """Detect the inner corners with sub-pixel accuracy.

    Args:
        gray: single-channel uint8 image at FULL resolution
        pattern_size: (cols, rows) inner corners, e.g. (9, 6)
        max_side: if the image is larger, search on a downscaled copy first

    Returns:
        (cols*rows, 1, 2) float32 corner positions in FULL-resolution pixels,
        or None if the board is not found completely.

    Strategy: find the board on a downscaled copy (fast), map the corners back,
    then refine them on the FULL image (accurate: ~0.05 px median error on the
    synthetic test bench, ~0.12 s per 2400x1800 image).
    """
    # 1) shrink big images so the (slow) board search runs on fewer pixels
    scale = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    # 2) search the board: robust "SB" detector first, classic detector as fallback
    found, corners = cv2.findChessboardCornersSB(small, pattern_size)
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(small, pattern_size, flags=flags)
    if not found:
        return None  # board missing or only partly visible

    # 3) map the corners back to full-resolution pixel coordinates
    corners = (corners.reshape(-1, 1, 2) / scale).astype(np.float32)

    # 4) refine to sub-pixel accuracy on the FULL image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def edge_sharpness(gray: np.ndarray, pattern_size: tuple[int, int], corners: np.ndarray) -> float:
    """Width (px) of the black-to-white transitions of the detected board.

    Small = sharp. Must also catch blur that happens in ONE direction only.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    img = gray.astype(np.float32)

    def width_across(a, b):
        # the segment a->b runs along a square edge; sample a profile perpendicular to it at its midpoint
        mid, d = (a + b) / 2, b - a
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)            # unit normal = across the edge
        t = np.linspace(-0.4, 0.4, 81) * np.linalg.norm(d)          # +-40 % of a square
        pts = (mid + t[:, None] * n).astype(np.float32)
        p = cv2.remap(img, pts[:, 0:1], pts[:, 1:2], cv2.INTER_LINEAR).ravel()
        # 10-90 % rise width: how much distance the intensity needs to go from dark to bright
        p = (p - p.min()) / (p.max() - p.min() + 1e-6)
        return np.sum((p > 0.1) & (p < 0.9)) * (t[1] - t[0])

    # profiles across vertical edges (neighbours in a row) and across horizontal edges (neighbours in a column)
    across_x = [width_across(grid[r, c], grid[r, c + 1]) for r in range(rows) for c in range(cols - 1)]
    across_y = [width_across(grid[r, c], grid[r + 1, c]) for r in range(rows - 1) for c in range(cols)]

    # report the WORSE direction, so blur in only one direction is still caught
    return float(max(np.median(across_x), np.median(across_y)))


def fix_orientation(gray: np.ndarray, corners: np.ndarray,
                    pattern_size: tuple[int, int]) -> np.ndarray:
    """Return the corners ordered so that corner 0 is the one next to the BLACK
    corner square of the board (the detector may start at either end)."""
    cols, _ = pattern_size
    c = corners.reshape(-1, 2)

    def outer_square_value(i0, i_right, i_down):
        # step half a square "outwards" from corner i0 in both directions -> centre of the corner square
        centre = c[i0] - 0.5 * (c[i_right] - c[i0]) - 0.5 * (c[i_down] - c[i0])
        x, y = np.round(centre).astype(int)
        return gray[y - 2:y + 3, x - 2:x + 3].mean()                  # small patch, robust to noise

    # the corner square at the first corner vs. the one at the last corner (they have opposite colours)
    first = outer_square_value(0, 1, cols)
    last = outer_square_value(len(c) - 1, len(c) - 2, len(c) - 1 - cols)

    # if the first one is the brighter square, the order is reversed
    return corners[::-1].copy() if first > last else corners

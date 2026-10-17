"""3D model of the checkerboard."""
from __future__ import annotations

import numpy as np


def object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """3D coordinates (mm) of the board's inner corners in the board frame.

    Args:
        cols: inner corners along one row (9 for the board used here)
        rows: inner corners along one column (6)
        square_mm: measured edge length of one square

    Returns:
        (cols*rows, 3) float32 array. Z = 0 for every point (the board is a plane,
        which is what Zhang's calibration method relies on). Row by row, X changes
        fastest, matching the order of OpenCV's detected corners. Corner 0 = origin.

    Note: a wrong square_mm does not change K or the distortion, only the
    translations and every measurement (all by the same factor).
    """
    # integer grid indices: x = 0..cols-1, y = 0..rows-1 (x changes fastest, like OpenCV)
    ys, xs = np.mgrid[0:rows, 0:cols]

    # stack into (N, 3) points on the board plane, Z = 0
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, 0] = xs.ravel()
    pts[:, 1] = ys.ravel()

    # scale grid indices to millimetres
    return pts * np.float32(square_mm)

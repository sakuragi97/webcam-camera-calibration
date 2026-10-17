"""The calibration file (YAML via cv2.FileStorage, readable from Python and C++)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

import numpy as np

from .calibrate import CalibResult


def save_calibration(path: str | Path, result: CalibResult, extra: dict | None = None) -> None:
    """Write K, dist, image size, RMS and metadata (date, camera, board, resolution)
    so that any OpenCV program (Python or C++) can read it."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)

    # the numbers every OpenCV program needs (matrices are stored as !!opencv-matrix)
    fs.write("camera_matrix", np.asarray(result.K, dtype=np.float64))
    fs.write("dist_coeffs", np.asarray(result.dist, dtype=np.float64).reshape(1, -1))
    fs.write("image_width", int(result.image_size[0]))
    fs.write("image_height", int(result.image_size[1]))
    fs.write("rms_px", float(result.rms))

    # metadata: when, and anything else the caller passes (camera name, board, ...)
    fs.write("date", datetime.now().isoformat(timespec="seconds"))
    for key, value in (extra or {}).items():
        fs.write(key, value if isinstance(value, (int, float)) else str(value))
    fs.release()


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return (K, dist, (width, height)) from a file written by save_calibration."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"cannot read calibration file {path}")

    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat().ravel()
    size = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()))
    fs.release()
    return K, dist, size

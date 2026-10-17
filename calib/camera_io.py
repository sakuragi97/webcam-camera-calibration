"""Opening the webcam and checking what it really delivers."""
from __future__ import annotations

import time

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720,
                backend: int | None = None) -> cv2.VideoCapture:
    """Open a camera, request a resolution and verify it.

    Raises RuntimeError if the camera cannot be opened or delivers no frame.
    Prints a warning if the delivered resolution differs from the request
    (webcams silently fall back to the nearest supported mode).
    """
    cap = cv2.VideoCapture(index, cv2.CAP_ANY if backend is None else backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index} with backend {backend}")

    # MJPG is needed by most webcams for 1080p at 30 fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # a requested resolution is only a wish: check the shape of a real frame
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Camera {index} opened but no frame could be read")

    actual_h, actual_w = frame.shape[:2]
    if (actual_w, actual_h) != (width, height):
        print(f"Warning: requested {width}x{height}, got {actual_w}x{actual_h}")
    return cap


def actual_resolution(cap: cv2.VideoCapture) -> tuple[int, int]:
    """(width, height) of the frames the camera really delivers (from a frame, not a property)."""
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Frame grab failed")
    return frame.shape[1], frame.shape[0]


def measure_fps(cap: cv2.VideoCapture, n_frames: int = 60, warmup: int = 20) -> float:
    """Measured frames per second (CAP_PROP_FPS is often wrong)."""
    # skip the first frames: the camera is still adjusting exposure
    for _ in range(warmup):
        cap.read()

    good = 0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            good += 1
    elapsed = time.perf_counter() - t0
    if good == 0:
        raise RuntimeError("No frame could be read while measuring fps")
    return good / elapsed

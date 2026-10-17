"""Read-only access to the synthetic ground truth.

Used only to MEASURE results (how many pixels the detected corners are from the
true ones, how far the estimated K is from the true K). The calibration code never
reads it: the real webcam has no answer key either.

    from testbench import gt
    K_true, dist_true, size = gt.camera("full")
    c_true = gt.corners("full/img_03.jpg")        # (54, 1, 2) float32, pixels
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .make_dataset import DATA, main as _make


def ensure_dataset():
    if not (DATA / "ground_truth.json").exists():
        print("Synthetic test data not found: generating it now (1-2 minutes, only once)...")
        _make()


@lru_cache(maxsize=1)
def _gt():
    ensure_dataset()
    return json.loads((DATA / "ground_truth.json").read_text())


def data_dir(mode: str = "full") -> Path:
    """Folder with the images of a synthetic set: 'full', 'hd' or 'apps'."""
    ensure_dataset()
    return DATA / mode


def camera(mode: str = "full"):
    """True (K, dist, (width, height)) of the synthetic camera in 'full' or 'hd' mode."""
    c = _gt()["cameras"][mode]
    return np.array(c["K"]), np.array(c["dist"]), tuple(c["size"])


def corners(name: str) -> np.ndarray:
    """True corner positions (54, 1, 2) for e.g. 'full/img_03.jpg'.

    Order: row by row starting at the corner next to the BLACK corner square.
    A detector may return the reversed order (see calib.detect.fix_orientation).
    """
    return np.array(_gt()["images"][name]["corners_px"], np.float32).reshape(-1, 1, 2)


def pose(name: str):
    """True (rvec, tvec) of the board in the camera frame for an image."""
    im = _gt()["images"][name]
    return np.array(im["rvec"]), np.array(im["tvec"])


def card_edge_midpoints(app_name: str) -> np.ndarray:
    """True pixel positions of the bank card's left, right, top, bottom edge midpoints."""
    return np.array(_gt()["images"][f"apps/{app_name}"]["card_edge_midpoints_px"])


def board_square_mm() -> float:
    return float(_gt()["board"]["square_mm"])


# used by the tests
def _names(mode: str, good_only: bool = False):
    return sorted(k for k, v in _gt()["images"].items()
                  if k.startswith(mode + "/") and (not good_only or v["note"] == "good"))


def _note(name: str) -> str:
    return _gt()["images"][name]["note"]

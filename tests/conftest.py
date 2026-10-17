"""Shared helpers: every test compares the pipeline with the synthetic ground truth."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `calib` and `testbench` importable

from calib.board import object_points  # noqa: E402
from testbench import gt               # noqa: E402

PATTERN = (9, 6)
SQUARE = 25.0           # square size of the synthetic board (mm)


@pytest.fixture(scope="session", autouse=True)
def dataset():
    """Render the synthetic images once (1-2 minutes on the first run)."""
    gt.ensure_dataset()


def gray(name: str) -> np.ndarray:
    """Load e.g. 'full/img_03.jpg' as a grayscale image."""
    folder, file = name.split("/")
    return cv2.imread(str(gt.data_dir(folder) / file), cv2.IMREAD_GRAYSCALE)


def corner_error(found, truth):
    """(median, max) distance in px, trying both corner orders."""
    f, t = found.reshape(-1, 2), truth.reshape(-1, 2)
    d = min((np.linalg.norm(f - t, axis=1), np.linalg.norm(f[::-1] - t, axis=1)), key=np.median)
    return float(np.median(d)), float(d.max())


def noisy_corners(names, sigma, seed=0):
    """True corners + Gaussian noise: isolates calibration from detection."""
    rng = np.random.default_rng(seed)
    return [gt.corners(n) + rng.normal(0, sigma, gt.corners(n).shape).astype(np.float32) for n in names]


def objp():
    return object_points(*PATTERN, SQUARE)

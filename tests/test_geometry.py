"""Intrinsics under resize/crop, the calibration file, and measuring in mm."""
import cv2
import numpy as np
import pytest

from calib.calibrate import CalibResult
from calib.geometry import (crop_intrinsics, fov_degrees, image_to_board_plane, pixel_to_normalized,
                            ray_plane_intersection, scale_intrinsics)
from calib.io_yaml import load_calibration, save_calibration
from conftest import SQUARE
from testbench import gt

K, D, SIZE = gt.camera("full")


def test_fov_matches_opencv():
    fx, fy = fov_degrees(K, SIZE)
    ref = cv2.calibrationMatrixValues(K, SIZE, 0, 0)
    assert abs(fx - ref[0]) < 0.05 and abs(fy - ref[1]) < 0.05


def test_scale_intrinsics_with_half_pixel_convention():
    K2 = scale_intrinsics(K, 0.5, 0.25)
    assert not np.shares_memory(K2, K)
    assert np.allclose([K2[0, 0], K2[1, 1]], [1000, 501])
    assert np.allclose([K2[0, 2], K2[1, 2]], [(K[0, 2] + 0.5) * 0.5 - 0.5, (K[1, 2] + 0.5) * 0.25 - 0.5])


def test_crop_only_moves_principal_point():
    K2 = crop_intrinsics(K, 100, 50)
    assert np.allclose(K2[:2, :2], K[:2, :2])
    assert np.allclose(K2[:2, 2], K[:2, 2] - [100, 50])


def test_calibration_file_round_trip(tmp_path):
    res = CalibResult(K=K.copy(), dist=D.copy(), rvecs=[], tvecs=[], rms=0.12, image_size=SIZE)
    save_calibration(tmp_path / "cal.yaml", res, {"camera": "test"})
    K2, d2, s2 = load_calibration(tmp_path / "cal.yaml")
    assert np.allclose(K2, K) and np.allclose(d2, D) and s2 == SIZE


def test_ray_plane_intersection():
    o = np.array([0.0, 0.0, -10.0])
    dirs = np.array([[0, 0, 1.0], [1, 0, 1.0], [1, 0, 0.0]])
    X = ray_plane_intersection(o, dirs, np.zeros(3), np.array([0, 0, 1.0]))
    assert np.allclose(X[0], 0) and np.allclose(X[1], [10, 0, 0]) and np.all(np.isnan(X[2]))
    X2 = ray_plane_intersection(o, dirs[:2], np.array([0, 0, 5.0]), np.array([0, 0, -2.0]))
    assert np.allclose(X2[1], [15, 0, 5])          # non-unit, flipped normal


def test_pixel_to_normalized_removes_distortion():
    n = gt._names("full", good_only=True)[0]
    rv, tv = gt.pose(n)
    obj = np.zeros((54, 3))
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE
    ideal, _ = cv2.projectPoints(obj, rv, tv, np.eye(3), np.zeros(5))
    xn = pixel_to_normalized(gt.corners(n).reshape(-1, 2), K, D)
    assert np.abs(xn - ideal.reshape(-1, 2)).max() < 1e-4


@pytest.mark.parametrize("app", ["app_01_top.jpg", "app_02_oblique.jpg", "app_03_side.jpg"])
def test_bank_card_measured_in_mm(app):
    rv, tv = gt.pose(f"apps/{app}")
    XY = image_to_board_plane(gt.corners(f"apps/{app}").reshape(-1, 2), K, D, rv, tv)
    assert np.abs(XY - np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE).max() < 0.05
    m = image_to_board_plane(gt.card_edge_midpoints(app), K, D, rv, tv)
    assert abs(np.linalg.norm(m[1] - m[0]) - 85.60) < 0.1      # ISO 7810 card width
    assert abs(np.linalg.norm(m[3] - m[2]) - 53.98) < 0.1      # and height

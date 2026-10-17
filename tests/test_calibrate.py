"""Calibration and validation: recover the known camera from noisy corners."""
import cv2
import numpy as np
import pytest

from calib.calibrate import calibrate, per_view_errors, total_rms
from calib.evaluate import find_outlier_views, holdout_rms
from conftest import noisy_corners, objp
from testbench import gt

K_TRUE, D_TRUE, SIZE = gt.camera("full")
NAMES = gt._names("full", good_only=True)


@pytest.fixture(scope="module")
def setup():
    pts = noisy_corners(NAMES, sigma=0.1)
    obj = [objp()] * len(NAMES)
    return calibrate(obj, pts, SIZE, view_names=NAMES), obj, pts


def test_calibrate_recovers_true_camera(setup):
    res, _, _ = setup
    assert tuple(res.image_size) == SIZE
    est = [res.K[0, 0], res.K[1, 1], res.K[0, 2], res.K[1, 2]]
    true = [K_TRUE[0, 0], K_TRUE[1, 1], K_TRUE[0, 2], K_TRUE[1, 2]]
    assert np.abs(np.subtract(est, true)).max() < 1.5          # px, with 0.1 px corner noise
    assert abs(res.dist[0] - D_TRUE[0]) < 0.005
    assert 0.05 < res.rms < 0.3


def test_per_view_errors_match_opencv(setup):
    res, obj, pts = setup
    ref = cv2.calibrateCameraExtended(obj, pts, SIZE, None, None)[-1].ravel()
    assert np.allclose(per_view_errors(res, obj, pts), ref, atol=2e-3)


def test_total_rms_matches_opencv(setup):
    res, obj, pts = setup
    assert abs(total_rms(res, obj, pts) - res.rms) < 1e-3


def test_outlier_rule_is_robust_and_quiet():
    names = [f"v{i}" for i in range(7)]
    # one huge error must not hide a second, smaller outlier (a mean-based rule fails here)
    assert set(find_outlier_views(np.array([0.1, 0.12, 0.11, 0.09, 0.1, 0.9, 5.0]), names)) == {"v5", "v6"}
    assert find_outlier_views(np.array([0.12, 0.15, 0.1, 0.18, 0.14, 0.11, 0.2]), names) == []


def test_outlier_rule_finds_the_refocused_view():
    names = [n for n in gt._names("full") if "outside" not in gt._note(n)]
    pts = noisy_corners(names, sigma=0.1, seed=3)
    obj = [objp()] * len(names)
    res = calibrate(obj, pts, SIZE)
    out = find_outlier_views(per_view_errors(res, obj, pts), names)
    assert len(out) == 1 and "focus" in gt._note(out[0])


def test_holdout_rewards_true_K_and_punishes_wrong_K():
    names = NAMES[:6]
    pts = noisy_corners(names, sigma=0.2, seed=5)
    obj = [objp()] * len(names)
    h_true = holdout_rms(K_TRUE, D_TRUE, obj, pts)
    K_bad = K_TRUE.copy()
    K_bad[0, 0] *= 1.03                                       # 3 % wrong focal length
    assert 0.12 < h_true < 0.45
    assert holdout_rms(K_bad, D_TRUE, obj, pts) > 2 * h_true

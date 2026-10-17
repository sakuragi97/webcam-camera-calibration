"""Board model and corner detection against the true corner positions."""
import numpy as np
import pytest

from calib.board import object_points
from calib.detect import edge_sharpness, find_corners, fix_orientation
from conftest import PATTERN, corner_error, gray
from testbench import gt


def test_object_points_shape_order_units():
    p = object_points(9, 6, 25.0)
    assert p.shape == (54, 3) and p.dtype == np.float32
    assert np.allclose(p[:, 2], 0)
    assert np.allclose(p[1], [25, 0, 0])        # X changes fastest
    assert np.allclose(p[9], [0, 25, 0])        # second row
    assert np.allclose(object_points(9, 6, 30.0)[-1], [240, 150, 0])


@pytest.mark.parametrize("name", gt._names("full", good_only=True)[:4])
def test_find_corners_subpixel_accuracy(name):
    c = find_corners(gray(name), PATTERN)
    assert c is not None and c.shape == (54, 1, 2)
    med, mx = corner_error(c, gt.corners(name))
    assert med < 0.3 and mx < 1.5, f"median {med:.3f} px, max {mx:.3f} px"


def test_find_corners_rejects_partial_board():
    cut = [n for n in gt._names("full") if "outside" in gt._note(n)][0]
    assert find_corners(gray(cut), PATTERN) is None


def test_edge_sharpness_flags_one_directional_motion_blur():
    blur = [n for n in gt._names("full") if "blur" in gt._note(n)][0]
    good = gt._names("full", good_only=True)[:4]
    s_good = [edge_sharpness(gray(n), PATTERN, gt.corners(n)) for n in good]
    s_blur = edge_sharpness(gray(blur), PATTERN, find_corners(gray(blur), PATTERN))
    assert max(s_good) < 4 and s_blur > 2 * max(s_good)


def test_fix_orientation_puts_corner_0_at_black_square():
    n = gt._names("full", good_only=True)[0]
    truth = gt.corners(n)
    for given in (truth.copy(), truth[::-1].copy()):
        out = fix_orientation(gray(n), given, PATTERN)
        assert np.allclose(np.asarray(out).reshape(-1, 2), truth.reshape(-1, 2))

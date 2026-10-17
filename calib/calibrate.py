"""Camera calibration and reprojection errors."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CalibResult:
    """Everything a calibration produces. """

    K: np.ndarray                       # (3, 3) camera matrix
    dist: np.ndarray                    # (5,) k1, k2, p1, p2, k3  (or more with other models)
    rvecs: list[np.ndarray]             # one (3, 1) rotation vector per view: board -> camera
    tvecs: list[np.ndarray]             # one (3, 1) translation per view, same units as object points
    rms: float                          # overall RMS reprojection error in px, as returned by OpenCV
    image_size: tuple[int, int]         # (width, height)
    view_names: list[str] = field(default_factory=list)


def calibrate(object_points_list: list[np.ndarray], image_points_list: list[np.ndarray],
              image_size: tuple[int, int], flags: int = 0,
              view_names: list[str] | None = None) -> CalibResult:
    """Run OpenCV's calibration and package the result.

    Args:
        object_points_list: one (N, 3) float32 array per view (from board.object_points)
        image_points_list: one (N, 1, 2) float32 array per view (from detect.find_corners)
        image_size: (width, height)  -- careful, NumPy shapes are (height, width)
        flags: OpenCV calibration flags, e.g. to switch distortion off
    """
    # OpenCV estimates K, the distortion and one pose (rvec, tvec) per view in one call.
    # None, None = no initial guess for K and dist.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None, flags=flags)

    # package everything into one object
    return CalibResult(K=K, dist=dist.ravel(), rvecs=list(rvecs), tvecs=list(tvecs),
                       rms=float(rms), image_size=tuple(image_size),
                       view_names=list(view_names or []))


def per_view_errors(result: CalibResult, object_points_list: list[np.ndarray],
                    image_points_list: list[np.ndarray]) -> np.ndarray:
    """RMS reprojection error (px) of EACH view: project the 3D corners with the
    view's pose and compare with the detections (matches OpenCV's per-view errors).

    Returns: (n_views,) float array.
    """
    errors = []
    for objp, imgp, rvec, tvec in zip(object_points_list, image_points_list, result.rvecs, result.tvecs):
        # project the known 3D board corners into the image with this view's pose
        projected, _ = cv2.projectPoints(objp, rvec, tvec, result.K, result.dist)

        # distance (px) between each detected corner and where the model puts it
        d = np.linalg.norm(projected.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)

        # RMS = square root of the MEAN of the squared distances
        errors.append(np.sqrt(np.mean(d ** 2)))
    return np.array(errors)


def total_rms(result: CalibResult, object_points_list: list[np.ndarray],
              image_points_list: list[np.ndarray]) -> float:
    """Overall RMS over ALL corners of ALL views (equals result.rms).

    This is the quadratic mean of the per-view values, not their plain mean,
    so a few bad views raise it more."""
    # collect the squared distances of ALL corners of ALL views in one list
    squared = []
    for objp, imgp, rvec, tvec in zip(object_points_list, image_points_list, result.rvecs, result.tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, result.K, result.dist)
        d = np.linalg.norm(projected.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)
        squared.extend(d ** 2)

    # one RMS over every corner (NOT the mean of the per-view RMS values)
    return float(np.sqrt(np.mean(squared)))

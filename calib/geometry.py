"""Geometry with the intrinsics: resize/crop of K, field of view, pixel -> mm on a plane."""
from __future__ import annotations

import cv2
import numpy as np


def fov_degrees(K: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float]:
    """Horizontal and vertical field of view in degrees (pinhole, ignore distortion)."""
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]

    # half the image width seen at distance fx -> half angle = atan((w/2) / fx); full angle = 2x
    fov_x = 2 * np.degrees(np.arctan(w / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(h / (2 * fy)))
    return float(fov_x), float(fov_y)


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """K for the same camera after the image is RESIZED by sx (width) and sy (height)."""
    K2 = np.array(K, dtype=float)        # copy: never change the caller's K

    # focal lengths are measured in pixels, so they scale with the pixel size
    K2[0, 0] *= sx
    K2[1, 1] *= sy

    # principal point: pixel CENTRES sit at integer coordinates, so scale around
    # the image edge (-0.5), not around pixel 0  ->  c' = (c + 0.5) * s - 0.5
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K2


def crop_intrinsics(K: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """K after CROPPING the image so that pixel (x0, y0) becomes the new (0, 0)."""
    K2 = np.array(K, dtype=float)        # copy

    # cropping only shifts the pixel coordinates: f stays, the principal point moves
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return K2


def pixel_to_normalized(pts_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(N, 2) pixels -> (N, 2) undistorted normalized coordinates (x/z, y/z).
    Uses cv2.undistortPoints (iterative inverse of the distortion model)."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)

    # undistortPoints removes the lens distortion AND applies K^-1:
    # without P= it returns normalized coordinates (x/z, y/z) directly
    xn = cv2.undistortPoints(pts, K, dist)
    return xn.reshape(-1, 2)


def ray_plane_intersection(origin: np.ndarray, directions: np.ndarray,
                           plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Intersect rays X(λ) = origin + λ·d with a plane. Pure NumPy.

    Args:
        origin: (3,) common start point of all rays (the camera centre)
        directions: (N, 3) ray directions
        plane_point: (3,) any point on the plane
        plane_normal: (3,) plane normal
    Returns: (N, 3) intersection points. Points where a ray is parallel to the
    plane should become NaN instead of crashing.
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    p0 = np.asarray(plane_point, dtype=float)
    n = np.asarray(plane_normal, dtype=float)

    # plug X = o + lambda*d into n.(X - p0) = 0  ->  lambda = n.(p0 - o) / n.d
    denom = d @ n                                   # one value per ray
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(np.abs(denom) > 1e-12, (p0 - o) @ n / denom, np.nan)   # parallel ray -> NaN

    return o + lam[:, None] * d


def image_to_board_plane(pts_px: np.ndarray, K: np.ndarray, dist: np.ndarray,
                         rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """(N, 2) pixels -> (N, 2) X, Y on the board plane Z = 0 (board frame, mm).

    pixel -> normalized coordinates -> ray in the camera frame -> ray in the board
    frame (centre C = -R^T t, direction R^T d) -> intersection with Z = 0.
    """
    # 1) pixels -> normalized image coordinates -> ray directions (x, y, 1) in the CAMERA frame
    xn = pixel_to_normalized(pts_px, K, dist)
    d_cam = np.column_stack([xn, np.ones(len(xn))])

    # 2) move the rays into the BOARD frame. The pose maps board -> camera: X_cam = R X_board + t
    #    so the camera centre (X_cam = 0) is at C = -R^T t and directions rotate with R^T
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    t = np.asarray(tvec, dtype=float).reshape(3)
    C = -R.T @ t
    d_board = d_cam @ R                             # same as (R^T @ d_cam.T).T

    # 3) intersect with the board plane Z = 0 and keep X, Y (mm)
    X = ray_plane_intersection(C, d_board, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    return X[:, :2]

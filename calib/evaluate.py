"""Validation: inconsistent views and hold-out error."""
from __future__ import annotations

import cv2
import numpy as np


def find_outlier_views(errors: np.ndarray, names: list[str], factor: float = 2.5,
                       floor_px: float = 0.25) -> list[str]:
    """Names of views whose error is suspiciously large compared with a TYPICAL view.

    Rule: error > max(factor * median, floor_px). The median is used because one
    terrible view would pull a mean up and could hide itself below the threshold.
    """
    errors = np.asarray(errors, dtype=float)

    # "typical" error = MEDIAN: one terrible view cannot pull it up (the mean would be)
    typical = np.median(errors)

    # a view is suspicious if it is factor x worse than typical AND above an absolute floor
    # (the floor avoids flagging 0.2 px vs 0.08 px on very clean data)
    limit = max(factor * typical, floor_px)
    return [name for name, e in zip(names, errors) if e > limit]


def holdout_rms(K: np.ndarray, dist: np.ndarray, object_points_list: list[np.ndarray],
                image_points_list: list[np.ndarray]) -> float:
    """RMS reprojection error (px) on views that were NOT used for calibration.

    K and dist stay fixed; only each view's pose is estimated (solvePnP). A value
    close to the training RMS shows the model generalises instead of overfitting.
    """
    squared = []
    for objp, imgp in zip(object_points_list, image_points_list):
        # K and dist are fixed: only find where the board is (its pose) in this unseen view
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)

        # reproject with that pose and measure the distances to the detected corners
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        d = np.linalg.norm(projected.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)
        squared.extend(d ** 2)

    # one RMS over all corners of all held-out views
    return float(np.sqrt(np.mean(squared)))


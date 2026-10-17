"""Virtual camera that renders photos of a checkerboard.

It produces images from a camera whose intrinsics, distortion and poses are known
exactly (with noise, blur and a few deliberately bad views), so the calibration
pipeline can be tested against the truth before it is used on real webcam data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Board geometry
# --------------------------------------------------------------------------- #
@dataclass
class Board:
    """Checkerboard described by its INNER corners (not squares)."""

    cols: int = 9            # inner corners along a row   (10 squares)
    rows: int = 6            # inner corners along a column (7 squares)
    square_mm: float = 25.0  # measured edge length of one square

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    @property
    def n_corners(self) -> int:
        return self.cols * self.rows

    def _gt_corners_mm(self) -> np.ndarray:
        """(N, 3) float32 corner coordinates in mm, Z = 0, same order as OpenCV."""
        objp = np.zeros((self.n_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.square_mm
        return objp

    @property
    def size_mm(self) -> tuple[float, float]:
        """Width/height of the black-and-white area (squares, not corners)."""
        return ((self.cols + 1) * self.square_mm, (self.rows + 1) * self.square_mm)


# --------------------------------------------------------------------------- #
# Small geometry utilities
# --------------------------------------------------------------------------- #
def rot(rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0) -> np.ndarray:
    """Rotation matrix R = Rx @ Ry @ Rz (angles in degrees)."""
    a, b, c = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def distort_normalized(xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """OpenCV 5-parameter model (k1, k2, p1, p2, k3) applied to normalized coords."""
    k1, k2, p1, p2, k3 = [float(v) for v in np.ravel(dist)[:5]]
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], -1)


def undistort_normalized(xy_d: np.ndarray, dist: np.ndarray, iters: int = 40) -> np.ndarray:
    """Invert ``distort_normalized`` by fixed-point iteration (what OpenCV does)."""
    k1, k2, p1, p2, k3 = [float(v) for v in np.ravel(dist)[:5]]
    x0, y0 = xy_d[..., 0], xy_d[..., 1]
    x, y = x0.copy(), y0.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial
    return np.stack([x, y], -1)


# --------------------------------------------------------------------------- #
# Synthetic scene: printed board + credit card + ArUco marker on a table
# --------------------------------------------------------------------------- #
CARD_MM = (85.60, 53.98)            # ISO/IEC 7810 ID-1 (bank card) size


@dataclass
class SceneLayout:
    """Everything lies on the table plane Z = 0 of the board frame (mm)."""

    card_x0: float = 272.0          # card top-left corner
    card_y0: float = -12.0
    marker_mm: float = 50.0         # black square side of the ArUco marker
    marker_x0: float = 290.0        # marker top-left corner
    marker_y0: float = 92.0
    marker_id: int = 0
    tex_xmin: float = -95.0         # texture extent
    tex_xmax: float = 400.0
    tex_ymin: float = -85.0
    tex_ymax: float = 235.0

    def card_corners(self) -> np.ndarray:
        w, h = CARD_MM
        x, y = self.card_x0, self.card_y0
        return np.array([[x, y, 0], [x + w, y, 0], [x + w, y + h, 0], [x, y + h, 0]], np.float64)

    def card_edge_midpoints(self) -> np.ndarray:
        """Left, right, top, bottom edge midpoints (avoids the rounded corners)."""
        w, h = CARD_MM
        x, y = self.card_x0, self.card_y0
        return np.array([[x, y + h / 2, 0], [x + w, y + h / 2, 0],
                         [x + w / 2, y, 0], [x + w / 2, y + h, 0]], np.float64)

    def marker_center(self) -> np.ndarray:
        s = self.marker_mm
        return np.array([self.marker_x0 + s / 2, self.marker_y0 + s / 2, 0.0])


def make_scene_texture(board: Board, layout: SceneLayout, ppm: float = 6.0,
                       extras: bool = True, seed: int = 0) -> np.ndarray:
    """Top-down image of the table plane, ``ppm`` pixels per mm (grayscale)."""
    rng = np.random.default_rng(seed)
    W = int(round((layout.tex_xmax - layout.tex_xmin) * ppm))
    H = int(round((layout.tex_ymax - layout.tex_ymin) * ppm))
    # table: mid-grey with low-frequency texture
    noise = cv2.resize(rng.normal(0, 1, (H // 40 + 2, W // 40 + 2)).astype(np.float32), (W, H),
                       interpolation=cv2.INTER_CUBIC)
    tex = np.clip(128 + 9 * noise, 0, 255).astype(np.float32)

    def px(x_mm, y_mm):
        return (int(round((x_mm - layout.tex_xmin) * ppm)), int(round((y_mm - layout.tex_ymin) * ppm)))

    # A4 landscape sheet centred on the board
    bw, bh = board.size_mm
    cx, cy = -board.square_mm + bw / 2, -board.square_mm + bh / 2
    cv2.rectangle(tex, px(cx - 148.5, cy - 105), px(cx + 148.5, cy + 105), 238, -1)
    # squares: square (i, j) spans X in [(i-1)s, i s], Y in [(j-1)s, j s]
    s = board.square_mm
    for j in range(board.rows + 1):
        for i in range(board.cols + 1):
            if (i + j) % 2 == 0:
                cv2.rectangle(tex, px((i - 1) * s, (j - 1) * s), px(i * s, j * s), 22, -1)
    if extras:
        # bank card (dark) with a thin light print stripe
        c = layout.card_corners()
        cv2.rectangle(tex, px(*c[0, :2]), px(*c[2, :2]), 55, -1)
        cv2.rectangle(tex, px(c[0, 0] + 8, c[0, 1] + 10), px(c[0, 0] + 22, c[0, 1] + 20), 170, -1)
        # ArUco marker on a small white sheet
        m = layout.marker_mm
        x0, y0 = layout.marker_x0, layout.marker_y0
        cv2.rectangle(tex, px(x0 - 12, y0 - 12), px(x0 + m + 12, y0 + m + 12), 238, -1)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        side = int(round(m * ppm))
        mk = cv2.aruco.generateImageMarker(dictionary, layout.marker_id, side, borderBits=1)
        mk = np.where(mk > 127, 238, 22).astype(np.float32)
        u0, v0 = px(x0, y0)
        tex[v0:v0 + side, u0:u0 + side] = mk
    return tex


class SyntheticCamera:
    """Renders the scene texture with a known pinhole + distortion camera."""

    def __init__(self, K, dist, size, board: Board, layout: SceneLayout, ppm: float = 6.0):
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64).ravel()
        self.size = tuple(size)                     # (width, height)
        self.board, self.layout, self.ppm = board, layout, ppm
        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        xd = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]], -1)
        self._xn = undistort_normalized(xd, self.dist)       # ideal ray per pixel
        self.textures = {True: make_scene_texture(board, layout, ppm, True),
                         False: make_scene_texture(board, layout, ppm, False)}

    def project(self, pts_mm, rvec, tvec):
        p, _ = cv2.projectPoints(np.asarray(pts_mm, np.float64), rvec, tvec, self.K, self.dist)
        return p.reshape(-1, 2)

    def render(self, rvec, tvec, extras=False, rng=None, blur=0.6, noise=1.5,
               motion_blur=0, light_gradient=True) -> np.ndarray:
        rng = rng or np.random.default_rng()
        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
        A = np.column_stack([R[:, 0], R[:, 1], np.ravel(tvec)])     # (X, Y, 1) -> ray
        q = np.einsum("ij,hwj->hwi", np.linalg.inv(A),
                      np.concatenate([self._xn, np.ones(self._xn.shape[:2] + (1,))], -1))
        valid = q[..., 2] > 1e-9
        X = q[..., 0] / np.where(valid, q[..., 2], 1)
        Y = q[..., 1] / np.where(valid, q[..., 2], 1)
        L = self.layout
        mx = ((X - L.tex_xmin) * self.ppm).astype(np.float32)
        my = ((Y - L.tex_ymin) * self.ppm).astype(np.float32)
        mx[~valid] = -10
        my[~valid] = -10
        img = cv2.remap(self.textures[extras], mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=110)
        if light_gradient:
            h, w = img.shape
            gx = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.95, 1.08), w, dtype=np.float32)
            gy = np.linspace(rng.uniform(0.9, 1.0), rng.uniform(0.95, 1.05), h, dtype=np.float32)
            img = img * gy[:, None] * gx[None, :]
        if blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        if motion_blur > 0:
            k = np.zeros((motion_blur, motion_blur), np.float32)
            k[motion_blur // 2, :] = 1.0 / motion_blur
            img = cv2.filter2D(img, -1, k)
        img = img + rng.normal(0, noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # mild colour tint so it looks like a photo, saved as 3-channel JPEG
        return cv2.merge([img, np.clip(img * 0.97, 0, 255).astype(np.uint8),
                          np.clip(img * 1.02, 0, 255).astype(np.uint8)])

    def pose_looking_at(self, u, v, depth_mm, R, target_mm=None):
        """tvec such that board point ``target_mm`` appears near pixel (u, v)."""
        if target_mm is None:
            bw, bh = self.board.size_mm
            target_mm = np.array([bw / 2 - self.board.square_mm, bh / 2 - self.board.square_mm, 0.0])
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        t = depth_mm * ray - R @ np.asarray(target_mm, np.float64)
        rvec, _ = cv2.Rodrigues(R)
        return rvec.ravel(), t

    def corners_inside(self, rvec, tvec, margin=40) -> bool:
        b = self.board
        s = b.square_mm
        outline = np.array([[-s, -s, 0], [b.cols * s, -s, 0], [b.cols * s, b.rows * s, 0],
                            [-s, b.rows * s, 0]], np.float64)
        p = self.project(outline, rvec, tvec)
        w, h = self.size
        return bool(np.all((p[:, 0] > margin) & (p[:, 0] < w - margin) &
                           (p[:, 1] > margin) & (p[:, 1] < h - margin)))


def default_synthetic_camera():
    """A camera similar to a webcam / wide phone lens with clear barrel distortion."""
    size = (2400, 1800)
    K = np.array([[2000.0, 0, 1212.0], [0, 2004.0, 886.0], [0, 0, 1]])
    dist = np.array([-0.30, 0.11, 0.0008, -0.0005, -0.015])
    return K, dist, size


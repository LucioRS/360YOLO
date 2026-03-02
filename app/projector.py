from __future__ import annotations
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from app.config import ViewSpec


@dataclass
class RemapView:
    spec: ViewSpec
    map_x: np.ndarray  # float32 [H,W]
    map_y: np.ndarray  # float32 [H,W]


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float32)


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


class EquirectProjector:
    def __init__(self, pano_w: int, pano_h: int, views: list[ViewSpec]) -> None:
        self.pano_w = int(pano_w)
        self.pano_h = int(pano_h)
        self.views = views
        self.remaps: list[RemapView] = []

        t0 = time.perf_counter()
        for spec in views:
            mx, my = self._build_remap(spec)
            self.remaps.append(RemapView(spec, mx, my))
        self.build_time_ms = (time.perf_counter() - t0) * 1000.0

    def _build_remap(self, spec: ViewSpec) -> tuple[np.ndarray, np.ndarray]:
        w, h = spec.out_w, spec.out_h
        hfov = math.radians(spec.hfov_deg)
        tan_half_h = math.tan(hfov / 2.0)
        tan_half_v = tan_half_h * (h / w)

        xs = ((np.arange(w, dtype=np.float32) + 0.5) / w) * 2.0 - 1.0
        ys = 1.0 - ((np.arange(h, dtype=np.float32) + 0.5) / h) * 2.0
        xx, yy = np.meshgrid(xs * tan_half_h, ys * tan_half_v)

        dirs = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float32)
        dirs /= np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)

        R = (_rot_y(spec.yaw_deg) @ _rot_x(-spec.pitch_deg)).astype(np.float32)
        d = dirs.reshape(-1, 3) @ R.T
        dx, dy, dz = d[:, 0], np.clip(d[:, 1], -1.0, 1.0), d[:, 2]

        lon = np.arctan2(dx, dz)
        lat = np.arcsin(dy)

        map_x = ((lon / (2.0 * math.pi)) + 0.5) * self.pano_w
        map_y = (0.5 - (lat / math.pi)) * self.pano_h

        map_x = np.mod(map_x, self.pano_w).astype(np.float32)
        map_y = np.clip(map_y, 0, self.pano_h - 1).astype(np.float32)
        return map_x.reshape(h, w), map_y.reshape(h, w)

    def project(self, pano_bgr: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for rv in self.remaps:
            out[rv.spec.name] = cv2.remap(
                pano_bgr, rv.map_x, rv.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return out
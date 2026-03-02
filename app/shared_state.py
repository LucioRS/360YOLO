from __future__ import annotations
from dataclasses import dataclass, field
import threading
import time
from typing import Optional, Dict

import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    ts_capture: float
    pano_bgr: np.ndarray  # full-res pano for projection/inference


@dataclass
class InferenceStats:
    frame_id: int
    ts_capture: float
    ts_infer_start: float
    ts_infer_end: float
    total_dets: int
    per_view_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class PreviewImages:
    frame_id: int = -1
    ts: float = 0.0
    pano_bgr_small: Optional[np.ndarray] = None
    views_bgr_small: Dict[str, np.ndarray] = field(default_factory=dict)


class SharedState:
    """
    Latest-value slots (no unbounded queue):
      - Capture thread overwrites latest pano frame + pano preview
      - Inference thread processes latest pano frame and overwrites view previews + stats
      - GUI thread reads snapshots and uploads to textures
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Optional[FramePacket] = None
        self._latest_stats: Optional[InferenceStats] = None
        self._latest_preview: PreviewImages = PreviewImages()

        self._paused = False
        self._conf = 0.25
        self._imgsz = 640

        self._capture_fps_est = 0.0
        self._infer_fps_est = 0.0
        self._status = "starting..."
        self._last_error: Optional[str] = None

        self._projector_build_ms: Optional[float] = None
        self._model_loaded = False

        self._pano_overlays = []          # list of dicts: {"u": np.ndarray, "v": np.ndarray, "label": str}
        self._pano_overlays_frame_id = -1

        self._start_ts = time.perf_counter()

    # ----- runtime controls -----
    def get_runtime(self) -> tuple[bool, float, int]:
        with self._lock:
            return self._paused, self._conf, self._imgsz

    def set_paused(self, v: bool) -> None:
        with self._lock:
            self._paused = bool(v)

    def toggle_paused(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def set_conf(self, v: float) -> None:
        with self._lock:
            self._conf = float(v)

    def set_imgsz(self, v: int) -> None:
        with self._lock:
            self._imgsz = int(v)

    # ----- capture -----
    def put_frame(self, pkt: FramePacket) -> None:
        with self._lock:
            self._latest_frame = pkt

    def get_latest_frame(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest_frame

    def put_pano_preview(self, frame_id: int, pano_small: np.ndarray, ts: float) -> None:
        pano_small = np.ascontiguousarray(pano_small)
        with self._lock:
            self._latest_preview.frame_id = frame_id
            self._latest_preview.ts = ts
            self._latest_preview.pano_bgr_small = pano_small

    # ----- inference -----
    def put_stats(self, st: InferenceStats) -> None:
        dt = max(st.ts_infer_end - st.ts_infer_start, 1e-6)
        fps = 1.0 / dt
        with self._lock:
            self._latest_stats = st
            self._infer_fps_est = fps if self._infer_fps_est == 0.0 else (0.9 * self._infer_fps_est + 0.1 * fps)

    def get_latest_stats(self) -> Optional[InferenceStats]:
        with self._lock:
            return self._latest_stats

    def put_view_previews(self, frame_id: int, views_small: Dict[str, np.ndarray], ts: float) -> None:
        views_small = {k: np.ascontiguousarray(v) for k, v in views_small.items()}
        with self._lock:
            self._latest_preview.frame_id = frame_id
            self._latest_preview.ts = ts
            self._latest_preview.views_bgr_small = views_small

    def get_latest_previews(self) -> PreviewImages:
        with self._lock:
            return self._latest_preview

    # ----- status -----
    def set_status(self, s: str) -> None:
        with self._lock:
            self._status = s

    def set_error(self, e: str) -> None:
        with self._lock:
            self._last_error = e

    def clear_error(self) -> None:
        with self._lock:
            self._last_error = None

    def set_capture_fps(self, fps: float) -> None:
        with self._lock:
            self._capture_fps_est = fps if self._capture_fps_est == 0.0 else (0.9 * self._capture_fps_est + 0.1 * fps)

    def set_projector_build_ms(self, ms: float) -> None:
        with self._lock:
            self._projector_build_ms = ms

    def set_model_loaded(self, v: bool) -> None:
        with self._lock:
            self._model_loaded = v
    
    def put_pano_overlays(self, frame_id: int, overlays: list[dict], ts: float) -> None:
        with self._lock:
            self._pano_overlays = overlays
            self._pano_overlays_frame_id = frame_id

    def get_pano_overlays(self) -> tuple[int, list[dict]]:
        with self._lock:
            return self._pano_overlays_frame_id, self._pano_overlays

    def ui_snapshot(self) -> dict:
        with self._lock:
            uptime = time.perf_counter() - self._start_ts
            stats = self._latest_stats
            infer_ms = None
            e2e_ms = None
            total = None
            per_view = {}
            if stats:
                infer_ms = (stats.ts_infer_end - stats.ts_infer_start) * 1000.0
                e2e_ms = (stats.ts_infer_end - stats.ts_capture) * 1000.0
                total = stats.total_dets
                per_view = dict(stats.per_view_counts)

            return {
                "uptime_s": uptime,
                "status": self._status,
                "last_error": self._last_error,
                "capture_fps": self._capture_fps_est,
                "infer_fps": self._infer_fps_est,
                "infer_ms": infer_ms,
                "e2e_ms": e2e_ms,
                "paused": self._paused,
                "conf": self._conf,
                "imgsz": self._imgsz,
                "total_dets": total,
                "per_view": per_view,
                "projector_build_ms": self._projector_build_ms,
                "model_loaded": self._model_loaded,
            }
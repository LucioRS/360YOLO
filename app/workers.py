from __future__ import annotations
import threading
import time
from typing import Optional, Dict

import cv2
import numpy as np

from camera import FFmpegDShowCamera
from config import AppConfig
from detector import YOLODetectorCPU, Det
from projector import EquirectProjector
from shared_state import SharedState, FramePacket, InferenceStats


def draw_dets(img: np.ndarray, dets: list[Det]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{d.cls_name} {d.conf:.2f}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
    return out


# -----------------------------
# Pano overlay helpers
# -----------------------------
def _sample_rect_border(x1, y1, x2, y2, n: int = 10) -> np.ndarray:
    """
    Sample points along a rectangle border (view coords).
    Keep n small for performance.
    """
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    xs_top = np.linspace(x1, x2, n, dtype=np.float32)
    xs_bot = np.linspace(x2, x1, n, dtype=np.float32)
    ys_lft = np.linspace(y2, y1, n, dtype=np.float32)
    ys_rgt = np.linspace(y1, y2, n, dtype=np.float32)

    top = np.stack([xs_top, np.full(n, y1, np.float32)], axis=1)
    right = np.stack([np.full(n, x2, np.float32), ys_rgt], axis=1)
    bottom = np.stack([xs_bot, np.full(n, y2, np.float32)], axis=1)
    left = np.stack([np.full(n, x1, np.float32), ys_lft], axis=1)

    return np.concatenate([top, right, bottom, left], axis=0)


def _map_view_poly_to_pano(map_x: np.ndarray, map_y: np.ndarray, poly_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses remap grids: (view x,y) -> (pano u,v)
    """
    h, w = map_x.shape
    ix = np.clip(np.round(poly_xy[:, 0]).astype(np.int32), 0, w - 1)
    iy = np.clip(np.round(poly_xy[:, 1]).astype(np.int32), 0, h - 1)
    u = map_x[iy, ix]
    v = map_y[iy, ix]
    return u.astype(np.float32), v.astype(np.float32)


def _draw_poly_on_pano_small(pano_small: np.ndarray, u: np.ndarray, v: np.ndarray, sx: float, sy: float,
                            color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    pts = np.stack([u * sx, v * sy], axis=1)
    pts = np.round(pts).astype(np.int32)
    cv2.polylines(pano_small, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return pts


def _draw_label(pano_small: np.ndarray, x: int, y: int, text: str) -> None:
    """
    Draw readable label (black outline + green text).
    """
    x = int(max(0, min(pano_small.shape[1] - 1, x)))
    y = int(max(0, min(pano_small.shape[0] - 1, y)))

    # outline
    cv2.putText(pano_small, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    # main text
    cv2.putText(pano_small, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


class StoppableThread(threading.Thread):
    def __init__(self, name: str):
        super().__init__(name=name, daemon=True)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()


class CaptureWorker(StoppableThread):
    """
    Produces:
      - latest full pano frame (for inference)
      - pano preview image at pano_preview_max_fps

    IMPORTANT:
      The pano preview is ALWAYS overlaid with the latest detections (stored in SharedState),
      so boxes/polys do NOT flicker even if inference preview is slower than pano preview.
    """
    def __init__(self, cfg: AppConfig, state: SharedState):
        super().__init__("CaptureWorker")
        c = cfg.camera
        self.cfg = cfg
        self.state = state
        self.cam = FFmpegDShowCamera(c.dshow_name, c.width, c.height, c.fps, c.in_pixfmt)
        self._last_preview_ts = 0.0

    def run(self) -> None:
        c = self.cfg.camera
        frame_id = 0
        try:
            self.cam.open()
            self.state.set_status(f"Camera: {c.dshow_name} ({c.width}x{c.height}@{c.fps} in={c.in_pixfmt})")
            self.state.clear_error()
        except Exception as e:
            self.state.set_error(f"Camera open failed: {e}")
            return

        last = time.perf_counter()

        try:
            while not self.stopped():
                pano = self.cam.read_frame()
                now = time.perf_counter()

                if pano is None:
                    self.state.set_error("Camera read failed (ffmpeg short read).")
                    time.sleep(0.02)
                    continue

                dt = now - last
                last = now
                if dt > 1e-6:
                    self.state.set_capture_fps(1.0 / dt)

                # Latest full frame for inference
                self.state.put_frame(FramePacket(frame_id=frame_id, ts_capture=now, pano_bgr=pano))

                # pano preview throttle/downscale (with overlays)
                if (now - self._last_preview_ts) >= (1.0 / max(1e-3, c.pano_preview_max_fps)):
                    H, W = pano.shape[:2]
                    new_w = c.pano_preview_width
                    new_h = int(new_w * (H / W))
                    pano_small = cv2.resize(pano, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    sx = new_w / W
                    sy = new_h / H

                    # Get latest overlays produced by inference worker
                    _, overlays = self.state.get_pano_overlays()
                    for ov in overlays:
                        u = ov["u"]
                        v = ov["v"]
                        label = ov.get("label", "")

                        pts = _draw_poly_on_pano_small(pano_small, u, v, sx, sy, color=(0, 255, 0), thickness=2)

                        if label:
                            # label near polygon centroid
                            cx = int(np.mean(pts[:, 0]))
                            cy = int(np.mean(pts[:, 1]))
                            _draw_label(pano_small, cx, cy, label)

                    self.state.put_pano_preview(frame_id, pano_small, now)
                    self._last_preview_ts = now

                frame_id += 1

        finally:
            self.cam.close()


class InferenceWorker(StoppableThread):
    """
    Produces:
      - stats
      - view previews (annotated)
      - pano overlays (polygons + labels) stored in state (NOT pano image itself)
    """
    def __init__(self, cfg: AppConfig, state: SharedState, projector: EquirectProjector, detector: YOLODetectorCPU):
        super().__init__("InferenceWorker")
        self.cfg = cfg
        self.state = state
        self.projector = projector
        self.detector = detector
        self._last_frame_id: Optional[int] = None
        self._last_preview_ts = 0.0

    def run(self) -> None:
        self.state.set_model_loaded(True)
        idle = max(0.001, self.cfg.idle_sleep_ms / 1000.0)

        while not self.stopped():
            paused, conf, imgsz = self.state.get_runtime()
            if paused:
                time.sleep(idle)
                continue

            pkt = self.state.get_latest_frame()
            if pkt is None or pkt.frame_id == self._last_frame_id:
                time.sleep(idle)
                continue

            # Detach from capture buffer
            pano = pkt.pano_bgr.copy()

            t0 = time.perf_counter()

            # 1) project views from pano
            views = self.projector.project(pano)

            # 2) infer on batch
            dets = self.detector.infer_batch(views, conf=conf, imgsz=imgsz)

            t1 = time.perf_counter()

            per_view_counts = {k: len(v) for k, v in dets.items()}
            total = sum(per_view_counts.values())

            self.state.put_stats(InferenceStats(
                frame_id=pkt.frame_id,
                ts_capture=pkt.ts_capture,
                ts_infer_start=t0,
                ts_infer_end=t1,
                total_dets=total,
                per_view_counts=per_view_counts,
            ))

            # 3) build annotated view previews + pano overlays (throttled)
            now = time.perf_counter()
            inf = self.cfg.inference

            if (now - self._last_preview_ts) >= (1.0 / max(1e-3, inf.view_preview_max_fps)):
                # ---- 3a) view previews (annotated) ----
                previews: Dict[str, np.ndarray] = {}
                for name, img in views.items():
                    img_anno = draw_dets(img, dets.get(name, []))

                    # downscale for GUI upload if needed
                    h, w = img_anno.shape[:2]
                    new_w = inf.view_preview_width
                    if w != new_w:
                        new_h = int(new_w * (h / w))
                        img_anno = cv2.resize(img_anno, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    previews[name] = img_anno

                self.state.put_view_previews(pkt.frame_id, previews, now)

                # ---- 3b) pano overlays (store only geometry+labels) ----
                overlays: list[dict] = []
                max_boxes_per_view = 30  # cap to keep it fast

                for rv in self.projector.remaps:
                    vname = rv.spec.name
                    view_dets = dets.get(vname, [])
                    if not view_dets:
                        continue

                    for d in view_dets[:max_boxes_per_view]:
                        x1, y1, x2, y2 = d.xyxy
                        poly = _sample_rect_border(x1, y1, x2, y2, n=10)
                        u, v = _map_view_poly_to_pano(rv.map_x, rv.map_y, poly)

                        overlays.append({
                            "u": u,
                            "v": v,
                            "label": f"{d.cls_name} {d.conf:.2f}",
                        })

                self.state.put_pano_overlays(pkt.frame_id, overlays, now)

                self._last_preview_ts = now

            self._last_frame_id = pkt.frame_id
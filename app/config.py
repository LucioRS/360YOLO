from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ViewSpec:
    name: str
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    out_w: int
    out_h: int


def make_views(n: int) -> List[ViewSpec]:
    """
    Reasonable CPU-friendly default layouts.
    - 4: yaw 0/90/180/270
    - 6: 4 yaw + pitch +30, -30 (front)
    - 8: 4 yaw + pitch +30 at yaw 0/180 and pitch -30 at yaw 0/180
    """
    if n not in (4, 6, 8):
        raise ValueError("n must be 4, 6, or 8")

    base = [
        ViewSpec("front", 0.0, 0.0, 90.0, 640, 640),
        ViewSpec("right", 90.0, 0.0, 90.0, 640, 640),
        ViewSpec("back", 180.0, 0.0, 90.0, 640, 640),
        ViewSpec("left", 270.0, 0.0, 90.0, 640, 640),
    ]
    if n == 4:
        return base
    if n == 6:
        # return base + [
        #     ViewSpec("up_front", 0.0, +30.0, 90.0, 640, 640),
        #     ViewSpec("down_front", 0.0, -30.0, 90.0, 640, 640),
        # ]
        views = [
            ViewSpec("front", 0.0, 0.0, 60.0, 320, 320),
            ViewSpec("right_front", 60.0, 0.0, 60.0, 320, 320),
            ViewSpec("right_back", 120.0, 0.0, 60.0, 320, 320),
            ViewSpec("back", 180.0, 0.0, 60.0, 320, 320),
            ViewSpec("left_back", 240.0, 0.0, 60.0, 320, 320),
            ViewSpec("left_front", 300.0, 0.0, 60.0, 320, 320)
        ]
        return views
    # n == 8
    return base + [
        ViewSpec("up_front", 0.0, +30.0, 90.0, 640, 640),
        ViewSpec("up_back", 180.0, +30.0, 90.0, 640, 640),
        ViewSpec("down_front", 0.0, -30.0, 90.0, 640, 640),
        ViewSpec("down_back", 180.0, -30.0, 90.0, 640, 640),
    ]


@dataclass
class CameraConfig:
    # Use FFmpeg DirectShow by device NAME (recommended for THETA UVC).
    dshow_name: str = "RICOH THETA UVC"
    in_pixfmt: str = "nv12"  # "nv12" or "yuyv422" (from ffmpeg -list_options)
    width: int = 1920
    height: int = 960
    fps: int = 30

    # Tip: GUI preview downscale (don’t upload 3840x1920 every frame)
    pano_preview_width = width
    pano_preview_max_fps = fps


@dataclass
class InferenceConfig:
    model_path: str = "./models/yolo26n.pt"
    imgsz: int = 320
    conf: float = 0.5
    device: str = "cpu"

    # GUI preview for views
    view_preview_width: int = 320
    view_preview_max_fps: float = 15.0


@dataclass
class ProjectionConfig:
    num_views: int = 6
    views: List[ViewSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.views:
            self.views = make_views(self.num_views)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    # Worker idling
    idle_sleep_ms: int = 2
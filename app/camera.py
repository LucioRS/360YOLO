from __future__ import annotations
import subprocess
from typing import Optional

import numpy as np


class FFmpegDShowCamera:
    """
    Captures from DirectShow device by name via FFmpeg and outputs BGR24 frames.
    IMPORTANT: Uses readinto() + preallocated buffers to avoid per-frame allocations.
    """
    def __init__(self, name: str, w: int, h: int, fps: int, in_pixfmt: str):
        self.name = name
        self.w = int(w)
        self.h = int(h)
        self.fps = int(fps)
        self.in_pixfmt = in_pixfmt

        self._proc: Optional[subprocess.Popen] = None

        # Double-buffered frames (capture overwrites alternating buffers)
        self._frames = [
            np.empty((self.h, self.w, 3), dtype=np.uint8),
            np.empty((self.h, self.w, 3), dtype=np.uint8),
        ]
        self._mvs = [memoryview(f).cast("B") for f in self._frames]
        self._idx = 0

    def open(self) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "dshow",
            "-rtbufsize", "512M",
            "-video_size", f"{self.w}x{self.h}",
            "-framerate", str(self.fps),
            "-pixel_format", self.in_pixfmt,   # nv12 or yuyv422
            "-i", f"video={self.name}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        # Use DEVNULL so ffmpeg stderr can't fill and block the process
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,  # unbuffered
        )

    def read_frame(self) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            return None

        mv = self._mvs[self._idx]
        total = len(mv)
        got = 0

        # Fill the whole buffer (readinto may return partial reads)
        while got < total:
            n = self._proc.stdout.readinto(mv[got:])
            if n is None or n == 0:
                return None
            got += n

        frame = self._frames[self._idx]
        self._idx ^= 1
        return frame

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
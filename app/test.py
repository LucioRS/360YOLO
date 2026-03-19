"""
HelloImGui + OpenGL texture + FFmpeg camera capture (single file).

Dependencies:
  pip install imgui-bundle PyOpenGL
Requirements:
  ffmpeg available in PATH

Run:
  python app/test.py --name "RICOH THETA UVC" --size 3840x1920 --fps 30 --in-pixfmt nv12
"""

import argparse
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from imgui_bundle import hello_imgui, imgui

# PyOpenGL
from OpenGL.GL import (
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_LINEAR,
    GL_CLAMP_TO_EDGE,
    GL_UNPACK_ALIGNMENT,
    GL_RGB,
    GL_BGR,
    GL_UNSIGNED_BYTE,
)


def parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


@dataclass
class LatestFrame:
    frame_id: int = -1
    ts: float = 0.0
    bgr_bytes: Optional[bytes] = None
    status: str = "starting..."
    fps_est: float = 0.0


class FFmpegCaptureThread:
    """
    Reads raw bgr24 frames from FFmpeg stdout and keeps only the latest frame.
    """
    def __init__(self, name: str, w: int, h: int, fps: int, in_pixfmt: str):
        self.name = name
        self.w = w
        self.h = h
        self.fps = fps
        self.in_pixfmt = in_pixfmt

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = LatestFrame()

        self._proc: Optional[subprocess.Popen] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def snapshot(self) -> LatestFrame:
        with self._lock:
            # Return a shallow copy (bytes is immutable, safe)
            return LatestFrame(
                frame_id=self._latest.frame_id,
                ts=self._latest.ts,
                bgr_bytes=self._latest.bgr_bytes,
                status=self._latest.status,
                fps_est=self._latest.fps_est,
            )

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._latest.status = status

    def _run(self) -> None:
        frame_bytes = self.w * self.h * 3

        # Force input pixel_format (nv12 or yuyv422) as advertised by your dshow device.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "dshow",
            "-rtbufsize", "512M",
            "-video_size", f"{self.w}x{self.h}",
            "-framerate", str(self.fps),
            "-pixel_format", self.in_pixfmt,
            "-i", f"video={self.name}",
            # Convert to raw bgr24 for Python/OpenGL upload:
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7,
            )
            self._proc = proc
            self._set_status(f"ffmpeg started (in={self.in_pixfmt} -> bgr24)")

            last = time.perf_counter()
            fps_est = 0.0
            frame_id = 0

            while not self._stop.is_set():
                if proc.stdout is None:
                    raise RuntimeError("ffmpeg stdout is None")

                raw = proc.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    err = ""
                    try:
                        if proc.stderr is not None:
                            err = proc.stderr.read(2000).decode(errors="ignore")
                    except Exception:
                        pass
                    self._set_status(f"short read ({len(raw)} bytes). ffmpeg: {err.strip() or 'no stderr'}")
                    time.sleep(0.05)
                    continue

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 1e-6:
                    inst = 1.0 / dt
                    fps_est = inst if fps_est == 0.0 else (0.9 * fps_est + 0.1 * inst)

                with self._lock:
                    self._latest.frame_id = frame_id
                    self._latest.ts = now
                    self._latest.bgr_bytes = raw  # bytes is immutable, safe to share
                    self._latest.status = "running"
                    self._latest.fps_est = fps_est

                frame_id += 1

        except FileNotFoundError:
            self._set_status("ERROR: ffmpeg not found in PATH")
        except Exception as e:
            self._set_status(f"ERROR: {e}")


class GLVideoTexture:
    """
    Owns an OpenGL texture and updates it from raw BGR bytes.
    """
    def __init__(self) -> None:
        self.tex_id: int = 0
        self.w: int = 0
        self.h: int = 0

    def ensure_created(self) -> None:
        if self.tex_id != 0:
            return
        self.tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def ensure_size(self, w: int, h: int) -> None:
        self.ensure_created()
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # Allocate storage (no data yet)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, None)

    def update_bgr(self, w: int, h: int, bgr_bytes: bytes) -> None:
        if not bgr_bytes:
            return
        self.ensure_size(w, h)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, bgr_bytes)

    def destroy(self) -> None:
        if self.tex_id != 0:
            try:
                glDeleteTextures([self.tex_id])
            except Exception:
                # In case context is already gone; ignore
                pass
            self.tex_id = 0
            self.w = 0
            self.h = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help='DirectShow device name, e.g. "RICOH THETA UVC"')
    ap.add_argument("--size", default="1920x960", help="WxH (e.g. 1920x960, 3840x1920)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--in-pixfmt", choices=["nv12", "yuyv422"], default="nv12",
                    help="Must match what ffmpeg -list_options shows for your camera")
    args = ap.parse_args()

    w, h = parse_size(args.size)

    cap = FFmpegCaptureThread(args.name, w, h, args.fps, args.in_pixfmt)
    cap.start()

    tex = GLVideoTexture()
    ui = {"fit_width": True}

    def show_gui() -> None:
        snap = cap.snapshot()

        imgui.begin("Camera (HelloImGui + OpenGL texture)")

        imgui.text(f'Device: {args.name}')
        imgui.text(f"Mode: {w}x{h}@{args.fps}   in_pixfmt={args.in_pixfmt}")
        imgui.text(f"Status: {snap.status}")
        imgui.text(f"Capture FPS (est): {snap.fps_est:.1f}")
        imgui.separator()

        _, ui["fit_width"] = imgui.checkbox("Fit to window width", ui["fit_width"])

        if snap.bgr_bytes is None:
            imgui.text_disabled("Waiting for frames...")
            imgui.end()
            return

        # Update texture with latest frame
        tex.update_bgr(w, h, snap.bgr_bytes)

        avail_w = max(200.0, float(imgui.get_content_region_avail().x))
        if ui["fit_width"]:
            disp_w = int(avail_w)
            disp_h = int(disp_w * (h / w))
        else:
            disp_w, disp_h = w, h

        # ImGui Bundle uses ImTextureRef (wraps OpenGL GLuint) :contentReference[oaicite:1]{index=1}
        tex_ref = imgui.ImTextureRef(tex.tex_id)

        # Flip vertically (OpenGL texture origin differs from image origin)
        imgui.image(tex_ref, (disp_w, disp_h), uv0=(0, 1), uv1=(1, 0))

        imgui.end()

    def before_exit() -> None:
        # Called while OpenGL is still alive
        tex.destroy()
        cap.stop()

    runner = hello_imgui.RunnerParams()
    runner.app_window_params.window_title = "THETA UVC Viewer"
    runner.app_window_params.window_geometry.size = (1100, 700)

    # For live video, disable idling (otherwise it may slow down when idle) :contentReference[oaicite:2]{index=2}
    runner.fps_idling.enable_idling = False

    runner.callbacks.show_gui = show_gui
    runner.callbacks.before_exit = before_exit

    hello_imgui.run(runner)


if __name__ == "__main__":
    main()
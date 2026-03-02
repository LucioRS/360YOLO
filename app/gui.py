from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math

import numpy as np
from imgui_bundle import hello_imgui, imgui

from OpenGL.GL import (
    glBindTexture, glDeleteTextures, glGenTextures, glPixelStorei,
    glTexImage2D, glTexParameteri, glTexSubImage2D,
    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_UNPACK_ALIGNMENT,
    GL_RGB, GL_BGR, GL_UNSIGNED_BYTE,
)

from .shared_state import SharedState
from .ptz_shader import PTZRenderer, PTZState


# =========================
# OpenGL texture wrapper
# =========================
@dataclass
class GLTexture:
    tex_id: int = 0
    w: int = 0
    h: int = 0

    def ensure(self) -> None:
        if self.tex_id != 0:
            return
        self.tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def allocate(self, w: int, h: int) -> None:
        self.ensure()
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, None)

    def upload_bgr(self, img_bgr: np.ndarray) -> None:
        h, w = img_bgr.shape[:2]
        self.allocate(w, h)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, img_bgr)

    def destroy(self) -> None:
        if self.tex_id != 0:
            try:
                glDeleteTextures([self.tex_id])
            except Exception:
                pass
            self.tex_id = 0
            self.w = 0
            self.h = 0


# =========================
# Geometry / ROI helpers
# =========================
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s],
                     [0.0,  s,  c]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=np.float32)


def _frustum_outline_uv_for_thumbnail(ptz: PTZState, aspect: float, samples_per_edge: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a polyline (u,v) of the PTZ frustum outline on the pano for the THUMBNAIL overlay.

    Conventions / fixes:
    - Fix left/right direction: use yaw = -yaw (grab-style motion)
    - Fix up/down direction: use pitch = -pitch (grab-style motion)  <-- YOUR REQUEST #1
    - Draw as polyline (not bbox) => no stretching near poles.
    """
    yaw = math.radians(-ptz.yaw_deg)
    pitch = math.radians(-ptz.pitch_deg)  # <-- IMPORTANT: fixes vertical direction in thumbnail
    hfov = math.radians(ptz.hfov_deg)

    tan_half_h = math.tan(hfov * 0.5)
    tan_half_v = tan_half_h / max(1e-6, aspect)

    # Keep same rotation structure as shader (rotY * rotX(-pitch)), but pitch already sign-flipped above.
    R = (_rot_y(yaw) @ _rot_x(-pitch)).astype(np.float32)

    n = max(12, int(samples_per_edge))
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, n, dtype=np.float32)

    top    = np.stack([xs, np.full(n,  1.0, np.float32)], axis=1)
    right  = np.stack([np.full(n, 1.0, np.float32), ys[::-1]], axis=1)
    bottom = np.stack([xs[::-1], np.full(n, -1.0, np.float32)], axis=1)
    left   = np.stack([np.full(n, -1.0, np.float32), ys], axis=1)
    ndc = np.concatenate([top, right, bottom, left], axis=0)

    rays = np.stack([ndc[:, 0] * tan_half_h,
                     ndc[:, 1] * tan_half_v,
                     np.ones(ndc.shape[0], dtype=np.float32)], axis=1)
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-8)

    d = (R @ rays.T).T
    lon = np.arctan2(d[:, 0], d[:, 2])
    lat = np.arcsin(np.clip(d[:, 1], -1.0, 1.0))

    u = (lon / (2.0 * math.pi)) + 0.5
    v = 0.5 - (lat / math.pi)

    u = u.astype(np.float32)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    return u, v


def _unwrap_u(u: np.ndarray) -> np.ndarray:
    uu = u.astype(np.float32).copy()
    for i in range(1, len(uu)):
        du = uu[i] - uu[i - 1]
        if du > 0.5:
            uu[i:] -= 1.0
        elif du < -0.5:
            uu[i:] += 1.0
    return uu


# =========================
# Main GUI class
# =========================
class ViewerGui:
    def __init__(self, state: SharedState) -> None:
        self.state = state

        self.pano_tex = GLTexture()
        self.view_tex: Dict[str, GLTexture] = {}

        self._last_uploaded_pano_id = -1
        self._last_uploaded_views_id = -1

        # --- PTZ ---
        self.ptz = PTZRenderer()
        self.ptz_state = PTZState(yaw_deg=0.0, pitch_deg=0.0, hfov_deg=90.0)
        self._ptz_dirty = True
        self._ptz_last_input_frame = -1
        self._ptz_out_tex_id = 0
        self._ptz_render_scale = 1.0  # affects FBO res only

    def _imgui_image(self, tex_id: int, disp_w: int, disp_h: int, *, flip_v: bool = False) -> None:
        tex_ref = imgui.ImTextureRef(int(tex_id))
        if flip_v:
            imgui.image(tex_ref, (disp_w, disp_h), uv0=(0, 1), uv1=(1, 0))
        else:
            imgui.image(tex_ref, (disp_w, disp_h), uv0=(0, 0), uv1=(1, 1))

    # ----- PTZ controls -----
    def _recenter_ptz_to_mouse(self, disp_w: int, disp_h: int) -> None:
        """
        Double-click: clicked point becomes PTZ center (yaw/pitch), zoom unchanged.
        Reset: returns to default view.
        """
        mp = imgui.get_mouse_pos()
        rmin = imgui.get_item_rect_min()

        rx = float(mp.x - rmin.x)
        ry = float(mp.y - rmin.y)

        u = max(0.0, min(1.0, rx / max(1.0, float(disp_w))))
        v = max(0.0, min(1.0, ry / max(1.0, float(disp_h))))

        ndc_x = 2.0 * u - 1.0
        ndc_y = 1.0 - 2.0 * v

        hfov = math.radians(self.ptz_state.hfov_deg)
        aspect = float(disp_w) / max(1.0, float(disp_h))
        tan_half_h = math.tan(hfov * 0.5)
        tan_half_v = tan_half_h / max(1e-6, aspect)

        ray = np.array([ndc_x * tan_half_h, ndc_y * tan_half_v, 1.0], dtype=np.float32)
        ray /= max(1e-8, float(np.linalg.norm(ray)))

        yaw = math.radians(self.ptz_state.yaw_deg)
        pitch = math.radians(self.ptz_state.pitch_deg)
        R = (_rot_y(yaw) @ _rot_x(-pitch)).astype(np.float32)
        d = R @ ray

        lon = math.atan2(float(d[0]), float(d[2]))
        lat = math.asin(max(-1.0, min(1.0, float(d[1]))))

        self.ptz_state.yaw_deg = float(math.degrees(lon))
        self.ptz_state.pitch_deg = float(math.degrees(lat))
        self.ptz_state.yaw_deg = float((self.ptz_state.yaw_deg + 180.0) % 360.0 - 180.0)
        self.ptz_state.pitch_deg = float(max(-89.0, min(89.0, self.ptz_state.pitch_deg)))
        self._ptz_dirty = True

    def _handle_ptz_interaction(self, disp_w: int, disp_h: int) -> None:
        io = imgui.get_io()
        if not imgui.is_item_hovered():
            return

        # Consume wheel to prevent vertical scroll while hovering PTZ image
        if io.mouse_wheel != 0.0:
            step = 8.0 if io.key_shift else 4.0
            self.ptz_state.hfov_deg -= io.mouse_wheel * step
            self.ptz_state.hfov_deg = float(max(20.0, min(120.0, self.ptz_state.hfov_deg)))
            self._ptz_dirty = True
            io.mouse_wheel = 0.0

        if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
            self._recenter_ptz_to_mouse(disp_w, disp_h)
            return

        if io.mouse_down[0]:
            dx, dy = float(io.mouse_delta.x), float(io.mouse_delta.y)
            if dx != 0.0 or dy != 0.0:
                hfov_deg = float(self.ptz_state.hfov_deg)
                aspect = float(disp_w) / max(1.0, float(disp_h))
                vfov_deg = math.degrees(
                    2.0 * math.atan(math.tan(math.radians(hfov_deg) * 0.5) / max(1e-6, aspect))
                )

                self.ptz_state.yaw_deg += (dx / max(1.0, disp_w)) * hfov_deg
                self.ptz_state.pitch_deg += (-dy / max(1.0, disp_h)) * vfov_deg

                self.ptz_state.yaw_deg = float((self.ptz_state.yaw_deg + 180.0) % 360.0 - 180.0)
                self.ptz_state.pitch_deg = float(max(-89.0, min(89.0, self.ptz_state.pitch_deg)))
                self._ptz_dirty = True

    # =========================
    # Dockable window GUIs
    # =========================
    def pano_window_gui(self) -> None:
        snap = self.state.ui_snapshot()
        previews = self.state.get_latest_previews()

        imgui.text(f"Status: {snap['status']}")
        imgui.same_line()
        imgui.text(f"Uptime: {snap['uptime_s']:.1f}s")

        imgui.text(f"Capture FPS(est): {snap['capture_fps']:.1f} | Infer FPS(est): {snap['infer_fps']:.1f}")
        if snap["infer_ms"] is not None:
            imgui.text(f"Infer: {snap['infer_ms']:.1f} ms | E2E: {snap['e2e_ms']:.1f} ms")

        if snap["projector_build_ms"] is not None:
            imgui.text(f"Projector build: {snap['projector_build_ms']:.1f} ms")
        imgui.text(f"Model loaded: {snap['model_loaded']}")

        imgui.separator()

        if imgui.button("Resume" if snap["paused"] else "Pause"):
            self.state.toggle_paused()

        changed, new_conf = imgui.slider_float("Conf", float(snap["conf"]), 0.05, 0.95)
        if changed:
            self.state.set_conf(float(new_conf))

        changed, new_imgsz = imgui.slider_int("imgsz", int(snap["imgsz"]), 320, 960)
        if changed:
            new_imgsz = max(32, int(round(new_imgsz / 32) * 32))
            self.state.set_imgsz(new_imgsz)

        imgui.separator()

        if previews.pano_bgr_small is None:
            imgui.text_disabled("Waiting for panorama preview...")
            return

        if previews.frame_id != self._last_uploaded_pano_id:
            self.pano_tex.upload_bgr(previews.pano_bgr_small)
            self._last_uploaded_pano_id = previews.frame_id

        avail_w = max(200.0, float(imgui.get_content_region_avail().x))
        h, w = previews.pano_bgr_small.shape[:2]
        disp_w = int(avail_w)
        disp_h = int(disp_w * (h / w))

        imgui.text("Panorama")
        self._imgui_image(self.pano_tex.tex_id, disp_w, disp_h, flip_v=False)

    def ptz_window_gui(self) -> None:
        previews = self.state.get_latest_previews()

        if previews.pano_bgr_small is None:
            imgui.text_disabled("Waiting for panorama preview...")
            return

        if previews.frame_id != self._last_uploaded_pano_id:
            self.pano_tex.upload_bgr(previews.pano_bgr_small)
            self._last_uploaded_pano_id = previews.frame_id

        if self.pano_tex.tex_id == 0 or self.pano_tex.w <= 0 or self.pano_tex.h <= 0:
            imgui.text_disabled("PTZ: pano texture not ready yet")
            return

        imgui.text("PTZ: drag LMB pan/tilt • wheel zoom • double-click recenter")
        imgui.text_disabled("Double-click recenters at cursor (keeps zoom). Reset returns to default.")

        if imgui.button("Reset PTZ"):
            self.ptz_state = PTZState(yaw_deg=0.0, pitch_deg=0.0, hfov_deg=90.0)
            self._ptz_dirty = True

        # ---- sliders on ONE line with equal widths ----
        spacing = float(imgui.get_style().item_spacing.x)
        avail_x = float(imgui.get_content_region_avail().x - 30 * spacing)

        item_w = max(60.0, (avail_x - 3 * 3 * spacing) / 4.0)

        imgui.push_item_width(item_w)

        ch, v = imgui.slider_float("Yaw##ptz_yaw", float(self.ptz_state.yaw_deg), -180.0, 180.0)
        if ch:
            self.ptz_state.yaw_deg = float(v)
            self._ptz_dirty = True

        imgui.same_line(0.0, 3 * spacing)
        ch, v = imgui.slider_float("Pitch##ptz_pitch", float(self.ptz_state.pitch_deg), -89.0, 89.0)
        if ch:
            self.ptz_state.pitch_deg = float(v)
            self._ptz_dirty = True

        imgui.same_line(0.0, 3 * spacing)
        ch, v = imgui.slider_float("HFOV##ptz_hfov", float(self.ptz_state.hfov_deg), 20.0, 120.0)
        if ch:
            self.ptz_state.hfov_deg = float(v)
            self._ptz_dirty = True

        imgui.same_line(0.0, 3 * spacing)
        ch, v = imgui.slider_float("Scale##ptz_scale", float(self._ptz_render_scale), 0.25, 1.0)
        if ch:
            self._ptz_render_scale = float(v)
            self._ptz_dirty = True

        imgui.pop_item_width()

        # rerender if input pano frame changes
        if previews.frame_id != self._ptz_last_input_frame:
            self._ptz_last_input_frame = previews.frame_id
            self._ptz_dirty = True

        out_w = max(1, int(round(self.pano_tex.w * self._ptz_render_scale)))
        out_h = max(1, int(round(self.pano_tex.h * self._ptz_render_scale)))

        if self._ptz_dirty:
            self._ptz_out_tex_id = self.ptz.render(self.pano_tex.tex_id, self.ptz_state, (out_w, out_h))
            self._ptz_dirty = False

        # ---- FIX #3: fit PTZ image to BOTH available width and height ----
        # no scrolling => image must be fully visible in remaining region.
        avail = imgui.get_content_region_avail()
        max_w = max(200.0, float(avail.x))
        max_h = max(120.0, float(avail.y))

        pano_aspect = self.pano_tex.w / max(1.0, float(self.pano_tex.h))
        # choose the largest that fits (w <= max_w, h <= max_h)
        disp_w = min(max_w, max_h * pano_aspect)
        disp_h = disp_w / max(1e-6, pano_aspect)

        disp_w_i = int(disp_w)
        disp_h_i = int(disp_h)

        # PTZ upside down fix: flip only PTZ output
        self._imgui_image(self._ptz_out_tex_id, disp_w_i, disp_h_i, flip_v=True)

        # mouse interaction targets the PTZ image item (wheel consumed)
        self._handle_ptz_interaction(disp_w_i, disp_h_i)

        # thumbnail overlay on top of PTZ image
        self._draw_pano_thumbnail_with_roi_poly(disp_w_i, disp_h_i)

    def _draw_pano_thumbnail_with_roi_poly(self, disp_w: int, disp_h: int) -> None:
        if self.pano_tex.tex_id == 0:
            return

        img_min = imgui.get_item_rect_min()
        img_max = imgui.get_item_rect_max()
        img_w = float(img_max.x - img_min.x)
        img_h = float(img_max.y - img_min.y)
        if img_w <= 2 or img_h <= 2:
            return

        margin = 10.0
        max_size = 170.0

        pano_aspect = self.pano_tex.w / max(1.0, float(self.pano_tex.h))
        if pano_aspect >= 1.0:
            thumb_w = max_size
            thumb_h = max_size / pano_aspect
        else:
            thumb_h = max_size
            thumb_w = max_size * pano_aspect

        max_w = max(30.0, img_w - 2 * margin)
        max_h = max(30.0, img_h - 2 * margin)
        s = min(max_w / thumb_w, max_h / thumb_h, 1.0)
        thumb_w *= s
        thumb_h *= s

        x0 = float(img_max.x - margin - thumb_w)
        y0 = float(img_max.y - margin - thumb_h)
        x1 = x0 + thumb_w
        y1 = y0 + thumb_h

        dl = imgui.get_window_draw_list()
        white = imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))

        # pano thumbnail
        tex = imgui.ImTextureRef(int(self.pano_tex.tex_id))
        dl.add_image_quad(
            tex,
            imgui.ImVec2(x0, y0),
            imgui.ImVec2(x0, y1),
            imgui.ImVec2(x1, y1),
            imgui.ImVec2(x1, y0),
            imgui.ImVec2(0.0, 0.0),
            imgui.ImVec2(0.0, 1.0),
            imgui.ImVec2(1.0, 1.0),
            imgui.ImVec2(1.0, 0.0),
            white,
        )
        dl.add_quad(
            imgui.ImVec2(x0, y0),
            imgui.ImVec2(x0, y1),
            imgui.ImVec2(x1, y1),
            imgui.ImVec2(x1, y0),
            white,
            2.0,
        )

        # ROI poly from current PTZ (use displayed aspect)
        ptz_aspect = float(disp_w) / max(1.0, float(disp_h))
        u, v = _frustum_outline_uv_for_thumbnail(self.ptz_state, ptz_aspect, samples_per_edge=32)
        uu = _unwrap_u(u)

        def to_xy(ui: float, vi: float) -> tuple[float, float]:
            return x0 + float((ui % 1.0)) * thumb_w, y0 + float(vi) * thumb_h

        # draw poly segments; skip seam-crossing lines
        for i in range(1, len(uu)):
            if int(math.floor(float(uu[i - 1]))) != int(math.floor(float(uu[i]))):
                continue
            xA, yA = to_xy(float(uu[i - 1]), float(v[i - 1]))
            xB, yB = to_xy(float(uu[i]), float(v[i]))
            dl.add_line(imgui.ImVec2(xA, yA), imgui.ImVec2(xB, yB), white, 2.0)

        if int(math.floor(float(uu[-1]))) == int(math.floor(float(uu[0]))):
            xA, yA = to_xy(float(uu[-1]), float(v[-1]))
            xB, yB = to_xy(float(uu[0]), float(v[0]))
            dl.add_line(imgui.ImVec2(xA, yA), imgui.ImVec2(xB, yB), white, 2.0)

    def views_window_gui(self) -> None:
        previews = self.state.get_latest_previews()
        if not previews.views_bgr_small:
            imgui.text_disabled("Waiting for view previews...")
            return

        if previews.frame_id != self._last_uploaded_views_id:
            for name, img in previews.views_bgr_small.items():
                if name not in self.view_tex:
                    self.view_tex[name] = GLTexture()
                self.view_tex[name].upload_bgr(img)
            self._last_uploaded_views_id = previews.frame_id

        names = list(previews.views_bgr_small.keys())
        n = len(names)
        if n == 0:
            imgui.text_disabled("No views yet...")
            return

        if n <= 4:
            cols = 2
        elif n <= 6:
            cols = 3
        else:
            cols = 4

        spacing = imgui.get_style().item_spacing.x
        avail_w = max(200.0, float(imgui.get_content_region_avail().x))
        col_w = (avail_w - spacing * (cols - 1)) / cols

        imgui.text(f"Views ({n})")
        for i, name in enumerate(names):
            if i % cols != 0:
                imgui.same_line()

            tex = self.view_tex[name]
            img = previews.views_bgr_small[name]
            h, w = img.shape[:2]

            disp_w = int(col_w)
            disp_h = int(disp_w * (h / w))

            imgui.begin_group()
            imgui.text(name)
            self._imgui_image(tex.tex_id, disp_w, disp_h, flip_v=False)
            imgui.end_group()

    def before_exit(self) -> None:
        self.pano_tex.destroy()
        for t in self.view_tex.values():
            t.destroy()
        self.ptz.destroy()


# =========================
# Docking layout
# =========================
def _create_default_docking_splits() -> List[hello_imgui.DockingSplit]:
    split = hello_imgui.DockingSplit()
    split.initial_dock = "MainDockSpace"
    split.new_dock = "BottomDockSpace"
    split.direction = imgui.Dir.down
    split.ratio = 0.40
    return [split]


def _create_dockable_windows(gui: ViewerGui) -> List[hello_imgui.DockableWindow]:
    wins: List[hello_imgui.DockableWindow] = []

    pano = hello_imgui.DockableWindow()
    pano.label = "Panorama"
    pano.dock_space_name = "MainDockSpace"
    pano.gui_function = gui.pano_window_gui
    pano.include_in_view_menu = True
    pano.remember_is_visible = True
    wins.append(pano)

    ptz = hello_imgui.DockableWindow()
    ptz.label = "PTZ"
    ptz.dock_space_name = "MainDockSpace"
    ptz.gui_function = gui.ptz_window_gui
    ptz.include_in_view_menu = True
    ptz.remember_is_visible = True
    ptz.imgui_window_flags = imgui.WindowFlags_.no_scroll_with_mouse | imgui.WindowFlags_.no_scrollbar
    wins.append(ptz)

    views = hello_imgui.DockableWindow()
    views.label = "Views"
    views.dock_space_name = "BottomDockSpace"
    views.gui_function = gui.views_window_gui
    views.include_in_view_menu = True
    views.remember_is_visible = True
    wins.append(views)

    return wins


def run_gui(state: SharedState) -> None:
    gui = ViewerGui(state)

    runner = hello_imgui.RunnerParams()
    runner.app_window_params.window_title = "360 YOLO Viewer"
    runner.app_window_params.window_geometry.size = (1280, 900)

    runner.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    runner.imgui_window_params.enable_viewports = False

    runner.imgui_window_params.show_menu_bar = True
    runner.imgui_window_params.show_menu_view = True
    # optional: keep app menu hidden if you only want View
    runner.imgui_window_params.show_menu_app = False

    runner.fps_idling.enable_idling = False

    docking = hello_imgui.DockingParams()
    docking.docking_splits = _create_default_docking_splits()
    docking.dockable_windows = _create_dockable_windows(gui)
    docking.layout_condition = hello_imgui.DockingLayoutCondition.application_start
    runner.docking_params = docking

    runner.callbacks.before_exit = gui.before_exit

    hello_imgui.run(runner)
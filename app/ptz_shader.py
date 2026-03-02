from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from OpenGL.GL import (
    glActiveTexture,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glCheckFramebufferStatus,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnableVertexAttribArray,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
    glDisable,
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COMPILE_STATUS,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_RGB,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
)
import ctypes


VERT_SRC = r"""
#version 330 core
layout (location = 0) in vec2 aPos;   // -1..1
layout (location = 1) in vec2 aUv;    // 0..1
out vec2 vUv;
void main() {
    vUv = aUv;
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
}
"""

FRAG_SRC = r"""
#version 330 core
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uPano;
uniform float uYaw;      // radians
uniform float uPitch;    // radians (positive = look up)
uniform float uHfov;     // radians
uniform vec2  uOutSize;  // (w, h)

const float PI = 3.14159265358979323846;

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
        1.0, 0.0, 0.0,
        0.0,  c, -s,
        0.0,  s,  c
    );
}
mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
         c, 0.0,  s,
        0.0, 1.0, 0.0,
        -s, 0.0,  c
    );
}

void main() {
    // NDC in [-1,1]
    vec2 ndc = vUv * 2.0 - 1.0;

    float aspect = uOutSize.x / uOutSize.y;

    float tanHalfH = tan(uHfov * 0.5);
    float tanHalfV = tanHalfH / aspect;

    // camera ray: x right, y up, z forward
    vec3 dir = normalize(vec3(ndc.x * tanHalfH, ndc.y * tanHalfV, 1.0));

    // Positive pitch should look UP -> apply rotX(-uPitch)
    dir = rotY(uYaw) * rotX(-uPitch) * dir;

    float lon = atan(dir.x, dir.z);             // [-pi, pi]
    float lat = asin(clamp(dir.y, -1.0, 1.0));  // [-pi/2, pi/2]

    float u = lon / (2.0 * PI) + 0.5;  // wrap
    float v = 0.5 - lat / PI;          // clamp

    u = fract(u);
    v = clamp(v, 0.0, 1.0);

    vec3 rgb = texture(uPano, vec2(u, v)).rgb;
    FragColor = vec4(rgb, 1.0);
}
"""


def _compile_shader(src: str, shader_type: int) -> int:
    sh = glCreateShader(shader_type)
    glShaderSource(sh, src)
    glCompileShader(sh)
    ok = glGetShaderiv(sh, GL_COMPILE_STATUS)
    if not ok:
        info = glGetShaderInfoLog(sh).decode(errors="ignore")
        raise RuntimeError(f"Shader compile failed:\n{info}")
    return sh


def _link_program(vs: int, fs: int) -> int:
    from OpenGL.GL import glAttachShader  # provided by PyOpenGL

    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    ok = glGetProgramiv(prog, GL_LINK_STATUS)
    if not ok:
        info = glGetProgramInfoLog(prog).decode(errors="ignore")
        raise RuntimeError(f"Program link failed:\n{info}")
    return prog


@dataclass
class PTZState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    hfov_deg: float = 90.0


class PTZRenderer:
    """
    Renders a PTZ view of an equirectangular pano texture into an FBO texture.
    Output texture is GL_RGB and can be shown via imgui.image(ImTextureRef(tex_id)).
    """
    def __init__(self) -> None:
        self._init = False
        self.program = 0
        self.vao = 0
        self.vbo = 0

        self.fbo = 0
        self.out_tex = 0
        self.out_w = 0
        self.out_h = 0

        # uniforms
        self.uPano = -1
        self.uYaw = -1
        self.uPitch = -1
        self.uHfov = -1
        self.uOutSize = -1

    def ensure_initialized(self) -> None:
        if self._init:
            return

        vs = _compile_shader(VERT_SRC, GL_VERTEX_SHADER)
        fs = _compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)

        self.uPano = glGetUniformLocation(self.program, "uPano")
        self.uYaw = glGetUniformLocation(self.program, "uYaw")
        self.uPitch = glGetUniformLocation(self.program, "uPitch")
        self.uHfov = glGetUniformLocation(self.program, "uHfov")
        self.uOutSize = glGetUniformLocation(self.program, "uOutSize")

        # Full-screen quad (triangle strip)
        quad = np.array(
            [
                # x, y,   u, v
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ],
            dtype=np.float32
        )

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats per vertex

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._init = True

    def ensure_fbo(self, w: int, h: int) -> None:
        self.ensure_initialized()
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            return
        if self.out_tex != 0 and self.fbo != 0 and w == self.out_w and h == self.out_h:
            return

        self.out_w, self.out_h = w, h

        if self.out_tex == 0:
            self.out_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.out_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        if self.fbo == 0:
            self.fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.out_tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO not complete (status={status})")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render(self, pano_tex_id: int, state: PTZState, out_size: Tuple[int, int]) -> int:
        """
        Returns the output texture id (GL_TEXTURE_2D) containing the PTZ view.
        """
        self.ensure_fbo(out_size[0], out_size[1])

        glDisable(GL_DEPTH_TEST)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.out_w, self.out_h)

        glUseProgram(self.program)
        glUniform1i(self.uPano, 0)
        glUniform1f(self.uYaw, math.radians(state.yaw_deg))
        glUniform1f(self.uPitch, math.radians(state.pitch_deg))
        glUniform1f(self.uHfov, math.radians(state.hfov_deg))
        glUniform2f(self.uOutSize, float(self.out_w), float(self.out_h))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, int(pano_tex_id))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return self.out_tex

    def destroy(self) -> None:
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = 0
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        if self.program:
            glDeleteProgram(self.program)
            self.program = 0
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = 0
        if self.out_tex:
            glDeleteTextures([self.out_tex])
            self.out_tex = 0
        self._init = False
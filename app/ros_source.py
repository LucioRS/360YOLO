from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    enc = str(msg.encoding).lower()

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")

    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8",):
        expected_row = w * 3
        arr = buf.reshape((h, step))[:, :expected_row]
        return np.ascontiguousarray(arr.reshape((h, w, 3)))

    if enc in ("rgb8",):
        expected_row = w * 3
        arr = buf.reshape((h, step))[:, :expected_row]
        rgb = np.ascontiguousarray(arr.reshape((h, w, 3)))
        return rgb[:, :, ::-1]

    if enc in ("bgra8",):
        expected_row = w * 4
        arr = buf.reshape((h, step))[:, :expected_row]
        bgra = np.ascontiguousarray(arr.reshape((h, w, 4)))
        return bgra[:, :, :3]

    if enc in ("rgba8",):
        expected_row = w * 4
        arr = buf.reshape((h, step))[:, :expected_row]
        rgba = np.ascontiguousarray(arr.reshape((h, w, 4)))
        rgb = rgba[:, :, :3]
        return rgb[:, :, ::-1]

    if enc in ("mono8", "8uc1"):
        expected_row = w
        arr = buf.reshape((h, step))[:, :expected_row]
        gray = np.ascontiguousarray(arr.reshape((h, w)))
        return np.repeat(gray[:, :, None], 3, axis=2)

    raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")


class _ROSImageSubscriber(Node):
    def __init__(
        self,
        *,
        node_name: str,
        topic: str,
        queue_size: int,
        on_image,
        on_error,
    ):
        super().__init__(node_name)
        self._on_image = on_image
        self._on_error = on_error

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(queue_size)),
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._sub = self.create_subscription(
            Image,
            topic,
            self._image_callback,
            qos,
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = ros_image_to_bgr(msg)
            self._on_image(frame)
        except Exception as e:
            print(f"[ROS callback] conversion failed: {e}", flush=True)
            self._on_error(f"ROS image conversion failed: {e}")


class ROSImageSource:
    def __init__(
        self,
        *,
        topic: str,
        node_name: str = "yolo360_ros_sub",
        queue_size: int = 1,
        wait_timeout_sec: float = 1.0,
    ):
        self.topic = topic
        self.node_name = node_name
        self.queue_size = int(queue_size)
        self.wait_timeout_sec = float(wait_timeout_sec)

        self._node: Optional[_ROSImageSubscriber] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._owns_rclpy = False

        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._last_read_seq = 0
        self._last_error: Optional[str] = None
        self._closed = False

    def _on_image(self, frame: np.ndarray) -> None:
        with self._cond:
            self._latest = frame
            self._last_error = None
            self._seq += 1
            self._cond.notify_all()

    def _on_error(self, error_msg: str) -> None:
        with self._cond:
            self._last_error = error_msg
            self._cond.notify_all()

    def get_last_error(self) -> Optional[str]:
        with self._cond:
            return self._last_error

    def open(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self._node = _ROSImageSubscriber(
            node_name=self.node_name,
            topic=self.topic,
            queue_size=self.queue_size,
            on_image=self._on_image,
            on_error=self._on_error,
        )

        self._spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self._node,),
            name="ROSImageSpin",
            daemon=True,
        )
        self._spin_thread.start()

    def read_frame(self) -> Optional[np.ndarray]:
        deadline = time.monotonic() + self.wait_timeout_sec

        with self._cond:
            start_seq = self._last_read_seq

            while not self._closed and self._seq == start_seq and self._last_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(timeout=remaining)

            if self._last_error is not None:
                return None

            if self._latest is None:
                return None

            self._last_read_seq = self._seq
            return self._latest.copy()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass

        try:
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

        self._node = None
        self._spin_thread = None
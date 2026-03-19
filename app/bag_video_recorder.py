from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


class PanoBagRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._uri: Optional[str] = None
        self._topic_name = "/panorama/annotated"
        self._frame_id = "panorama"
        self._writer = None
        self._thread: Optional[threading.Thread] = None
        self._q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=16)
        self._stop_evt = threading.Event()
        self._frames_written = 0
        self._last_error: Optional[str] = None

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active": self._active,
                "uri": self._uri,
                "frames_written": self._frames_written,
                "last_error": self._last_error,
            }

    def start(
        self,
        output_uri: str,
        *,
        topic_name: str = "/panorama/annotated",
        frame_id: str = "panorama",
    ) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("Recorder is already active")

            import rosbag2_py
            from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

            output_uri = str(Path(output_uri))
            self._topic_name = topic_name
            self._frame_id = frame_id
            self._uri = output_uri
            self._frames_written = 0
            self._last_error = None
            self._stop_evt.clear()

            writer = SequentialWriter()
            writer.open(
                StorageOptions(uri=output_uri, storage_id="mcap"),
                ConverterOptions("", ""),
            )
            writer.create_topic(
                TopicMetadata(
                    id=0,
                    name=topic_name,
                    type="sensor_msgs/msg/Image",
                    serialization_format="cdr",
                )
            )

            self._writer = writer
            self._active = True
            self._thread = threading.Thread(target=self._worker_loop, name="PanoBagRecorder", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._stop_evt.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        with self._lock:
            self._thread = None
            self._writer = None
            self._active = False

        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def enqueue_frame(self, bgr_image: np.ndarray, stamp_ns: Optional[int] = None) -> None:
        with self._lock:
            if not self._active:
                return

        if stamp_ns is None:
            stamp_ns = time.time_ns()

        img = np.ascontiguousarray(bgr_image, dtype=np.uint8)

        try:
            self._q.put_nowait((stamp_ns, img))
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((stamp_ns, img))
            except queue.Full:
                pass

    def _worker_loop(self) -> None:
        try:
            from rclpy.serialization import serialize_message
            from sensor_msgs.msg import Image

            while not self._stop_evt.is_set() or not self._q.empty():
                try:
                    stamp_ns, img = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue

                h, w = img.shape[:2]

                msg = Image()
                msg.header.stamp.sec = stamp_ns // 1_000_000_000
                msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
                msg.header.frame_id = self._frame_id
                msg.height = h
                msg.width = w
                msg.encoding = "bgr8"
                msg.is_bigendian = (sys.byteorder == "big")
                msg.step = w * 3
                msg.data = img.tobytes()

                with self._lock:
                    writer = self._writer
                    topic_name = self._topic_name

                writer.write(topic_name, serialize_message(msg), stamp_ns)

                with self._lock:
                    self._frames_written += 1

        except Exception as e:
            with self._lock:
                self._last_error = str(e)
                self._active = False
                self._writer = None
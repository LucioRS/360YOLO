from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Det:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


class YOLODetectorCPU:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def infer_batch(self, views: Dict[str, np.ndarray], conf: float, imgsz: int) -> Dict[str, List[Det]]:
        names = list(views.keys())
        imgs = [views[k] for k in names]

        results = self.model(imgs, device="cpu", conf=conf, imgsz=imgsz, verbose=False)
        out: Dict[str, List[Det]] = {}

        for name, r in zip(names, results):
            boxes = getattr(r, "boxes", None)
            names_map = getattr(r, "names", None) or getattr(self.model, "names", {})
            dets: List[Det] = []
            if boxes is not None:
                for b in boxes:
                    cls_id = int(b.cls[0]) if b.cls is not None else -1
                    conf_v = float(b.conf[0]) if b.conf is not None else 0.0
                    xyxy = tuple(float(x) for x in b.xyxy[0].tolist())
                    cls_name = str(names_map.get(cls_id, cls_id))
                    dets.append(Det(cls_id, cls_name, conf_v, xyxy))
            out[name] = dets
        return out
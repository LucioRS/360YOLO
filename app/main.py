from __future__ import annotations

from config import AppConfig
from shared_state import SharedState
from projector import EquirectProjector
from detector import YOLODetectorCPU
from workers import CaptureWorker, InferenceWorker
from gui import run_gui


def main() -> int:
    cfg = AppConfig()
    state = SharedState()

    # Precompute remap grids once (startup)
    projector = EquirectProjector(
        pano_w=cfg.camera.width,
        pano_h=cfg.camera.height,
        views=cfg.projection.views,
    )
    state.set_projector_build_ms(projector.build_time_ms)

    detector = YOLODetectorCPU(cfg.inference.model_path)
    state.set_model_loaded(True)

    state.set_conf(cfg.inference.conf)
    state.set_imgsz(cfg.inference.imgsz)
    state.set_paused(False)


    cap_w = CaptureWorker(cfg, state)
    inf_w = InferenceWorker(cfg, state, projector, detector)

    cap_w.start()
    inf_w.start()

    try:
        run_gui(state)
    finally:
        cap_w.stop()
        inf_w.stop()
        cap_w.join(timeout=2.0)
        inf_w.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
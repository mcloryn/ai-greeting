"""
scripts/check_color.py

Skrip pengujian manual Checkpoint 06. Menggabungkan seluruh pipeline
vision (Camera -> Detector -> Tracker -> TargetSelector -> TargetLock)
dan menambahkan deteksi + stabilisasi warna pakaian untuk target yang
terkunci. Menampilkan kotak ROI torso dan nama warna di layar.

Jalankan:
    python scripts/check_color.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config.settings import ConfigError, load_settings
from src.core.exceptions import CameraError, DetectorError
from src.core.logger import get_logger
from src.vision.camera import Camera
from src.vision.clothing_color import detect_clothing_color, extract_torso_roi
from src.vision.color_stabilizer import ColorStabilizer
from src.vision.detector import PersonDetector
from src.vision.target_lock import TargetLock
from src.vision.target_selector import score_candidates, select_target
from src.vision.tracker import Tracker
from src.vision.visualizer import draw_lock_status, draw_tracked_people

logger = get_logger(__name__)


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[KONFIGURASI SALAH] {exc}", file=sys.stderr)
        return 1

    detector = PersonDetector(settings.vision)
    try:
        print("Memuat model YOLO...")
        detector.load_model()
        detector.warmup()
    except DetectorError as exc:
        print(f"[ERROR DETECTOR] {exc}", file=sys.stderr)
        return 1

    tracker = Tracker()
    target_lock = TargetLock(settings.vision)
    color_stabilizer = ColorStabilizer(settings.vision.clothing_color)
    last_locked_track_id: int | None = None

    try:
        with Camera(settings.vision.camera) as cam:
            print("Semua komponen siap. Tekan 'q' untuk keluar.")
            while True:
                frame = cam.read()
                if frame is None:
                    print("Frame tidak terbaca. Berhenti.")
                    break

                now = time.time()
                detections = detector.detect_and_track(frame)
                tracked_people = tracker.update(detections)
                candidates = score_candidates(
                    tracked_people, frame.width, frame.height, settings.vision
                )
                target_id = select_target(candidates)
                lock_state = target_lock.update(target_id, tracked_people, now)

                overlay = draw_tracked_people(frame.image, tracked_people, candidates, target_id)
                overlay = draw_lock_status(overlay, lock_state)

                # Reset buffer warna setiap kali target berganti (Task 10)
                if lock_state.track_id != last_locked_track_id:
                    color_stabilizer.reset()
                    last_locked_track_id = lock_state.track_id

                color_text = "Warna: -"
                if lock_state.state.value == "LOCKED" and lock_state.bbox is not None:
                    result = detect_clothing_color(frame, lock_state.bbox, settings.vision.clothing_color)
                    color_stabilizer.add(result.color_name, now)
                    stable = color_stabilizer.get_stable_color()

                    color_text = f"Warna: {stable.color_name.value if stable.color_name else '...'}" \
                                 f" ({stable.ratio:.0%})"

                    # Gambar kotak ROI torso untuk verifikasi visual (Task 11)
                    roi_y1 = lock_state.bbox.y1 + int(lock_state.bbox.height * settings.vision.clothing_color.roi_top_ratio)
                    roi_y2 = lock_state.bbox.y1 + int(lock_state.bbox.height * settings.vision.clothing_color.roi_bottom_ratio)
                    roi_x1 = lock_state.bbox.x1 + int(lock_state.bbox.width * settings.vision.clothing_color.roi_left_ratio)
                    roi_x2 = lock_state.bbox.x1 + int(lock_state.bbox.width * settings.vision.clothing_color.roi_right_ratio)
                    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)

                cv2.putText(overlay, color_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 255), 2)

                fps = cam.current_fps()
                cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                cv2.imshow("AI Greeting Mikroskil - Check Color (CP06)", overlay)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except CameraError as exc:
        print(f"[ERROR KAMERA] {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
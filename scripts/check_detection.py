"""
scripts/check_detection.py

Skrip pengujian manual Checkpoint 04. Menggabungkan Camera + PersonDetector
(dengan tracking bawaan Ultralytics) + Tracker + TargetSelector +
Visualizer: menampilkan video live dengan kotak di sekeliling setiap
orang, track_id + skor masing-masing, dan satu orang ditandai sebagai
target terpilih (kotak kuning).

Jalankan:
    python scripts/check_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import time

from config.settings import ConfigError, load_settings
from src.core.exceptions import CameraError, DetectorError
from src.core.logger import get_logger
from src.vision.camera import Camera
from src.vision.detector import PersonDetector
from src.vision.target_lock import TargetLock
from src.vision.target_selector import score_candidates, select_target
from src.vision.tracker import Tracker
from src.vision.visualizer import draw_tracked_people, draw_lock_status
logger = get_logger(__name__)


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[KONFIGURASI SALAH] {exc}", file=sys.stderr)
        return 1

    detector = PersonDetector(settings.vision)
    try:
        print("Memuat model YOLO (unduhan pertama bisa makan waktu)...")
        detector.load_model()
        detector.warmup()
    except DetectorError as exc:
        print(f"[ERROR DETECTOR] {exc}", file=sys.stderr)
        return 1

    tracker = Tracker()
    target_lock = TargetLock(settings.vision)

    try:
        with Camera(settings.vision.camera) as cam:
            print("Kamera + detector + tracker siap. Tekan 'q' untuk keluar.")
            while True:
                frame = cam.read()
                if frame is None:
                    print("Frame tidak terbaca -- kamera mungkin terputus. Berhenti.")
                    break

                detections = detector.detect_and_track(frame)
                tracked_people = tracker.update(detections)
                candidates = score_candidates(
                    tracked_people, frame.width, frame.height, settings.vision
                )
                target_id = select_target(candidates)
                lock_state = target_lock.update(target_id, tracked_people, time.time())

                overlay = draw_tracked_people(frame.image, tracked_people, candidates, target_id)
                overlay = draw_lock_status(overlay, lock_state)

                fps = cam.current_fps()
                info_text = f"FPS: {fps:.1f}  Orang: {len(tracked_people)}  Target: {target_id}"
                cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                cv2.imshow("AI Greeting Mikroskil - Check Detection (CP05)", overlay)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Keluar diminta pengguna ('q').")
                    break
    except CameraError as exc:
        print(f"[ERROR KAMERA] {exc}", file=sys.stderr)
        logger.error("CameraError: %s", exc)
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
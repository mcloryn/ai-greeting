"""
scripts/check_camera.py

Skrip pengujian manual untuk Checkpoint 02. Membuka kamera, menampilkan
video di jendela OpenCV dengan overlay resolusi & FPS, dan keluar saat
tombol 'q' ditekan.

Jalankan:
    python scripts/check_camera.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config.settings import ConfigError, load_settings
from src.core.exceptions import CameraError
from src.core.logger import get_logger
from src.vision.camera import Camera

logger = get_logger(__name__)


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[KONFIGURASI SALAH] {exc}", file=sys.stderr)
        return 1

    try:
        with Camera(settings.vision.camera) as cam:
            print("Kamera terbuka. Tekan 'q' di jendela video untuk keluar.")
            while True:
                frame = cam.read()
                if frame is None:
                    print("Frame tidak terbaca -- kamera mungkin terputus. Berhenti.")
                    break

                fps = cam.current_fps()
                overlay = frame.image.copy()
                text = f"{frame.width}x{frame.height}  FPS: {fps:.1f}"
                cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                cv2.imshow("AI Greeting Mikroskil - Check Camera (CP02)", overlay)

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
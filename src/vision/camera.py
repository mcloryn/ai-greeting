"""
src/vision/camera.py

Pembungkus tipis di atas cv2.VideoCapture. Satu-satunya tanggung jawab
module ini: menyediakan aliran objek Frame dari webcam, dan melepaskannya
dengan bersih. Tidak ada logika deteksi, tracking, atau apa pun lain di
sini (Aturan #6 Build Plan -- vision/camera.py tidak boleh tahu soal YOLO).

Kontrak (Bagian 4.1 Build Plan):
    Input  : indeks perangkat, resolusi target, FPS target
    Output : Frame (lihat src/core/models.py)
    Public : open(), read() -> Frame | None, release(), is_opened(),
             dukungan `with`
"""

from __future__ import annotations

import time
from collections import deque
from types import TracebackType

import cv2
import numpy as np

from config.settings import CameraSettings
from src.core.exceptions import CameraError
from src.core.logger import get_logger
from src.core.models import Frame

logger = get_logger(__name__)


class Camera:
    """Membuka webcam, membaca frame, dan menghitung FPS berjalan.

    Dipakai seperti ini:

        with Camera(settings.vision.camera) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    break
                ...
    """

    def __init__(self, config: CameraSettings) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._frame_counter = 0
        # deque dengan maxlen otomatis membuang timestamp terlama -> ini
        # yang membuat rata-rata FPS "bergerak" (moving average), bukan
        # rata-rata dari seluruh riwayat sejak program mulai.
        self._frame_timestamps: deque[float] = deque(maxlen=config.fps_window)

    def open(self) -> None:
        """Buka perangkat kamera. Melempar CameraError kalau gagal."""
        self._cap = cv2.VideoCapture(self._config.device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            self._cap = None
            raise CameraError(
                f"Tidak bisa membuka kamera index {self._config.device_index}. "
                "Kemungkinan: index salah (coba 0, 1, 2, ...), kamera dipakai "
                "aplikasi lain, atau tidak ada webcam terpasang."
            )

        # Minta resolusi & FPS -- driver BOLEH mengabaikan ini diam-diam,
        # makanya nilai sebenarnya dibaca ulang di bawah dan dicatat ke log.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps_target)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps_reported = self._cap.get(cv2.CAP_PROP_FPS)

        if (actual_width, actual_height) != (self._config.requested_width,
                                              self._config.requested_height):
            logger.warning(
                "Resolusi diminta %dx%d, tapi kamera memberi %dx%d.",
                self._config.requested_width, self._config.requested_height,
                actual_width, actual_height,
            )

        logger.info(
            "Kamera index %d dibuka. Resolusi aktual=%dx%d, FPS dilaporkan driver=%.1f",
            self._config.device_index, actual_width, actual_height, actual_fps_reported,
        )

    def read(self) -> Frame | None:
        """Baca satu frame. Mengembalikan None kalau kamera terputus/gagal baca."""
        if self._cap is None:
            raise CameraError("read() dipanggil sebelum open(). Panggil open() dulu.")

        ok, image = self._cap.read()
        if not ok or image is None:
            # Ini kondisi "kabel USB lepas" -- jangan crash, cukup laporkan
            # ke pemanggil lewat None, biar pemanggil yang putuskan mau apa.
            logger.warning("Gagal membaca frame dari kamera (kemungkinan terputus).")
            return None

        now = time.time()
        self._frame_timestamps.append(now)
        self._frame_counter += 1

        height, width = image.shape[:2]
        return Frame(
            image=image,
            timestamp=now,
            frame_id=self._frame_counter,
            width=width,
            height=height,
        )

    def current_fps(self) -> float:
        """FPS rata-rata bergerak, dihitung dari timestamp N frame terakhir.

        Mengembalikan 0.0 kalau data belum cukup (baru mulai / cuma 1 frame),
        supaya tidak pernah membagi dengan nol.
        """
        return _compute_fps(list(self._frame_timestamps))

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Lepas kamera. Aman dipanggil berkali-kali (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Kamera index %d dilepas.", self._config.device_index)

    # --- dukungan context manager (`with Camera(...) as cam:`) ---

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def _compute_fps(timestamps: list[float]) -> float:
    """Fungsi murni (tidak butuh kamera) supaya bisa di-unit-test langsung.

    Dipisah dari method `current_fps()` di atas justru untuk itu: logika
    hitungnya bisa diuji dengan timestamp buatan, tanpa perlu webcam nyala.
    """
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0
    frame_count = len(timestamps) - 1
    return frame_count / elapsed
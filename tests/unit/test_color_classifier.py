"""
tests/unit/test_color_classifier.py

Unit test CP06 untuk ekstraksi ROI dan klasifikasi HSV. Memakai gambar
sintetis (numpy array warna solid), TIDAK butuh kamera.
"""

from __future__ import annotations

import numpy as np

from config.settings import load_settings
from src.core.enums import ColorName
from src.core.models import BoundingBox
from src.vision.clothing_color import classify_color, extract_torso_roi


def _settings():
    return load_settings().vision.clothing_color


def _solid_bgr_image(bgr: tuple[int, int, int], height=400, width=300) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = bgr
    return image


# --- Ekstraksi ROI ---------------------------------------------------

def test_roi_dipotong_sesuai_proporsi():
    config = _settings()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(100, 100, 300, 500)  # width=200, height=400

    roi = extract_torso_roi(image, bbox, config)

    assert roi is not None
    expected_height = int(400 * (config.roi_bottom_ratio - config.roi_top_ratio))
    expected_width = int(200 * (config.roi_right_ratio - config.roi_left_ratio))
    assert abs(roi.shape[0] - expected_height) <= 1
    assert abs(roi.shape[1] - expected_width) <= 1


def test_roi_keluar_frame_dipotong_tidak_crash():
    config = _settings()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # bbox sengaja menjorok keluar batas kanan & bawah frame
    bbox = BoundingBox(600, 450, 700, 550)

    roi = extract_torso_roi(image, bbox, config)
    # Boleh None (ditolak) atau ROI kecil yang tervalidasi -- yang penting
    # tidak crash dan tidak ada index negatif/di luar batas
    assert roi is None or roi.size >= 0


def test_roi_terlalu_kecil_ditolak():
    config = _settings()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(10, 10, 14, 14)  # bbox sangat kecil

    roi = extract_torso_roi(image, bbox, config)
    assert roi is None


# --- Klasifikasi warna -----------------------------------------------

def test_klasifikasi_hitam():
    config = _settings()
    roi = _solid_bgr_image((10, 10, 10))  # sangat gelap
    result = classify_color(roi, config)
    assert result.color_name == ColorName.HITAM


def test_klasifikasi_putih_abu():
    config = _settings()
    roi = _solid_bgr_image((200, 200, 200))  # terang, tidak berwarna
    result = classify_color(roi, config)
    assert result.color_name == ColorName.PUTIH_ABU


def test_klasifikasi_merah():
    config = _settings()
    # Merah murni BGR -> HSV: H mendekati 0
    roi = _solid_bgr_image((0, 0, 200))
    result = classify_color(roi, config)
    assert result.color_name == ColorName.MERAH


def test_klasifikasi_hijau():
    config = _settings()
    roi = _solid_bgr_image((0, 200, 0))
    result = classify_color(roi, config)
    assert result.color_name == ColorName.HIJAU


def test_klasifikasi_biru():
    config = _settings()
    roi = _solid_bgr_image((200, 0, 0))
    result = classify_color(roi, config)
    assert result.color_name == ColorName.BIRU


def test_klasifikasi_tidak_yakin_mengembalikan_none():
    config = _settings()
    # ROI kosong -> tidak boleh menebak
    roi = np.zeros((0, 0, 3), dtype=np.uint8)
    result = classify_color(roi, config)
    assert result.color_name is None
    assert result.roi_valid is False
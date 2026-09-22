"""
tests/unit/test_camera_config.py

Unit test CP02. TIDAK menyalakan kamera sungguhan -- hanya menguji:
  1. Konfigurasi kamera termuat & tervalidasi dengan benar.
  2. Logika hitung FPS (_compute_fps) benar, pakai timestamp buatan.

Ini sesuai pola test CP01 kamu: semua lolos tanpa hardware.
"""

from __future__ import annotations

from config.settings import load_settings
from src.vision.camera import _compute_fps


def test_camera_settings_termuat_dari_config_yaml():
    settings = load_settings()
    cam = settings.vision.camera

    assert cam.device_index >= 0
    assert cam.requested_width > 0
    assert cam.requested_height > 0
    assert cam.fps_target > 0
    assert cam.fps_window > 0


def test_compute_fps_dengan_kurang_dari_2_timestamp():
    assert _compute_fps([]) == 0.0
    assert _compute_fps([100.0]) == 0.0


def test_compute_fps_30_frame_dalam_1_detik():
    # 30 frame merata dalam 1 detik -> FPS harus mendekati 30
    timestamps = [i / 30 for i in range(31)]  # 0.0, 0.033, ..., 1.0
    fps = _compute_fps(timestamps)
    assert 29.0 <= fps <= 31.0


def test_compute_fps_elapsed_nol_tidak_membagi_dengan_nol():
    # Dua timestamp identik (kasus tepi yang tidak boleh crash)
    assert _compute_fps([50.0, 50.0]) == 0.0
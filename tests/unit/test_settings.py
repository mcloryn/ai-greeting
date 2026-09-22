"""
tests/unit/test_settings.py

Unit test untuk config/settings.py. Tidak butuh kamera maupun internet
(Aturan #23 & syarat tests/unit Bagian 6.1 Build Plan).
"""

from __future__ import annotations

import pytest

from config.settings import ConfigError, load_settings


def test_load_settings_berhasil_dengan_config_asli():
    """config.yaml bawaan project harus valid dan bisa dimuat tanpa error."""
    settings = load_settings()

    assert 0.0 <= settings.vision.confidence_threshold <= 1.0
    assert settings.rag.top_k > 0
    assert settings.llm.provider == "gemini"


def test_summary_tidak_mengandung_traceback():
    """summary() harus berupa teks ringkas, bukan objek mentah."""
    settings = load_settings()
    text = settings.summary()

    assert isinstance(text, str)
    assert "Traceback" not in text


def test_chunk_overlap_lebih_besar_dari_chunk_size_ditolak(tmp_path, monkeypatch):
    """Konfigurasi yang tidak masuk akal harus memicu ConfigError yang jelas,
    bukan lolos diam-diam atau menyebabkan traceback mentah.
    """
    bad_yaml = tmp_path / "config.yaml"
    bad_yaml.write_text(
        """
vision:

  camera:
    device_index: 0        # 0 = webcam default; ganti ke 1/2 kalau salah
    width: 640              # resolusi yang DIMINTA (bukan jaminan didapat)
    height: 480
    fps_target: 30           # FPS yang diminta ke driver
    fps_window: 30

  detection:
    # Model YOLO. "n" (nano) dipilih karena laptop tanpa GPU (Asumsi #1 Blueprint).
    model_name: "yolo11n-seg.pt"
    person_class_id: 0          # kelas "person" di COCO dataset
    confidence_threshold: 0.5  # Bagian 15 Build Plan

  tracker:
    # ~0.5 detik pada asumsi 30 FPS, dipakai sebagai syarat minimum umur track
    min_age_frames: 15

  target_selector:
    # Skor berbobot: gabungan luas bounding box + kedekatan ke tengah frame
    weight_area: 0.6
    weight_center: 0.4
    min_area_ratio: 0.08

  target_lock:
    confirm_timer_sec: 0.5   # lama kandidat harus stabil sebelum lock terkonfirmasi
    grace_period_sec: 2.5    # toleransi target hilang sebelum lock dilepas
    cooldown_sec: 30         # jeda sebelum orang yang sama boleh disapa lagi

  clothing_color:
    palette:
      - merah
      - oranye
      - kuning
      - hijau
      - biru
      - ungu
      - hitam
      - putih_abu
    stabilizer_buffer_sec: 0.5
    stabilizer_majority_ratio: 0.6
    color_wait_timeout_sec: 2.0

    # BARU -- pakai mask segmentasi (kalau model & Detection menyediakannya)
    # untuk ambil piksel torso, bukan cuma rectangle. Set false untuk balik
    # ke perilaku rectangle-murni tanpa ubah kode (buat isolasi A/B saat
    # debug live).
    use_mask: true

    # BARU -- proporsi ROI torso relatif terhadap bbox orang (Task 1)
    roi:
      top_ratio: 0.22
      bottom_ratio: 0.52
      left_ratio: 0.28
      right_ratio: 0.72
      min_pixel_count: 200

    # BARU -- deteksi garis bahu dari bentuk mask, menggantikan asumsi
    # rasio tetap saat mask tersedia (fallback ke roi.top_ratio dkk di
    # atas kalau garis bahu tidak terdeteksi, mis. mask terlalu tipis)
    shoulder:
      width_ratio: 0.72        # baris dianggap "bahu" kalau lebarnya >= ini * lebar_terlebar
      top_margin_ratio: 0.04   # jarak aman di bawah garis bahu (hindari kerah), relatif tinggi bbox
      torso_height_ratio: 0.32 # tinggi area torso yang diambil di bawah garis bahu, relatif tinggi bbox

      
    # BARU -- buang piksel yang terlalu gelap/terang sebelum dianalisis (Task 5)
    pixel_filter:
      v_min: 30
      v_max: 225

        # BARU -- ambang tidak-berwarna (hitam/putih-abu), dicek SEBELUM hue (Task 7)
    achromatic:
      v_black_max: 50        # V di bawah ini -> hitam
      s_gray_max: 40         # S di bawah ini (dan bukan hitam) -> putih_abu
      v_dark_noise_max: 70   # V di bawah ini hue-nya noise sensor -> dianggap gelap (hitam)

    # BARU -- ambang keyakinan (Blueprint 11.4 lapis 5)
    classification:
      min_dominance: 0.45    # warna pemenang minimal segini dari piksel valid, kalau tidak -> None

    # rentang Hue OpenCV (0-179) per warna. "merah" dibagi 2 rentang
    # karena posisinya melingkar di kedua ujung skala hue.
    hue_ranges:
      merah: [[0, 10], [170, 179]]
      oranye: [[11, 25]]
      kuning: [[26, 35]]
      hijau: [[36, 85]]
      biru: [[86, 125]]
      ungu: [[126, 155]]

      
application:
  conversation:
    idle_timeout_sec: 10.0
    max_history_turns: 6
    farewell_words:
      - "terima kasih"
      - "makasih"
      - "sudah cukup"
      - "selesai"
      - "bye"
      - "dadah"
  error_recovery:
    max_consecutive_failures: 3
  greeting:                        # <- 2 spasi, SEJAJAR dengan conversation
    templates:                     # <- 4 spasi
      - "Selamat {waktu}, Kak! Saya lihat Kakak memakai baju {warna}. Ada yang bisa saya bantu?"
      - "Halo Kak yang berbaju {warna}, selamat {waktu}! Silakan, mau tanya apa hari ini?"
      - "Selamat {waktu}! Kakak yang memakai baju {warna}, saya siap membantu menjawab pertanyaan Kakak."
    fallback_templates:
      - "Selamat {waktu}, Kak! Ada yang bisa saya bantu?"
      - "Halo Kak, selamat {waktu}! Silakan, mau tanya apa hari ini?"
    color_display_names:
      merah: "merah"
      oranye: "oranye"
      kuning: "kuning"
      hijau: "hijau"
      biru: "biru"
      ungu: "ungu"
      hitam: "hitam"
      putih_abu: "putih keabu-abuan"
    time_of_day:
      pagi_start_hour: 4
      siang_start_hour: 11
      sore_start_hour: 15
      malam_start_hour: 18

  error_recovery:
    max_consecutive_failures: 3

  state_timeouts_sec:
    detecting: 5
    target_locked: 3
    color_stabilizing: 2
    greeting: 10
    waiting_for_input: 10
    retrieving: 5
    generating: 15
    speaking: 30
    session_ending: 2
    cooldown: 30
    error_recovery: 5

knowledge:
  chunking:
    chunk_size_words: 100
    chunk_overlap_words: 500

  embedding:
    # Mendukung Bahasa Indonesia, gratis, jalan offline (Asumsi #7 Blueprint)
    model_name: "intfloat/multilingual-e5-small"

  vector_store:
    persist_directory: "data/vector_store"
    collection_name: "mikroskil_docs"

rag:
  top_k: 3                    # Bagian 13.2 Build Plan
  similarity_threshold: 0.35  # titik awal, WAJIB dikalibrasi ulang di CP12/CP13

llm:
  # Keputusan Anda: LLM cloud gratis. Ganti provider cukup di sini (Aturan #9).
  provider: "gemini"
  model_name: "gemini-2.0-flash"   # sesuaikan bila nama model berubah saat CP14
  temperature: 0.2
  max_tokens: 300
  timeout_sec: 15

tts:
  provider: "edge-tts"
  voice: "id-ID-ArdiNeural"    # suara Bahasa Indonesia; boleh diganti
  cache_directory: "data/audio_cache"

logging:
  level: "INFO"                # DEBUG/INFO/WARNING/ERROR (Aturan #24 Build Plan)
  log_file: "data/logs/app.log"
""",
        encoding="utf-8",
    )

    import config.settings as settings_module

    monkeypatch.setattr(settings_module, "CONFIG_YAML_PATH", bad_yaml)

    with pytest.raises(ConfigError, match="overlap"):
        settings_module.load_settings()
#!/usr/bin/env python3
"""
setup_cp01.py
=============

Script otomatis untuk membangun struktur project CHECKPOINT 01 (Foundation)
proyek "AI Greeting Universitas Mikroskil", sesuai:

  - AI Greeting Mikroskil Blueprint (arsitektur & alasan desain)
  - AI Greeting Mikroskil - Master Development & Build Plan (Bagian 6, 7, 8)

Yang dilakukan script ini:
  1. Membuat seluruh struktur folder sesuai Bagian 6 Build Plan
     (termasuk folder yang baru dipakai di checkpoint-checkpoint berikutnya,
     supaya struktur akhir tidak perlu dibongkar-pasang lagi nanti).
  2. Mengisi __init__.py di setiap folder yang merupakan Python package.
  3. Menulis file-file CP01 yang sudah berisi kode nyata:
       config/config.yaml, config/settings.py,
       src/core/enums.py, src/core/models.py,
       src/core/exceptions.py, src/core/logger.py,
       main.py, requirements.txt, README.md,
       .gitignore, .env.example, .env,
       tests/unit/test_settings.py, tests/unit/test_models.py,
       tests/conftest.py
  4. TIDAK menimpa file yang sudah ada, kecuali dijalankan dengan --force.
     Ini supaya aman dijalankan ulang tanpa menghapus pekerjaan Anda.

Cara pakai:
    1. Salin file ini ke folder project kosong, misalnya:
       ai-greeting-mikroskil/setup_cp01.py
    2. Aktifkan virtual environment Anda.
    3. Jalankan:
       python setup_cp01.py
    4. Lanjutkan dengan:
       pip install -r requirements.txt
       python main.py
       pytest tests/unit

Catatan penting (Aturan #10 Build Plan - semua angka ajaib di konfigurasi):
    Nilai-nilai di config.yaml di bawah adalah "nilai awal" yang memang
    disebutkan di Blueprint/Build Plan. Sebagian akan dikalibrasi ulang
    di checkpoint yang relevan (mis. ambang warna di CP06, ambang RAG
    di CP12/CP13) -- itu wajar dan sudah diperkirakan di dokumen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. STRUKTUR FOLDER (Bagian 6 Build Plan)
# ---------------------------------------------------------------------------
# Folder dibuat semua sekarang (kosong) supaya struktur project tidak berubah
# bentuk di checkpoint-checkpoint berikutnya -- hanya isinya yang bertambah.

FOLDERS = [
    "config",
    "src",
    "src/core",
    "src/vision",
    "src/application",
    "src/knowledge",
    "src/knowledge/parsers",
    "src/rag",
    "src/llm",
    "src/tts",
    "src/ui",
    "data/documents/raw",
    "data/documents/sample",
    "data/documents/processed",
    "data/vector_store",
    "data/audio_cache",
    "data/logs",
    "data/logs/sessions",
    "scripts",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "models",
]

# Folder yang merupakan Python package -> perlu __init__.py
PACKAGE_FOLDERS = [
    "config",
    "src",
    "src/core",
    "src/vision",
    "src/application",
    "src/knowledge",
    "src/knowledge/parsers",
    "src/rag",
    "src/llm",
    "src/tts",
    "src/ui",
    "tests",
    "tests/unit",
    "tests/integration",
]

# Folder kosong (bukan package) yang perlu ditahan git dengan .gitkeep.
# Folder yang sudah masuk .gitignore (vector_store, audio_cache, logs, models)
# tidak perlu .gitkeep karena memang sengaja tidak di-track git.
GITKEEP_FOLDERS = [
    "data/documents/raw",
    "data/documents/sample",
    "data/documents/processed",
    "scripts",
    "tests/fixtures",
]


# ---------------------------------------------------------------------------
# 2. ISI FILE
# ---------------------------------------------------------------------------

GITIGNORE = """\
# Virtual environment
venv/
.venv/

# Rahasia -- WAJIB tidak pernah masuk git (Aturan #21 Build Plan)
.env

# Cache Python
__pycache__/
*.pyc

# Data yang dihasilkan runtime, bukan source code
data/vector_store/
data/audio_cache/
data/logs/

# Model besar yang diunduh otomatis (YOLO, embedding)
models/

# Editor / OS
.vscode/
.idea/
.DS_Store

# Pytest
.pytest_cache/
"""

ENV_EXAMPLE = """\
# Salin file ini menjadi ".env" lalu isi nilainya.
# JANGAN pernah commit file ".env" (sudah ada di .gitignore).

# API key untuk LLM cloud (Gemini). Daftar gratis di Google AI Studio.
GEMINI_API_KEY=

# Opsional: dipakai kalau nanti fallback ke LLM lokal (Ollama) diaktifkan.
OLLAMA_BASE_URL=http://localhost:11434
"""

README = """\
# AI Greeting Universitas Mikroskil

Sistem penyapa calon mahasiswa berbasis computer vision (deteksi orang +
warna pakaian) dan RAG (tanya-jawab berbasis dokumen resmi Mikroskil).

Dokumen sumber kebenaran project ini:

- **AI Greeting Mikroskil Blueprint** -- kenapa sistem dirancang seperti ini.
- **AI Greeting Mikroskil - Master Development & Build Plan** -- checkpoint
  apa yang dikerjakan sekarang, dan aturan yang mengikat selama development.

Status saat ini: **Checkpoint 01 -- Foundation** (kerangka project, belum
ada logika vision/RAG apa pun).

## Menjalankan

```bash
python -m venv venv
venv\\Scripts\\activate        # Windows
pip install -r requirements.txt
copy .env.example .env         # lalu isi GEMINI_API_KEY
python main.py
pytest tests/unit
```

## Struktur Project

Lihat Bagian 6 di Master Development & Build Plan untuk penjelasan lengkap
setiap folder dan di checkpoint mana file tersebut diisi.

```
main.py                    Titik masuk aplikasi
config/                    Konfigurasi (.yaml) + pemuat & validatornya
src/core/                  Tipe data inti, enum, exception, logger
src/vision/                Kamera, deteksi orang, tracking, warna baju
src/application/           State machine, sapaan, percakapan, orchestrator
src/knowledge/             Loader dokumen, chunking, embedding, vector store
src/rag/                   Retrieval + penyusunan prompt + pipeline jawaban
src/llm/                   Interface & implementasi provider LLM
src/tts/                   Interface & implementasi text-to-speech
src/ui/                    Input pengguna & tampilan chat
data/                      Dokumen, vector store, cache audio, log
scripts/                   Skrip pemeriksaan manual per komponen
tests/                     Unit test & integration test
```

## Aturan yang Mengikat

Lihat Bagian 2 Build Plan. Ringkas: modular, satu sumber state, tidak boleh
mengarang fakta Universitas Mikroskil, satu checkpoint pada satu waktu.
"""

REQUIREMENTS_TXT = """\
# --- Checkpoint 01: hanya dependency ringan ---
python-dotenv==1.0.1
pyyaml==6.0.2
pytest==8.3.3

# --- Akan ditambah pada checkpoint berikutnya (jangan install dulu di CP01) ---
# opencv-python           # CP02 - Camera
# ultralytics             # CP03 - YOLO Person Detection
# sentence-transformers   # CP11 - Embedding
# chromadb                # CP11 - Vector Database
# google-generativeai     # CP14 - LLM (Gemini)
# edge-tts                # CP15 - TTS
# pygame                  # CP15 - Audio player
"""

CONFIG_YAML = """\
# =============================================================================
# config.yaml -- SEMUA angka ajaib project hidup di sini (Aturan #10 Build Plan)
# Dilarang menulis threshold/timer/bobot langsung di dalam kode Python.
#
# Nilai di bawah adalah "nilai awal" sesuai Blueprint & Build Plan.
# Beberapa akan dikalibrasi ulang di checkpoint yang relevan -- itu memang
# direncanakan, bukan kesalahan.
# =============================================================================

vision:
  detection:
    # Model YOLO. "n" (nano) dipilih karena laptop tanpa GPU (Asumsi #1 Blueprint).
    model_name: "yolo11n.pt"
    person_class_id: 0          # kelas "person" di COCO dataset
    confidence_threshold: 0.50  # Bagian 15 Build Plan

  tracker:
    # ~0.5 detik pada asumsi 30 FPS, dipakai sebagai syarat minimum umur track
    min_age_frames: 15

  target_selector:
    # Skor berbobot: gabungan luas bounding box + kedekatan ke tengah frame
    weight_area: 0.6
    weight_center: 0.4

  target_lock:
    confirm_timer_sec: 0.5   # lama kandidat harus stabil sebelum lock terkonfirmasi
    grace_period_sec: 2.5    # toleransi target hilang sebelum lock dilepas
    cooldown_sec: 30         # jeda sebelum orang yang sama boleh disapa lagi

  clothing_color:
    # Palet warna dibatasi 8 agar deteksi tetap sederhana dan mudah dijelaskan
    palette:
      - merah
      - oranye
      - kuning
      - hijau
      - biru
      - ungu
      - hitam
      - putih_abu
    stabilizer_buffer_sec: 0.5      # ~15 frame pada 30 FPS
    stabilizer_majority_ratio: 0.6  # ambang voting mayoritas warna
    color_wait_timeout_sec: 2.0     # batas tunggu sebelum lanjut tanpa warna

application:
  conversation:
    idle_timeout_sec: 10       # sesi ditutup otomatis setelah diam sekian detik
    max_history_turns: 10      # riwayat percakapan dipotong di titik ini

  # Timeout pengaman per state, supaya sistem tidak pernah macet selamanya
  # di satu state (Aturan #4 Bagian 16 Build Plan: setiap state non-IDLE
  # wajib punya timeout).
  state_timeouts_sec:
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
    chunk_size_words: 500
    chunk_overlap_words: 50

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
"""

CONFIG_INIT_PY = ""  # package marker kosong

SETTINGS_PY = '''\
"""
config/settings.py

Memuat konfigurasi dari dua sumber:
  - config/config.yaml  -> semua angka & pengaturan non-rahasia
  - .env                -> rahasia (API key)

lalu memvalidasi nilainya dan menyediakan SATU objek konfigurasi (`Settings`)
yang dipakai oleh seluruh module lain. Tidak ada module lain yang boleh
membaca config.yaml atau .env secara langsung -- semua lewat sini.

Ini memenuhi Aturan #9 dan #10 Build Plan: layanan eksternal & angka ajaib
dikonfigurasi di satu tempat, bukan tersebar di kode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Error khusus untuk kesalahan konfigurasi.

    Sengaja dibuat di sini (bukan di src/core/exceptions.py) karena
    settings.py harus bisa berdiri sendiri tanpa bergantung pada src/.
    """


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _require(mapping: dict[str, Any], dotted_key: str) -> Any:
    """Ambil nilai dari dict bersarang pakai key bertitik, mis. "vision.detection.model_name".

    Melempar ConfigError yang jelas (bukan KeyError mentah) kalau hilang.
    """
    node: Any = mapping
    parts = dotted_key.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            path_so_far = ".".join(parts[: i + 1])
            raise ConfigError(
                f"Konfigurasi wajib '{dotted_key}' tidak ditemukan "
                f"(berhenti di '{path_so_far}'). Periksa config/config.yaml."
            )
        node = node[part]
    return node


def _require_type(value: Any, expected_type: type, key: str) -> Any:
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"Konfigurasi '{key}' harus bertipe {expected_type.__name__}, "
            f"tapi ditemukan {type(value).__name__} (nilai: {value!r})."
        )
    return value


def _require_range(value: float, low: float, high: float, key: str) -> float:
    if not (low <= value <= high):
        raise ConfigError(
            f"Konfigurasi '{key}' harus berada di rentang [{low}, {high}], "
            f"tapi nilainya {value}."
        )
    return value


@dataclass(frozen=True)
class VisionSettings:
    model_name: str
    person_class_id: int
    confidence_threshold: float
    min_age_frames: int
    weight_area: float
    weight_center: float
    confirm_timer_sec: float
    grace_period_sec: float
    cooldown_sec: float
    color_palette: list[str]
    stabilizer_buffer_sec: float
    stabilizer_majority_ratio: float
    color_wait_timeout_sec: float


@dataclass(frozen=True)
class ApplicationSettings:
    idle_timeout_sec: float
    max_history_turns: int
    state_timeouts_sec: dict[str, float]


@dataclass(frozen=True)
class KnowledgeSettings:
    chunk_size_words: int
    chunk_overlap_words: int
    embedding_model_name: str
    vector_store_dir: str
    vector_store_collection: str


@dataclass(frozen=True)
class RAGSettings:
    top_k: int
    similarity_threshold: float


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_sec: float
    api_key: str | None  # dari .env, boleh None di CP01 (belum dipakai)


@dataclass(frozen=True)
class TTSSettings:
    provider: str
    voice: str
    cache_dir: str


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    log_file: str


@dataclass(frozen=True)
class Settings:
    """Satu objek konfigurasi tervalidasi untuk seluruh aplikasi."""

    vision: VisionSettings
    application: ApplicationSettings
    knowledge: KnowledgeSettings
    rag: RAGSettings
    llm: LLMSettings
    tts: TTSSettings
    logging: LoggingSettings
    project_root: Path = field(default=PROJECT_ROOT)

    def summary(self) -> str:
        """Ringkasan singkat untuk dicetak main.py -- tidak mencetak API key."""
        lines = [
            "=== Ringkasan Konfigurasi ===",
            f"Vision  : model={self.vision.model_name}, "
            f"conf_threshold={self.vision.confidence_threshold}, "
            f"palet_warna={len(self.vision.color_palette)} warna",
            f"App     : idle_timeout={self.application.idle_timeout_sec}s",
            f"RAG     : top_k={self.rag.top_k}, "
            f"threshold={self.rag.similarity_threshold}",
            f"LLM     : provider={self.llm.provider}, model={self.llm.model_name}, "
            f"api_key_ada={'ya' if self.llm.api_key else 'BELUM DIISI'}",
            f"TTS     : provider={self.tts.provider}, voice={self.tts.voice}",
            f"Logging : level={self.logging.level}, file={self.logging.log_file}",
        ]
        return "\\n".join(lines)


def load_settings() -> Settings:
    """Titik masuk utama: baca .env + config.yaml, validasi, kembalikan Settings.

    Melempar ConfigError dengan pesan jelas kalau ada yang salah -- sesuai
    Definition of Done CP01 ("pesan error yang jelas, bukan traceback mentah").
    """
    load_dotenv(dotenv_path=ENV_PATH)

    if not CONFIG_YAML_PATH.exists():
        raise ConfigError(
            f"File konfigurasi tidak ditemukan: {CONFIG_YAML_PATH}. "
            "Pastikan config/config.yaml ada."
        )

    with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml kosong atau formatnya tidak valid.")

    vision = VisionSettings(
        model_name=_require_type(_require(raw, "vision.detection.model_name"), str,
                                  "vision.detection.model_name"),
        person_class_id=_require_type(_require(raw, "vision.detection.person_class_id"), int,
                                       "vision.detection.person_class_id"),
        confidence_threshold=_require_range(
            _require_type(_require(raw, "vision.detection.confidence_threshold"), float,
                          "vision.detection.confidence_threshold"),
            0.0, 1.0, "vision.detection.confidence_threshold"),
        min_age_frames=_require_type(_require(raw, "vision.tracker.min_age_frames"), int,
                                      "vision.tracker.min_age_frames"),
        weight_area=_require_type(_require(raw, "vision.target_selector.weight_area"), float,
                                   "vision.target_selector.weight_area"),
        weight_center=_require_type(_require(raw, "vision.target_selector.weight_center"), float,
                                     "vision.target_selector.weight_center"),
        confirm_timer_sec=_require_type(_require(raw, "vision.target_lock.confirm_timer_sec"), float,
                                         "vision.target_lock.confirm_timer_sec"),
        grace_period_sec=_require_type(_require(raw, "vision.target_lock.grace_period_sec"), float,
                                        "vision.target_lock.grace_period_sec"),
        cooldown_sec=_require_type(_require(raw, "vision.target_lock.cooldown_sec"), (int, float),
                                    "vision.target_lock.cooldown_sec"),
        color_palette=_require_type(_require(raw, "vision.clothing_color.palette"), list,
                                     "vision.clothing_color.palette"),
        stabilizer_buffer_sec=_require_type(
            _require(raw, "vision.clothing_color.stabilizer_buffer_sec"), float,
            "vision.clothing_color.stabilizer_buffer_sec"),
        stabilizer_majority_ratio=_require_range(
            _require_type(_require(raw, "vision.clothing_color.stabilizer_majority_ratio"), float,
                          "vision.clothing_color.stabilizer_majority_ratio"),
            0.0, 1.0, "vision.clothing_color.stabilizer_majority_ratio"),
        color_wait_timeout_sec=_require_type(
            _require(raw, "vision.clothing_color.color_wait_timeout_sec"), float,
            "vision.clothing_color.color_wait_timeout_sec"),
    )

    application = ApplicationSettings(
        idle_timeout_sec=_require_type(_require(raw, "application.conversation.idle_timeout_sec"),
                                        (int, float), "application.conversation.idle_timeout_sec"),
        max_history_turns=_require_type(_require(raw, "application.conversation.max_history_turns"),
                                         int, "application.conversation.max_history_turns"),
        state_timeouts_sec=_require_type(_require(raw, "application.state_timeouts_sec"), dict,
                                          "application.state_timeouts_sec"),
    )

    knowledge = KnowledgeSettings(
        chunk_size_words=_require_type(_require(raw, "knowledge.chunking.chunk_size_words"), int,
                                        "knowledge.chunking.chunk_size_words"),
        chunk_overlap_words=_require_type(_require(raw, "knowledge.chunking.chunk_overlap_words"), int,
                                           "knowledge.chunking.chunk_overlap_words"),
        embedding_model_name=_require_type(_require(raw, "knowledge.embedding.model_name"), str,
                                            "knowledge.embedding.model_name"),
        vector_store_dir=_require_type(_require(raw, "knowledge.vector_store.persist_directory"), str,
                                        "knowledge.vector_store.persist_directory"),
        vector_store_collection=_require_type(_require(raw, "knowledge.vector_store.collection_name"), str,
                                               "knowledge.vector_store.collection_name"),
    )

    if knowledge.chunk_overlap_words >= knowledge.chunk_size_words:
        raise ConfigError(
            "knowledge.chunking.chunk_overlap_words harus lebih kecil dari "
            "chunk_size_words (overlap tidak boleh >= ukuran chunk)."
        )

    rag = RAGSettings(
        top_k=_require_type(_require(raw, "rag.top_k"), int, "rag.top_k"),
        similarity_threshold=_require_range(
            _require_type(_require(raw, "rag.similarity_threshold"), float, "rag.similarity_threshold"),
            0.0, 1.0, "rag.similarity_threshold"),
    )

    llm = LLMSettings(
        provider=_require_type(_require(raw, "llm.provider"), str, "llm.provider"),
        model_name=_require_type(_require(raw, "llm.model_name"), str, "llm.model_name"),
        temperature=_require_type(_require(raw, "llm.temperature"), float, "llm.temperature"),
        max_tokens=_require_type(_require(raw, "llm.max_tokens"), int, "llm.max_tokens"),
        timeout_sec=_require_type(_require(raw, "llm.timeout_sec"), (int, float), "llm.timeout_sec"),
        api_key=os.getenv("GEMINI_API_KEY") or None,
    )

    tts = TTSSettings(
        provider=_require_type(_require(raw, "tts.provider"), str, "tts.provider"),
        voice=_require_type(_require(raw, "tts.voice"), str, "tts.voice"),
        cache_dir=_require_type(_require(raw, "tts.cache_directory"), str, "tts.cache_directory"),
    )

    logging_settings = LoggingSettings(
        level=_require_type(_require(raw, "logging.level"), str, "logging.level"),
        log_file=_require_type(_require(raw, "logging.log_file"), str, "logging.log_file"),
    )

    return Settings(
        vision=vision,
        application=application,
        knowledge=knowledge,
        rag=rag,
        llm=llm,
        tts=tts,
        logging=logging_settings,
    )
'''

ENUMS_PY = '''\
"""
src/core/enums.py

Enum untuk seluruh nilai bertipe "salah satu dari daftar tetap" di project
ini. Memakai Enum (bukan string bebas) supaya salah ketik terdeteksi oleh
editor/type-checker, bukan muncul diam-diam saat runtime.

core/ tidak boleh meng-import module lain (Aturan #6 Build Plan), jadi file
ini hanya memakai library standar Python.
"""

from __future__ import annotations

from enum import Enum


class AppState(str, Enum):
    """Seluruh state pada StateMachine aplikasi (Bagian 16 Build Plan).

    Diwarisi dari `str` supaya mudah dibandingkan dan di-log sebagai teks.
    """

    IDLE = "IDLE"
    DETECTING = "DETECTING"
    TARGET_LOCKED = "TARGET_LOCKED"
    COLOR_STABILIZING = "COLOR_STABILIZING"
    GREETING = "GREETING"
    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
    RETRIEVING = "RETRIEVING"
    GENERATING = "GENERATING"
    SPEAKING = "SPEAKING"
    SESSION_ENDING = "SESSION_ENDING"
    COOLDOWN = "COOLDOWN"
    ERROR_RECOVERY = "ERROR_RECOVERY"


class LockPhase(str, Enum):
    """Fase internal TargetLock (Bagian 5.6 Build Plan, field `LockState.state`)."""

    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    LOCKED = "LOCKED"
    LOST_GRACE = "LOST_GRACE"
    COOLDOWN = "COOLDOWN"


class ColorName(str, Enum):
    """Palet 8 warna baju yang dikenali (Blueprint bagian D5)."""

    MERAH = "merah"
    ORANYE = "oranye"
    KUNING = "kuning"
    HIJAU = "hijau"
    BIRU = "biru"
    UNGU = "ungu"
    HITAM = "hitam"
    PUTIH_ABU = "putih_abu"


class ReleaseReason(str, Enum):
    """Alasan TargetLock dilepas, untuk log dan analisis (field `release_reason`).

    TARGET_LOST eksplisit disebut di Build Plan; nilai lain ditambahkan agar
    lengkap secara logis dan bisa dipakai langsung tanpa menunggu CP05.
    """

    TARGET_LOST = "TARGET_LOST"      # target hilang melewati grace period
    SESSION_ENDED = "SESSION_ENDED"  # dilepas karena sesi percakapan selesai
    MANUAL = "MANUAL"                # dilepas manual (mis. untuk testing)
    ERROR = "ERROR"                  # dilepas karena error tak tertangani


class EndReason(str, Enum):
    """Alasan ConversationSession berakhir (Bagian 5.14 Build Plan)."""

    TIMEOUT = "TIMEOUT"
    FAREWELL = "FAREWELL"
    TARGET_LOST = "TARGET_LOST"
    MANUAL = "MANUAL"
'''

MODELS_PY = '''\
"""
src/core/models.py

Seluruh dataclass "kontrak data" dari Bagian 5 Build Plan. Hanya bentuk
data (field) -- TIDAK ADA logika bisnis di file ini (lihat catatan di
Bagian 5 Build Plan: "definisi bentuk data, bukan implementasi").

core/ tidak boleh meng-import module lain (Aturan #6 Build Plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.enums import AppState, ColorName, LockPhase, ReleaseReason


# --------------------------------------------------------------------------
# Lapisan Vision (Bagian 5.1 - 5.8)
# --------------------------------------------------------------------------

@dataclass
class Frame:
    """Satu frame mentah dari kamera. Dibuat oleh Camera."""

    image: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int


@dataclass(frozen=True)
class BoundingBox:
    """Kotak posisi objek dalam piksel. Dibuat oleh PersonDetector."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(frozen=True)
class Detection:
    """Satu hasil deteksi YOLO mentah. Dibuat oleh PersonDetector."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    frame_id: int


@dataclass
class TrackedPerson:
    """Satu orang dengan identitas konsisten antar-frame. Dibuat oleh Tracker."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    first_seen: float
    last_seen: float
    age_frames: int


@dataclass
class TargetCandidate:
    """Skor satu kandidat target. Dibuat oleh TargetSelector."""

    track_id: int
    score: float
    area_score: float
    center_score: float
    is_eligible: bool


@dataclass
class LockState:
    """State TargetLock saat ini. Dibuat oleh TargetLock."""

    state: LockPhase
    track_id: int | None = None
    bbox: BoundingBox | None = None
    locked_since: float | None = None
    last_seen: float | None = None
    lost_duration: float = 0.0
    release_reason: ReleaseReason | None = None


@dataclass
class ColorResult:
    """Hasil klasifikasi warna satu frame. Dibuat oleh ClothingColorDetector."""

    color_name: ColorName | None
    confidence: float
    hsv_median: tuple[int, int, int]
    pixel_count: int
    roi_valid: bool


@dataclass
class StableColor:
    """Hasil warna yang sudah distabilkan lewat buffer. Dibuat oleh ColorStabilizer."""

    color_name: ColorName | None
    ratio: float
    sample_count: int
    is_stable: bool


# --------------------------------------------------------------------------
# Lapisan Knowledge (Bagian 5.9 - 5.10)
# --------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Satu potongan teks siap embed. Dibuat oleh Chunker."""

    chunk_id: str
    text: str
    source_file: str
    page: int | None
    section: str | None
    char_count: int
    ingested_at: str


@dataclass
class RetrievedChunk:
    """Satu hasil pencarian vector store. Dibuat oleh Retriever."""

    chunk: DocumentChunk
    score: float
    rank: int
    passed_threshold: bool


# --------------------------------------------------------------------------
# Lapisan AI (Bagian 5.11 - 5.13)
# --------------------------------------------------------------------------

@dataclass
class RAGContext:
    """Konteks yang dirakit RAGPipeline sebelum dikirim ke PromptBuilder."""

    question: str
    chunks: list[RetrievedChunk]
    history: list[dict]
    has_context: bool
    top_score: float


@dataclass
class LLMResponse:
    """Jawaban final dari LLMService / RAGPipeline."""

    text: str
    sources: list[str]
    used_context: bool
    is_fallback: bool
    latency_ms: int
    provider: str
    error: str | None = None


@dataclass
class TTSResponse:
    """Hasil sintesis suara dari TTSService."""

    audio_path: str | None
    duration_sec: float
    from_cache: bool
    success: bool
    error: str | None = None


# --------------------------------------------------------------------------
# Lapisan Application (Bagian 5.14 - 5.15)
# --------------------------------------------------------------------------

@dataclass
class ConversationSession:
    """Satu sesi percakapan dari sapaan sampai penutup. Dibuat oleh ConversationManager."""

    session_id: str
    track_id: int
    clothing_color: ColorName | None
    started_at: float
    last_activity_at: float
    history: list[dict] = field(default_factory=list)
    turn_count: int = 0
    unanswered_count: int = 0
    end_reason: str | None = None


@dataclass
class Event:
    """Satu event yang mengalir ke StateMachine / Orchestrator."""

    type: str
    payload: dict
    timestamp: float
'''

EXCEPTIONS_PY = '''\
"""
src/core/exceptions.py

Exception khusus project, supaya error dari komponen kita bisa dibedakan
dari error pustaka pihak ketiga (mis. cv2.error, requests.Timeout).

core/ tidak boleh meng-import module lain (Aturan #6 Build Plan).
"""

from __future__ import annotations


class AppError(Exception):
    """Kelas dasar untuk semua exception khusus project ini."""


class CameraError(AppError):
    """Gagal membuka, membaca, atau menutup perangkat kamera."""


class DetectorError(AppError):
    """Gagal memuat atau menjalankan model deteksi (YOLO)."""


class KnowledgeError(AppError):
    """Gagal memuat, memproses, atau mengindeks dokumen pengetahuan."""


class RetrievalError(AppError):
    """Gagal melakukan pencarian di vector store."""


class LLMError(AppError):
    """Gagal memanggil atau mendapat respons dari penyedia LLM."""


class TTSError(AppError):
    """Gagal mensintesis atau memutar audio hasil text-to-speech."""
'''

LOGGER_PY = '''\
"""
src/core/logger.py

Konfigurasi logging terpusat: satu fungsi `get_logger()` yang dipakai oleh
seluruh module, supaya format log seragam di semua tempat (Aturan #24
Build Plan):
    DEBUG   -> detail per-frame
    INFO    -> peristiwa (lock, greeting, sesi)
    WARNING -> kondisi tidak ideal tapi sistem tetap jalan
    ERROR   -> kegagalan komponen

core/ tidak boleh meng-import module lain di luar library standar
(Aturan #6 Build Plan), jadi level default dibaca dari environment
variable, bukan dari config/settings.py (untuk menghindari import silang).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_CONFIGURED = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = _project_root() / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Dapatkan logger dengan konfigurasi seragam.

    Dipanggil seperti: `logger = get_logger(__name__)` di setiap module.
    """
    _configure_root_logger()
    return logging.getLogger(name)
'''

MAIN_PY = '''\
"""
main.py

Titik masuk tunggal aplikasi. TIDAK BOLEH berisi logika bisnis (Bagian 6.1
Build Plan) -- tugasnya hanya merakit objek dan menjalankan Orchestrator.

Versi Checkpoint 01: baru memuat konfigurasi dan mencetak ringkasannya,
untuk membuktikan fondasi (config + logging + model data) sudah berjalan.
Orchestrator sungguhan baru dirakit di CP16.
"""

from __future__ import annotations

import sys

from config.settings import ConfigError, load_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    logger.info("Memulai AI Greeting Mikroskil - Checkpoint 01 (Foundation)")

    try:
        settings = load_settings()
    except ConfigError as exc:
        # Pesan jelas untuk manusia, bukan traceback mentah (DoD CP01)
        print(f"[KONFIGURASI SALAH] {exc}", file=sys.stderr)
        logger.error("Gagal memuat konfigurasi: %s", exc)
        return 1

    print(settings.summary())
    logger.info("Foundation OK -- konfigurasi termuat dan tervalidasi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

CONFTEST_PY = '''\
"""
tests/conftest.py

Fixture bersama untuk seluruh test suite. Kosong untuk CP01 -- akan diisi
fixture kamera palsu / frame sintetis mulai CP02 ke atas.
"""
'''

TEST_SETTINGS_PY = '''\
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
  detection:
    model_name: "yolo11n.pt"
    person_class_id: 0
    confidence_threshold: 0.5
  tracker:
    min_age_frames: 15
  target_selector:
    weight_area: 0.6
    weight_center: 0.4
  target_lock:
    confirm_timer_sec: 0.5
    grace_period_sec: 2.5
    cooldown_sec: 30
  clothing_color:
    palette: ["merah", "biru"]
    stabilizer_buffer_sec: 0.5
    stabilizer_majority_ratio: 0.6
    color_wait_timeout_sec: 2.0
application:
  conversation:
    idle_timeout_sec: 10
    max_history_turns: 10
  state_timeouts_sec:
    greeting: 10
knowledge:
  chunking:
    chunk_size_words: 100
    chunk_overlap_words: 500
  embedding:
    model_name: "x"
  vector_store:
    persist_directory: "data/vector_store"
    collection_name: "x"
rag:
  top_k: 3
  similarity_threshold: 0.35
llm:
  provider: "gemini"
  model_name: "x"
  temperature: 0.2
  max_tokens: 300
  timeout_sec: 15
tts:
  provider: "edge-tts"
  voice: "id-ID-ArdiNeural"
  cache_directory: "data/audio_cache"
logging:
  level: "INFO"
  log_file: "data/logs/app.log"
""",
        encoding="utf-8",
    )

    import config.settings as settings_module

    monkeypatch.setattr(settings_module, "CONFIG_YAML_PATH", bad_yaml)

    with pytest.raises(ConfigError, match="overlap"):
        settings_module.load_settings()
'''

TEST_MODELS_PY = '''\
"""
tests/unit/test_models.py

Unit test untuk src/core/models.py -- memastikan setiap dataclass bisa
dibuat dan properti turunan (mis. BoundingBox.area) menghitung dengan benar.
"""

from __future__ import annotations

import numpy as np

from src.core.enums import ColorName, LockPhase, ReleaseReason
from src.core.models import (
    BoundingBox,
    ColorResult,
    ConversationSession,
    Detection,
    Event,
    Frame,
    LockState,
    StableColor,
    TargetCandidate,
    TrackedPerson,
)


def test_frame_bisa_dibuat():
    frame = Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=123.0,
        frame_id=1,
        width=640,
        height=480,
    )
    assert frame.width == 640
    assert frame.height == 480


def test_bounding_box_properti_turunan():
    box = BoundingBox(x1=100, y1=50, x2=200, y2=250)

    assert box.width == 100
    assert box.height == 200
    assert box.area == 100 * 200
    assert box.center == (150.0, 150.0)
    assert box.aspect_ratio == 100 / 200


def test_bounding_box_height_nol_tidak_crash():
    box = BoundingBox(x1=0, y1=0, x2=10, y2=0)
    assert box.aspect_ratio == 0.0


def test_detection_dan_tracked_person():
    box = BoundingBox(0, 0, 10, 10)
    detection = Detection(bbox=box, confidence=0.9, class_id=0, frame_id=1)
    assert detection.confidence == 0.9

    person = TrackedPerson(
        track_id=1, bbox=box, confidence=0.9,
        first_seen=0.0, last_seen=1.0, age_frames=30,
    )
    assert person.track_id == 1


def test_lock_state_default_values():
    lock = LockState(state=LockPhase.IDLE)
    assert lock.track_id is None
    assert lock.release_reason is None


def test_lock_state_dengan_release_reason():
    lock = LockState(state=LockPhase.LOST_GRACE, release_reason=ReleaseReason.TARGET_LOST)
    assert lock.release_reason == ReleaseReason.TARGET_LOST


def test_color_result_dan_stable_color():
    result = ColorResult(
        color_name=ColorName.BIRU, confidence=0.8,
        hsv_median=(110, 200, 180), pixel_count=5000, roi_valid=True,
    )
    assert result.color_name == ColorName.BIRU

    stable = StableColor(color_name=ColorName.BIRU, ratio=0.7, sample_count=15, is_stable=True)
    assert stable.is_stable is True


def test_target_candidate():
    candidate = TargetCandidate(
        track_id=1, score=0.75, area_score=0.8, center_score=0.6, is_eligible=True,
    )
    assert candidate.is_eligible is True


def test_conversation_session_default_history_kosong():
    session = ConversationSession(
        session_id="abc123", track_id=1, clothing_color=ColorName.HIJAU,
        started_at=0.0, last_activity_at=0.0,
    )
    assert session.history == []
    assert session.turn_count == 0


def test_event_bisa_dibuat():
    event = Event(type="PERSON_DETECTED", payload={"track_id": 1}, timestamp=0.0)
    assert event.type == "PERSON_DETECTED"
'''

INIT_PY = ""  # marker package kosong, dipakai untuk semua __init__.py


# ---------------------------------------------------------------------------
# 3. PEMETAAN FILE -> ISI
# ---------------------------------------------------------------------------

FILES: dict[str, str] = {
    ".gitignore": GITIGNORE,
    ".env.example": ENV_EXAMPLE,
    ".env": ENV_EXAMPLE,  # salinan awal; user mengisi nilainya sendiri
    "README.md": README,
    "requirements.txt": REQUIREMENTS_TXT,
    "main.py": MAIN_PY,
    "config/config.yaml": CONFIG_YAML,
    "config/settings.py": SETTINGS_PY,
    "src/core/enums.py": ENUMS_PY,
    "src/core/models.py": MODELS_PY,
    "src/core/exceptions.py": EXCEPTIONS_PY,
    "src/core/logger.py": LOGGER_PY,
    "tests/conftest.py": CONFTEST_PY,
    "tests/unit/test_settings.py": TEST_SETTINGS_PY,
    "tests/unit/test_models.py": TEST_MODELS_PY,
}

# __init__.py untuk setiap package folder
for _pkg in PACKAGE_FOLDERS:
    FILES[f"{_pkg}/__init__.py"] = INIT_PY


# ---------------------------------------------------------------------------
# 4. LOGIKA SCAFFOLDING
# ---------------------------------------------------------------------------

def build(root: Path, force: bool) -> None:
    created, skipped = [], []

    for folder in FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)

    for rel_path, content in FILES.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            skipped.append(rel_path)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(rel_path)

    for folder in GITKEEP_FOLDERS:
        keep = root / folder / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    print(f"Folder dibuat/diverifikasi : {len(FOLDERS)}")
    print(f"File ditulis                : {len(created)}")
    for f in created:
        print(f"  + {f}")
    if skipped:
        print(f"File dilewati (sudah ada)   : {len(skipped)}  (pakai --force untuk menimpa)")
        for f in skipped:
            print(f"  = {f}")

    print()
    print("Checkpoint 01 (Foundation) selesai di-scaffold.")
    print("Langkah selanjutnya:")
    print("  1. pip install -r requirements.txt")
    print("  2. copy .env.example ke .env lalu isi GEMINI_API_KEY (boleh nanti, dipakai mulai CP14)")
    print("  3. python main.py         -> harus mencetak ringkasan konfigurasi tanpa error")
    print("  4. pytest tests/unit      -> harus semua lolos (hijau)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold Checkpoint 01 (Foundation) - AI Greeting Mikroskil"
    )
    parser.add_argument(
        "--dir", default=".", help="Folder tujuan project (default: folder saat ini)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Timpa file yang sudah ada (defaultnya file yang sudah ada dilewati)",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Scaffolding project di: {root}")
    print()
    build(root, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
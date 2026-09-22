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

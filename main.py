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

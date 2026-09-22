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

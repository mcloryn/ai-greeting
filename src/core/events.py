"""
src/core/events.py

Definisi seluruh JENIS EVENT dan STRUKTUR PAYLOAD-nya (CP07, Bagian 16).

Event adalah "surat pendek" yang dikirim lapisan mana pun (vision, UI, RAG,
TTS, ...) ke StateMachine untuk memberi tahu: "sesuatu baru saja terjadi".
Bentuk objek Event-nya sendiri (type, payload, timestamp) sudah didefinisikan
di models.py (Bagian 5.15). File ini melengkapinya dengan:

  1. EventType          : daftar tertutup jenis event.
  2. REQUIRED_PAYLOAD_KEYS : kunci payload yang WAJIB ada untuk tiap jenis.
  3. make_event()       : satu-satunya pintu resmi membuat Event yang valid.

Aturan pemakaian (penting untuk Orchestrator di CP16):
  - Event bersifat "edge-triggered": dikirim SEKALI saat sesuatu berubah,
    bukan setiap frame. Mengirim PERSON_DETECTED 30 kali per detik akan
    membanjiri log dengan peringatan.
  - Kegagalan komponen yang boleh diteruskan (TTS gagal, LLM gagal, vector
    store gagal) dilaporkan sebagai event NORMAL dengan payload
    {"degraded": True}. Hanya kegagalan fatal yang memakai ERROR.

core/ tidak boleh meng-import module lain di luar core (Aturan #6).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from src.core.models import Event


class EventType(str, Enum):
    """Seluruh jenis event. Nama sama persis dengan diagram di Bagian 16.1."""

    # --- Jalur vision (IDLE sampai COLOR_STABILIZING) ---
    PERSON_DETECTED = "PERSON_DETECTED"    # ada orang muncul di frame
    CANDIDATE_STABLE = "CANDIDATE_STABLE"  # kandidat bertahan cukup lama
    CANDIDATE_LOST = "CANDIDATE_LOST"      # kandidat hilang sebelum stabil
    LOCK_CONFIRMED = "LOCK_CONFIRMED"      # lock terkonfirmasi oleh TargetLock
    COLOR_READY = "COLOR_READY"            # warna baju sudah stabil
    COLOR_TIMEOUT = "COLOR_TIMEOUT"        # batas tunggu warna habis

    # --- Jalur percakapan ---
    SPEECH_DONE = "SPEECH_DONE"            # ucapan (TTS) selesai
    USER_MESSAGE = "USER_MESSAGE"          # pengguna mengirim pesan
    CONTEXT_READY = "CONTEXT_READY"        # retrieval selesai
    ANSWER_READY = "ANSWER_READY"          # LLM selesai menjawab

    # --- Penutupan sesi ---
    IDLE_TIMEOUT = "IDLE_TIMEOUT"          # pengguna diam terlalu lama
    FAREWELL = "FAREWELL"                  # pengguna mengucap kata penutup
    TARGET_LOST = "TARGET_LOST"            # target hilang melewati toleransi
    CLEANUP_DONE = "CLEANUP_DONE"          # pembersihan sesi selesai
    COOLDOWN_EXPIRED = "COOLDOWN_EXPIRED"  # masa cooldown habis

    # --- Error ---
    ERROR = "ERROR"                        # kegagalan fatal komponen
    RECOVERY_DONE = "RECOVERY_DONE"        # pemulihan berhasil
    RECOVERY_FAILED = "RECOVERY_FAILED"    # pemulihan gagal


# Kunci payload yang WAJIB ada. Kunci lain bersifat opsional; yang dipakai:
#   - "degraded": bool  -> komponen gagal/timeout, hasil berupa fallback
#                          (SPEECH_DONE, CONTEXT_READY, ANSWER_READY)
#   - "text": str       -> teks jawaban (ANSWER_READY, kalau tidak degraded)
#   - "context": ...    -> hasil retrieval (CONTEXT_READY)
#   - "timeout": bool   -> event dibuat oleh timeout StateMachine sendiri
REQUIRED_PAYLOAD_KEYS: dict[EventType, tuple[str, ...]] = {
    EventType.PERSON_DETECTED: (),
    EventType.CANDIDATE_STABLE: ("track_id",),
    EventType.CANDIDATE_LOST: (),
    EventType.LOCK_CONFIRMED: ("track_id",),
    EventType.COLOR_READY: ("color",),          # nama warna, mis. "merah"
    EventType.COLOR_TIMEOUT: (),
    EventType.SPEECH_DONE: (),
    EventType.USER_MESSAGE: ("text",),
    EventType.CONTEXT_READY: (),
    EventType.ANSWER_READY: (),
    EventType.IDLE_TIMEOUT: (),
    EventType.FAREWELL: (),
    EventType.TARGET_LOST: (),
    EventType.CLEANUP_DONE: (),
    EventType.COOLDOWN_EXPIRED: (),
    EventType.ERROR: ("source", "message"),     # komponen mana + pesan error
    EventType.RECOVERY_DONE: (),
    EventType.RECOVERY_FAILED: (),
}


def make_event(
    event_type: EventType | str,
    timestamp: float,
    payload: Mapping[str, Any] | None = None,
) -> Event:
    """Buat Event yang sudah divalidasi.

    Args:
        event_type: EventType, atau string yang cocok dengan nilainya.
        timestamp:  waktu kejadian. WAJIB diisi pemanggil (tanpa default)
                    supaya seluruh sistem memakai satu sumber jam yang sama.
        payload:    data pendukung; disalin agar tidak terpengaruh perubahan
                    dict asli di pemanggil.

    Raises:
        ValueError: jenis event tidak dikenal, atau kunci wajib payload hilang.
                    Ini menandakan BUG pada pemanggil, bukan kegagalan runtime.
    """
    etype = EventType(event_type)  # ValueError bila tidak dikenal
    data = dict(payload or {})

    missing = [key for key in REQUIRED_PAYLOAD_KEYS[etype] if key not in data]
    if missing:
        raise ValueError(
            f"Event {etype.value} kekurangan kunci payload wajib: {missing}"
        )

    return Event(type=etype, payload=data, timestamp=timestamp)
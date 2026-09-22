"""
Fungsi untuk membuat dan mengelola objek ConversationSession.

Catatan arsitektur: dataclass ConversationSession sendiri sudah
didefinisikan di src/core/models.py sejak CP01 (Bagian 5.14).
File ini TIDAK mendefinisikan ulang strukturnya -- hanya berisi
fungsi-fungsi yang mengoperasikan objek tersebut, supaya definisi
data (models.py) tetap terpisah dari logika (session.py).
"""

import time
import uuid
import json
from pathlib import Path

from src.core.models import ConversationSession
from src.core.enums import EndReason
from src.core.logger import get_logger

logger = get_logger(__name__)

SESSION_LOG_DIR = Path("data/logs/sessions")


def create_session(track_id: int, clothing_color: str | None) -> ConversationSession:
    """Membuat sesi percakapan baru untuk satu target yang baru disapa."""
    now = time.time()
    session = ConversationSession(
        session_id=str(uuid.uuid4()),
        track_id=track_id,
        clothing_color=clothing_color,
        started_at=now,
        last_activity_at=now,
        history=[],
        turn_count=0,
        unanswered_count=0,
        end_reason=None,
    )
    logger.info(f"Sesi dimulai: {session.session_id} (track_id={track_id})")
    return session


def add_user_message(session: ConversationSession, text: str, max_history_turns: int) -> None:
    """Menambahkan pesan user ke riwayat dan memperbarui waktu aktivitas."""
    session.history.append({"role": "user", "content": text, "ts": time.time()})
    session.last_activity_at = time.time()
    _trim_history(session, max_history_turns)


def add_assistant_message(session: ConversationSession, text: str, max_history_turns: int) -> None:
    """Menambahkan balasan asisten ke riwayat dan menghitung giliran selesai."""
    session.history.append({"role": "assistant", "content": text, "ts": time.time()})
    session.turn_count += 1
    _trim_history(session, max_history_turns)


def _trim_history(session: ConversationSession, max_history_turns: int) -> None:
    """
    Membatasi riwayat ke N giliran terakhir.
    Satu giliran = 1 pesan user + 1 pesan asisten = 2 entri di history.
    """
    max_entries = max_history_turns * 2
    if len(session.history) > max_entries:
        session.history = session.history[-max_entries:]


def is_farewell(text: str, farewell_words: list[str]) -> bool:
    """Mengecek apakah teks user mengandung kata penutup."""
    lowered = text.lower()
    return any(word in lowered for word in farewell_words)


def end_session(session: ConversationSession, reason: EndReason) -> None:
    """Menutup sesi, mencatat alasan, dan menyimpan arsip ke disk."""
    session.end_reason = reason
    logger.info(
        f"Sesi berakhir: {session.session_id} "
        f"(alasan={reason}, turn_count={session.turn_count})"
    )
    _archive_session(session)


def _archive_session(session: ConversationSession) -> None:
    """Menyimpan sesi sebagai file JSON di data/logs/sessions/."""
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SESSION_LOG_DIR / f"{session.session_id}.json"

    # __dict__ dipakai karena ConversationSession adalah dataclass biasa.
    # EndReason adalah enum, jadi diubah ke .value supaya bisa disimpan
    # sebagai JSON (JSON tidak mengenal tipe Enum Python).
    data = dict(session.__dict__)
    if isinstance(data.get("end_reason"), EndReason):
        data["end_reason"] = data["end_reason"].value

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.debug(f"Arsip sesi disimpan: {file_path}")
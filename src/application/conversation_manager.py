"""
src/application/conversation_manager.py

Pelaksana aksi StateMachine untuk siklus percakapan: mulai sesi, mencatat
riwayat, menyediakan jawaban (dummy untuk CP09), dan mengarsipkan sesi.

Pola modul ini SAMA seperti GreetingManager (CP08): StateMachine memutuskan
KAPAN, modul ini MELAKUKAN. ConversationManager tidak pernah memutuskan
sendiri kapan sesi berakhir atau kapan idle -- itu murni milik StateMachine
(Aturan #8, Satu Sumber State).
"""

from __future__ import annotations

import time

from src.core.models import ConversationSession
from src.core.enums import EndReason
from src.core.logger import get_logger
from src.application import session as session_ops

logger = get_logger(__name__)


class ConversationManager:
    def __init__(self, max_history_turns: int):
        self._max_history_turns = max_history_turns
        self._active_session: ConversationSession | None = None
        self._pending_question: str = ""

    @property
    def active_session(self) -> ConversationSession | None:
        return self._active_session

    def start_session(self, track_id: int, clothing_color) -> None:
        """Dipanggil saat aksi START_SESSION (exit state GREETING)."""
        self._active_session = session_ops.create_session(track_id, clothing_color)

    def touch_activity(self) -> None:
        """Dipanggil saat aksi TOUCH_ACTIVITY (mis. setelah SPEAKING selesai)."""
        if self._active_session is not None:
            self._active_session.last_activity_at = time.time()

    def receive_question(self, text: str) -> None:
        """Dipanggil saat aksi RUN_RETRIEVAL. Menyimpan pertanyaan ke riwayat.

        RAG asli baru dipasang CP13 -- untuk sekarang pertanyaan hanya
        disimpan, jawabannya dibuat oleh build_dummy_answer().
        """
        self._pending_question = text
        if self._active_session is not None:
            session_ops.add_user_message(self._active_session, text, self._max_history_turns)

    def build_dummy_answer(self) -> str:
        """Dipanggil saat aksi RUN_LLM. Balasan tiruan -- LLM asli baru CP14."""
        reply = (
            f'(dummy) Kamu bertanya: "{self._pending_question}". '
            "Jawaban asli menyusul di CP13."
        )
        if self._active_session is not None:
            session_ops.add_assistant_message(self._active_session, reply, self._max_history_turns)
        return reply

    def archive_session(self, reason: str | None) -> None:
        """Dipanggil saat aksi ARCHIVE_SESSION (entry state SESSION_ENDING)."""
        if self._active_session is None:
            return
        try:
            end_reason = EndReason(reason) if reason else EndReason.MANUAL
        except ValueError:
            logger.warning(f"end_reason tidak dikenal: {reason!r}, memakai MANUAL")
            end_reason = EndReason.MANUAL
        session_ops.end_session(self._active_session, end_reason)
        self._active_session = None
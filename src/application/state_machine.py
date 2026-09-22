"""
src/application/state_machine.py

Satu-satunya pemilik state aplikasi (Aturan #8, Bagian 16 Build Plan).

Cara kerja dalam satu kalimat:
    Event masuk -> mesin memeriksa TABEL TRANSISI -> state berpindah ->
    mesin mengembalikan DAFTAR AKSI yang nanti dijalankan Orchestrator.

Mesin ini TIDAK PERNAH memanggil module lain (kamera, LLM, TTS, ...).
Ia hanya menjawab "sekarang kita ada di state apa, dan apa yang harus
dilakukan?". Karena itu module ini bisa diuji tanpa kamera, tanpa internet,
dan tanpa pustaka berat -- hanya library standar Python dan src.core.

Ada dua cara mesin "digerakkan":
    handle(event) : ada sesuatu terjadi.
    tick(now)     : waktu berjalan. Orchestrator WAJIB memanggil ini di
                    setiap putaran loop utama; tanpa itu timeout tidak akan
                    pernah diperiksa (salah satu Failure Case CP07).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.core.enums import AppState, EndReason, ReleaseReason
from src.core.events import EventType, make_event
from src.core.models import Event

logger = logging.getLogger(__name__)


# =============================================================================
# 1. AKSI -- "perintah" yang dikembalikan mesin, dijalankan oleh Orchestrator
# =============================================================================

class ActionType(str, Enum):
    """Jenis aksi. Aksi hanya DESKRIPSI apa yang harus dilakukan."""

    # Kebersihan / reset
    CLEAR_TARGET = "CLEAR_TARGET"
    RESET_DETECTION = "RESET_DETECTION"
    RESET_COLOR_STABILIZER = "RESET_COLOR_STABILIZER"
    PURGE_EXPIRED_COOLDOWNS = "PURGE_EXPIRED_COOLDOWNS"

    # Vision
    START_CONFIRM_TIMER = "START_CONFIRM_TIMER"

    # Sapaan dan sesi
    BUILD_GREETING = "BUILD_GREETING"          # rakit kalimat + kirim ke antrean suara
    MARK_GREETED = "MARK_GREETED"              # tandai orang ini sudah disapa
    START_SESSION = "START_SESSION"
    TOUCH_ACTIVITY = "TOUCH_ACTIVITY"          # perbarui waktu aktivitas terakhir
    ARCHIVE_SESSION = "ARCHIVE_SESSION"        # data: {"reason": EndReason}
    RELEASE_LOCK = "RELEASE_LOCK"              # data: {"reason": ReleaseReason}
    REGISTER_COOLDOWN = "REGISTER_COOLDOWN"

    # Pipeline jawaban
    SHOW_PROGRESS_INDICATOR = "SHOW_PROGRESS_INDICATOR"
    RUN_RETRIEVAL = "RUN_RETRIEVAL"            # data: {"text": ...}
    RUN_LLM = "RUN_LLM"
    SPEAK_ANSWER = "SPEAK_ANSWER"

    # Error
    LOG_ERROR = "LOG_ERROR"
    ATTEMPT_RECOVERY = "ATTEMPT_RECOVERY"      # data: {"attempt": n}
    STOP_APPLICATION = "STOP_APPLICATION"


@dataclass(frozen=True)
class Action:
    """Satu perintah untuk Orchestrator.

    `data` membawa informasi pendukung. Sebagian besar disalin dari payload
    event pemicunya, sehingga Orchestrator tidak perlu menyimpan salinan
    sendiri (mis. teks pertanyaan pengguna).
    """

    type: ActionType
    data: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 2. TABEL TRANSISI -- "dari state mana, dengan event apa, menuju state mana"
# =============================================================================
# Kalau pasangan (state, event) TIDAK ada di tabel ini, event tersebut ditolak
# dengan log WARNING. Inilah seluruh "aturan main" aplikasi di satu tempat.

TRANSITIONS: dict[tuple[AppState, EventType], AppState] = {
    # Jalur vision
    (AppState.IDLE, EventType.PERSON_DETECTED): AppState.DETECTING,
    (AppState.DETECTING, EventType.CANDIDATE_STABLE): AppState.TARGET_LOCKED,
    (AppState.DETECTING, EventType.CANDIDATE_LOST): AppState.IDLE,
    (AppState.TARGET_LOCKED, EventType.LOCK_CONFIRMED): AppState.COLOR_STABILIZING,
    (AppState.COLOR_STABILIZING, EventType.COLOR_READY): AppState.GREETING,
    (AppState.COLOR_STABILIZING, EventType.COLOR_TIMEOUT): AppState.GREETING,

    # Jalur percakapan
    (AppState.GREETING, EventType.SPEECH_DONE): AppState.WAITING_FOR_INPUT,
    (AppState.WAITING_FOR_INPUT, EventType.USER_MESSAGE): AppState.RETRIEVING,
    (AppState.RETRIEVING, EventType.CONTEXT_READY): AppState.GENERATING,
    (AppState.GENERATING, EventType.ANSWER_READY): AppState.SPEAKING,
    (AppState.SPEAKING, EventType.SPEECH_DONE): AppState.WAITING_FOR_INPUT,

    # Penutupan sesi
    (AppState.WAITING_FOR_INPUT, EventType.IDLE_TIMEOUT): AppState.SESSION_ENDING,
    (AppState.WAITING_FOR_INPUT, EventType.FAREWELL): AppState.SESSION_ENDING,
    (AppState.SESSION_ENDING, EventType.CLEANUP_DONE): AppState.COOLDOWN,
    (AppState.COOLDOWN, EventType.COOLDOWN_EXPIRED): AppState.IDLE,

    # Pemulihan error. RECOVERY_FAILED kembali ke state yang sama = "coba lagi"
    # (batas percobaan diperiksa di _transition, bukan di tabel).
    (AppState.ERROR_RECOVERY, EventType.RECOVERY_DONE): AppState.IDLE,
    (AppState.ERROR_RECOVERY, EventType.RECOVERY_FAILED): AppState.ERROR_RECOVERY,
}

# Aturan 16.3 #5: TARGET_LOST berlaku di state mana pun SETELAH lock, dan
# selalu menuju SESSION_ENDING.
_STATES_AFTER_LOCK: tuple[AppState, ...] = (
    AppState.TARGET_LOCKED,
    AppState.COLOR_STABILIZING,
    AppState.GREETING,
    AppState.WAITING_FOR_INPUT,
    AppState.RETRIEVING,
    AppState.GENERATING,
    AppState.SPEAKING,
)
for _state in _STATES_AFTER_LOCK:
    TRANSITIONS[(_state, EventType.TARGET_LOST)] = AppState.SESSION_ENDING

# Bagian 16.1: dari state mana pun, ERROR -> ERROR_RECOVERY.
# (Kecuali saat sudah di ERROR_RECOVERY: ia sedang memulihkan diri.)
for _state in AppState:
    if _state is not AppState.ERROR_RECOVERY:
        TRANSITIONS[(_state, EventType.ERROR)] = AppState.ERROR_RECOVERY
del _state


# =============================================================================
# 3. TIMEOUT -- apa yang terjadi bila sebuah state terlalu lama
# =============================================================================
# Timeout diperlakukan sebagai EVENT BUATAN SENDIRI oleh mesin, lalu diproses
# lewat jalur handle() yang sama dengan event biasa. Jadi logging dan aturan
# transisinya tetap satu tempat. Durasinya ada di config.yaml
# (application.state_timeouts_sec), BUKAN di sini (Aturan #10).
#
# IDLE sengaja tidak ada: ia menunggu orang muncul, itu memang boleh selamanya.

TIMEOUT_EVENTS: dict[AppState, tuple[EventType, dict[str, Any]]] = {
    AppState.DETECTING: (EventType.CANDIDATE_LOST, {}),
    AppState.TARGET_LOCKED: (EventType.TARGET_LOST, {}),
    AppState.COLOR_STABILIZING: (EventType.COLOR_TIMEOUT, {}),
    AppState.GREETING: (EventType.SPEECH_DONE, {"degraded": True}),
    AppState.WAITING_FOR_INPUT: (EventType.IDLE_TIMEOUT, {}),
    AppState.RETRIEVING: (EventType.CONTEXT_READY, {"degraded": True}),
    AppState.GENERATING: (EventType.ANSWER_READY, {"degraded": True}),
    AppState.SPEAKING: (EventType.SPEECH_DONE, {"degraded": True}),
    AppState.SESSION_ENDING: (EventType.CLEANUP_DONE, {}),
    AppState.COOLDOWN: (EventType.COOLDOWN_EXPIRED, {}),
    AppState.ERROR_RECOVERY: (EventType.RECOVERY_FAILED, {}),
}

# Alasan sesi berakhir, ditentukan oleh event yang membawa kita ke SESSION_ENDING.
_END_REASON_BY_EVENT: dict[EventType, EndReason] = {
    EventType.IDLE_TIMEOUT: EndReason.TIMEOUT,
    EventType.FAREWELL: EndReason.FAREWELL,
    EventType.TARGET_LOST: EndReason.TARGET_LOST,
}


# =============================================================================
# 4. MESIN KEADAAN
# =============================================================================

class StateMachine:
    """Pemegang tunggal state aplikasi.

    Args:
        timeouts: isi `application.state_timeouts_sec` dari config.yaml.
            Kuncinya nama state huruf kecil ("color_stabilizing", dst). Setiap
            state selain IDLE WAJIB punya timeout (Aturan 16.3 #4).
        max_recovery_failures: berapa kali pemulihan boleh gagal berturut-turut
            sebelum aplikasi dihentikan (`application.error_recovery`).
        now: waktu awal mesin, dalam jam yang sama dengan timestamp Event.
    """

    def __init__(
        self,
        timeouts: Mapping[str, float],
        max_recovery_failures: int,
        *,
        now: float = 0.0,
    ) -> None:
        self._timeouts = self._validate_timeouts(timeouts)
        if max_recovery_failures < 1:
            raise ValueError("max_recovery_failures harus >= 1")
        self._max_recovery_failures = max_recovery_failures

        # --- SATU-SATUNYA tempat state disimpan ---
        self._state: AppState = AppState.IDLE
        self._entered_at: float = now
        self._now: float = now  # waktu terbaru yang pernah dilihat mesin

        # Konteks kecil milik mesin sendiri
        self._end_reason: EndReason | None = None
        self._recovery_failures: int = 0
        self._last_error: dict[str, Any] = {}
        self._stopped: bool = False

    # ------------------------------------------------------------------ util

    @staticmethod
    def _validate_timeouts(timeouts: Mapping[str, float]) -> dict[AppState, float]:
        """Pastikan setiap state non-IDLE punya timeout positif (fail fast)."""
        result: dict[AppState, float] = {}
        for state in AppState:
            if state is AppState.IDLE:
                continue
            key = state.value.lower()
            if key not in timeouts:
                raise ValueError(
                    f"config 'state_timeouts_sec' kekurangan kunci '{key}' "
                    "(setiap state selain IDLE wajib punya timeout)"
                )
            value = timeouts[key]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"timeout '{key}' harus berupa angka > 0, bukan {value!r}"
                )
            result[state] = float(value)
        return result

    @staticmethod
    def _parse_type(event: Event) -> EventType | None:
        try:
            return EventType(event.type)
        except ValueError:
            return None

    # ------------------------------------------------------ antarmuka publik

    @property
    def current_state(self) -> AppState:
        return self._state

    @property
    def is_stopped(self) -> bool:
        """True bila pemulihan error gagal berkali-kali dan app harus berhenti."""
        return self._stopped

    def time_in_state(self, now: float | None = None) -> float:
        """Sudah berapa detik mesin berada di state saat ini."""
        reference = self._now if now is None else now
        return reference - self._entered_at

    def can_transition(self, to: AppState) -> bool:
        """Apakah dari state saat ini ada SEBUAH event yang menuju `to`."""
        return any(
            source is self._state and destination is to
            for (source, _), destination in TRANSITIONS.items()
        )

    def handle(self, event: Event) -> list[Action]:
        """Proses satu event. Mengembalikan daftar aksi (bisa kosong).

        Tidak pernah melempar exception karena event yang salah alamat:
        event tak dikenal atau transisi terlarang cukup dicatat WARNING.
        """
        etype = self._parse_type(event)
        if etype is None:
            logger.warning(
                "Event dengan jenis tak dikenal diabaikan: %r (state=%s)",
                event.type, self._state.value,
            )
            return []

        if self._stopped:
            logger.warning(
                "Event %s diabaikan: aplikasi sudah dihentikan", etype.value
            )
            return []

        self._now = max(self._now, event.timestamp)

        destination = TRANSITIONS.get((self._state, etype))
        if destination is None:
            logger.warning(
                "Event %s diabaikan: tidak ada transisi dari state %s",
                etype.value, self._state.value,
            )
            return []

        return self._transition(destination, etype, event)

    def tick(self, now: float) -> list[Action]:
        """Periksa timeout. Dipanggil Orchestrator di setiap putaran loop.

        Paling banyak satu transisi per panggilan.
        """
        self._now = max(self._now, now)

        if self._stopped or self._state is AppState.IDLE:
            return []

        limit = self._timeouts[self._state]
        if self.time_in_state() < limit:
            return []

        etype, extra = TIMEOUT_EVENTS[self._state]
        logger.info(
            "Timeout %.1f dtk di state %s -> memicu event %s",
            limit, self._state.value, etype.value,
        )
        return self.handle(make_event(etype, now, {**extra, "timeout": True}))

    # ------------------------------------------------------------ transisi

    def _transition(
        self, destination: AppState, etype: EventType, event: Event
    ) -> list[Action]:
        origin = self._state

        # --- Pembukuan yang bergantung pada jenis event ---
        if etype is EventType.RECOVERY_FAILED:
            self._recovery_failures += 1
            if self._recovery_failures >= self._max_recovery_failures:
                self._stopped = True
                logger.error(
                    "Pemulihan gagal %d kali berturut-turut: aplikasi harus berhenti",
                    self._recovery_failures,
                )
                return [
                    Action(
                        ActionType.STOP_APPLICATION,
                        {
                            "reason": "recovery_failed",
                            "failures": self._recovery_failures,
                            "last_error": dict(self._last_error),
                        },
                    )
                ]
        elif etype is EventType.RECOVERY_DONE:
            self._recovery_failures = 0
        elif etype is EventType.ERROR:
            self._last_error = {"from_state": origin.value, **event.payload}

        if destination is AppState.SESSION_ENDING:
            self._end_reason = _END_REASON_BY_EVENT[etype]

        # --- Keluar dari state lama, pindah, masuk ke state baru ---
        actions = self._exit_actions(origin)
        self._state = destination
        self._entered_at = self._now

        logger.info(
            "STATE %s -> %s (event=%s%s)",
            origin.value, destination.value, etype.value,
            ", timeout" if event.payload.get("timeout") else "",
        )

        actions.extend(self._entry_actions(destination, event))
        return actions

    def _exit_actions(self, state: AppState) -> list[Action]:
        """Aksi keluar (kolom 'Exit' di Bagian 16.2)."""
        if state is AppState.GREETING:
            return [Action(ActionType.START_SESSION)]
        if state is AppState.SPEAKING:
            return [Action(ActionType.TOUCH_ACTIVITY)]
        if state is AppState.SESSION_ENDING:
            reason = (
                ReleaseReason.TARGET_LOST
                if self._end_reason is EndReason.TARGET_LOST
                else ReleaseReason.SESSION_ENDED
            )
            return [Action(ActionType.RELEASE_LOCK, {"reason": reason.value})]
        if state is AppState.COOLDOWN:
            return [Action(ActionType.PURGE_EXPIRED_COOLDOWNS)]
        return []

    def _entry_actions(self, state: AppState, event: Event) -> list[Action]:
        """Aksi masuk (kolom 'Entry' di Bagian 16.2)."""
        payload = dict(event.payload)

        if state is AppState.IDLE:
            return [
                Action(ActionType.CLEAR_TARGET),
                Action(ActionType.RESET_DETECTION),
                Action(ActionType.RESET_COLOR_STABILIZER),
                Action(ActionType.PURGE_EXPIRED_COOLDOWNS),
            ]
        if state is AppState.DETECTING:
            return [Action(ActionType.START_CONFIRM_TIMER)]
        if state is AppState.COLOR_STABILIZING:
            return [Action(ActionType.RESET_COLOR_STABILIZER)]
        if state is AppState.GREETING:
            # color = None berarti warna gagal -> Orchestrator memakai template cadangan
            return [
                Action(ActionType.BUILD_GREETING, {"color": payload.get("color")}),
                Action(ActionType.MARK_GREETED),
            ]
        if state is AppState.WAITING_FOR_INPUT:
            return [Action(ActionType.TOUCH_ACTIVITY)]
        if state is AppState.RETRIEVING:
            return [
                Action(ActionType.SHOW_PROGRESS_INDICATOR),
                Action(ActionType.RUN_RETRIEVAL, {"text": payload.get("text", "")}),
            ]
        if state is AppState.GENERATING:
            return [Action(ActionType.RUN_LLM, payload)]
        if state is AppState.SPEAKING:
            return [Action(ActionType.SPEAK_ANSWER, payload)]
        if state is AppState.SESSION_ENDING:
            reason = self._end_reason.value if self._end_reason else None
            return [Action(ActionType.ARCHIVE_SESSION, {"reason": reason})]
        if state is AppState.COOLDOWN:
            return [Action(ActionType.REGISTER_COOLDOWN)]
        if state is AppState.ERROR_RECOVERY:
            return [
                Action(ActionType.LOG_ERROR, dict(self._last_error)),
                Action(
                    ActionType.ATTEMPT_RECOVERY,
                    {"attempt": self._recovery_failures + 1},
                ),
            ]
        return []  # TARGET_LOCKED: tidak ada aksi masuk yang perlu dijalankan
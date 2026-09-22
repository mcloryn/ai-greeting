"""
tests/unit/test_state_machine.py

Pengujian StateMachine (CP07). Semuanya memakai event buatan dan waktu buatan:
tanpa kamera, tanpa internet, tanpa time.sleep().

Isi pengujian:
  1. Validasi konstruksi + kelengkapan config
  2. Setiap baris tabel transisi
  3. Setiap pasangan (state, event) yang TERLARANG ditolak dengan rapi
  4. Satu siklus penuh IDLE -> ... -> IDLE
  5. Setiap jalur timeout
  6. TARGET_LOST dari setiap state pasca-lock
  7. Jalur error dan pemulihan
  8. Tidak ada state buntu
  9. Module bebas dependensi eksternal
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import pytest
import yaml

from src.application import state_machine as sm_module
from src.application.state_machine import (
    TIMEOUT_EVENTS,
    TRANSITIONS,
    ActionType,
    StateMachine,
)
from src.core.enums import AppState
from src.core.events import REQUIRED_PAYLOAD_KEYS, EventType, make_event
from src.core.models import Event

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


# =============================================================================
# Fixture dan pembantu
# =============================================================================

@pytest.fixture(scope="module")
def app_config() -> dict:
    """Bagian `application` dari config.yaml yang ASLI."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["application"]


@pytest.fixture
def timeouts(app_config) -> dict:
    return app_config["state_timeouts_sec"]


@pytest.fixture
def machine(app_config) -> StateMachine:
    return StateMachine(
        app_config["state_timeouts_sec"],
        app_config["error_recovery"]["max_consecutive_failures"],
        now=0.0,
    )


def ev(etype: EventType, t: float = 0.0, **payload) -> Event:
    """Buat event valid; kunci wajib payload diisi nilai dummy."""
    data = {key: "x" for key in REQUIRED_PAYLOAD_KEYS[etype]}
    data.update(payload)
    return make_event(etype, t, data)


def action_types(actions) -> list[ActionType]:
    return [a.type for a in actions]


def find(actions, atype: ActionType):
    """Ambil aksi pertama bertipe `atype` (atau None)."""
    return next((a for a in actions if a.type is atype), None)


# Jalur bahagia dari IDLE; potongan awalnya membawa mesin ke tiap state.
HAPPY = [
    EventType.PERSON_DETECTED,   # -> DETECTING
    EventType.CANDIDATE_STABLE,  # -> TARGET_LOCKED
    EventType.LOCK_CONFIRMED,    # -> COLOR_STABILIZING
    EventType.COLOR_READY,       # -> GREETING
    EventType.SPEECH_DONE,       # -> WAITING_FOR_INPUT
    EventType.USER_MESSAGE,      # -> RETRIEVING
    EventType.CONTEXT_READY,     # -> GENERATING
    EventType.ANSWER_READY,      # -> SPEAKING
]

PATH_TO_STATE: dict[AppState, list[EventType]] = {
    AppState.IDLE: [],
    AppState.DETECTING: HAPPY[:1],
    AppState.TARGET_LOCKED: HAPPY[:2],
    AppState.COLOR_STABILIZING: HAPPY[:3],
    AppState.GREETING: HAPPY[:4],
    AppState.WAITING_FOR_INPUT: HAPPY[:5],
    AppState.RETRIEVING: HAPPY[:6],
    AppState.GENERATING: HAPPY[:7],
    AppState.SPEAKING: HAPPY[:8],
    AppState.SESSION_ENDING: HAPPY[:5] + [EventType.IDLE_TIMEOUT],
    AppState.COOLDOWN: HAPPY[:5] + [EventType.IDLE_TIMEOUT, EventType.CLEANUP_DONE],
    AppState.ERROR_RECOVERY: [EventType.ERROR],
}

TIMED_STATES = [s for s in AppState if s is not AppState.IDLE]
STATES_AFTER_LOCK = [
    AppState.TARGET_LOCKED, AppState.COLOR_STABILIZING, AppState.GREETING,
    AppState.WAITING_FOR_INPUT, AppState.RETRIEVING, AppState.GENERATING,
    AppState.SPEAKING,
]

EXPECTED_TIMEOUT_DEST = {
    AppState.DETECTING: AppState.IDLE,
    AppState.TARGET_LOCKED: AppState.SESSION_ENDING,
    AppState.COLOR_STABILIZING: AppState.GREETING,
    AppState.GREETING: AppState.WAITING_FOR_INPUT,
    AppState.WAITING_FOR_INPUT: AppState.SESSION_ENDING,
    AppState.RETRIEVING: AppState.GENERATING,
    AppState.GENERATING: AppState.SPEAKING,
    AppState.SPEAKING: AppState.WAITING_FOR_INPUT,
    AppState.SESSION_ENDING: AppState.COOLDOWN,
    AppState.COOLDOWN: AppState.IDLE,
    AppState.ERROR_RECOVERY: AppState.ERROR_RECOVERY,  # coba pulih lagi
}


def put_in_state(machine: StateMachine, state: AppState, t: float = 0.0) -> None:
    """Bawa mesin (yang baru dibuat, di IDLE) ke `state`, semua event pada waktu t."""
    for etype in PATH_TO_STATE[state]:
        machine.handle(ev(etype, t))
    assert machine.current_state is state


# =============================================================================
# 0. Kelengkapan pembantu pengujian itu sendiri
# =============================================================================

def test_path_helper_covers_every_state():
    """Bila suatu hari AppState bertambah, tes ini memaksa pembantu diperbarui."""
    assert set(PATH_TO_STATE) == set(AppState)


# =============================================================================
# 1. Konstruksi dan config
# =============================================================================

def test_config_has_timeout_for_every_non_idle_state(timeouts):
    for state in TIMED_STATES:
        assert state.value.lower() in timeouts, f"config kurang timeout {state.value}"


def test_construct_with_real_config_works(app_config):
    machine = StateMachine(
        app_config["state_timeouts_sec"],
        app_config["error_recovery"]["max_consecutive_failures"],
    )
    assert machine.current_state is AppState.IDLE
    assert machine.is_stopped is False


def test_missing_timeout_key_is_rejected(timeouts):
    broken = dict(timeouts)
    del broken["speaking"]
    with pytest.raises(ValueError, match="speaking"):
        StateMachine(broken, 3)


@pytest.mark.parametrize("bad_value", [0, -1, "10", None])
def test_invalid_timeout_value_is_rejected(timeouts, bad_value):
    broken = dict(timeouts)
    broken["greeting"] = bad_value
    with pytest.raises(ValueError, match="greeting"):
        StateMachine(broken, 3)


def test_invalid_max_recovery_failures_is_rejected(timeouts):
    with pytest.raises(ValueError):
        StateMachine(timeouts, 0)


# =============================================================================
# 2. Setiap baris tabel transisi
# =============================================================================

@pytest.mark.parametrize(
    "source,etype,destination",
    [(s, e, d) for (s, e), d in TRANSITIONS.items()],
    ids=[f"{s.value}--{e.value}-->{d.value}" for (s, e), d in TRANSITIONS.items()],
)
def test_every_transition_row_works(machine, source, etype, destination):
    put_in_state(machine, source, t=10.0)
    machine.handle(ev(etype, t=11.0))
    assert machine.current_state is destination


# =============================================================================
# 3. Transisi terlarang: ditolak dengan rapi, tanpa exception
# =============================================================================

INVALID_PAIRS = [
    (s, e) for s in AppState for e in EventType if (s, e) not in TRANSITIONS
]


@pytest.mark.parametrize(
    "state,etype",
    INVALID_PAIRS,
    ids=[f"{s.value}--{e.value}" for s, e in INVALID_PAIRS],
)
def test_forbidden_transition_is_rejected_with_warning(machine, caplog, state, etype):
    put_in_state(machine, state)
    with caplog.at_level(logging.WARNING, logger=sm_module.logger.name):
        actions = machine.handle(ev(etype, 1.0))

    assert actions == []
    assert machine.current_state is state
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_unknown_event_type_is_ignored_not_crash(machine, caplog):
    bogus = Event(type="BUKAN_EVENT", payload={}, timestamp=1.0)
    with caplog.at_level(logging.WARNING, logger=sm_module.logger.name):
        assert machine.handle(bogus) == []
    assert machine.current_state is AppState.IDLE
    assert any("tak dikenal" in r.getMessage() for r in caplog.records)


def test_make_event_rejects_missing_required_payload():
    with pytest.raises(ValueError, match="color"):
        make_event(EventType.COLOR_READY, 1.0, {})
    with pytest.raises(ValueError):
        make_event("BUKAN_EVENT", 1.0)


# =============================================================================
# 4. Satu siklus penuh IDLE -> IDLE
# =============================================================================

def test_full_cycle_idle_to_idle(machine, caplog):
    visited = [machine.current_state]

    def send(etype, t, **payload):
        actions = machine.handle(ev(etype, t, **payload))
        visited.append(machine.current_state)
        return actions

    with caplog.at_level(logging.INFO, logger=sm_module.logger.name):
        a = send(EventType.PERSON_DETECTED, 1.0)
        assert ActionType.START_CONFIRM_TIMER in action_types(a)

        send(EventType.CANDIDATE_STABLE, 2.0, track_id=7)
        a = send(EventType.LOCK_CONFIRMED, 3.0, track_id=7)
        assert ActionType.RESET_COLOR_STABILIZER in action_types(a)

        a = send(EventType.COLOR_READY, 4.0, color="merah")
        assert find(a, ActionType.BUILD_GREETING).data == {"color": "merah"}
        assert ActionType.MARK_GREETED in action_types(a)

        a = send(EventType.SPEECH_DONE, 5.0)
        assert ActionType.START_SESSION in action_types(a)  # keluar dari GREETING

        a = send(EventType.USER_MESSAGE, 6.0, text="apa saja jurusannya?")
        assert find(a, ActionType.RUN_RETRIEVAL).data == {"text": "apa saja jurusannya?"}

        a = send(EventType.CONTEXT_READY, 7.0, context=["chunk-1"])
        assert find(a, ActionType.RUN_LLM).data["context"] == ["chunk-1"]

        a = send(EventType.ANSWER_READY, 8.0, text="Jawaban dummy")
        assert find(a, ActionType.SPEAK_ANSWER).data["text"] == "Jawaban dummy"

        a = send(EventType.SPEECH_DONE, 9.0)
        assert ActionType.TOUCH_ACTIVITY in action_types(a)

        a = send(EventType.FAREWELL, 10.0)
        assert find(a, ActionType.ARCHIVE_SESSION).data == {"reason": "FAREWELL"}

        a = send(EventType.CLEANUP_DONE, 11.0)
        assert find(a, ActionType.RELEASE_LOCK).data == {"reason": "SESSION_ENDED"}
        assert ActionType.REGISTER_COOLDOWN in action_types(a)

        a = send(EventType.COOLDOWN_EXPIRED, 12.0)
        assert ActionType.PURGE_EXPIRED_COOLDOWNS in action_types(a)  # exit COOLDOWN
        assert ActionType.CLEAR_TARGET in action_types(a)             # entry IDLE
        assert ActionType.RESET_COLOR_STABILIZER in action_types(a)

    assert visited == [
        AppState.IDLE, AppState.DETECTING, AppState.TARGET_LOCKED,
        AppState.COLOR_STABILIZING, AppState.GREETING, AppState.WAITING_FOR_INPUT,
        AppState.RETRIEVING, AppState.GENERATING, AppState.SPEAKING,
        AppState.WAITING_FOR_INPUT, AppState.SESSION_ENDING, AppState.COOLDOWN,
        AppState.IDLE,
    ]
    # Setiap perpindahan tercatat di log INFO (Aturan 16.3 #2)
    transitions_logged = [r for r in caplog.records if "STATE " in r.getMessage()]
    assert len(transitions_logged) == 12


def test_two_questions_in_one_session_loop_back(machine):
    put_in_state(machine, AppState.SPEAKING)
    machine.handle(ev(EventType.SPEECH_DONE, 1.0))
    assert machine.current_state is AppState.WAITING_FOR_INPUT
    machine.handle(ev(EventType.USER_MESSAGE, 2.0, text="pertanyaan kedua"))
    assert machine.current_state is AppState.RETRIEVING


# =============================================================================
# 5. Timeout
# =============================================================================

@pytest.mark.parametrize("state", TIMED_STATES, ids=lambda s: s.value)
def test_timeout_fires_exactly_at_limit(machine, timeouts, state):
    t0 = 100.0
    put_in_state(machine, state, t0)
    limit = timeouts[state.value.lower()]

    # Sedikit sebelum batas: belum terjadi apa-apa
    assert machine.tick(t0 + limit - 0.001) == []
    assert machine.current_state is state

    # Tepat di batas: timeout memicu transisi yang diharapkan
    machine.tick(t0 + limit)
    assert machine.current_state is EXPECTED_TIMEOUT_DEST[state]
    # Timer state (baru) dimulai dari nol
    assert machine.time_in_state() == pytest.approx(0.0)


def test_every_timeout_event_has_a_valid_transition():
    """Timeout tidak boleh memicu event yang ditolak tabel."""
    assert set(TIMEOUT_EVENTS) == set(TIMED_STATES)
    for state, (etype, _) in TIMEOUT_EVENTS.items():
        assert (state, etype) in TRANSITIONS, f"timeout {state.value} menuju jalan buntu"


def test_idle_never_times_out(machine):
    assert machine.tick(1_000_000.0) == []
    assert machine.current_state is AppState.IDLE


def test_color_timeout_greets_without_color(machine, timeouts):
    put_in_state(machine, AppState.COLOR_STABILIZING, 0.0)
    actions = machine.tick(timeouts["color_stabilizing"])
    assert machine.current_state is AppState.GREETING
    assert find(actions, ActionType.BUILD_GREETING).data == {"color": None}


def test_retrieval_timeout_continues_with_degraded_context(machine, timeouts):
    put_in_state(machine, AppState.RETRIEVING, 0.0)
    actions = machine.tick(timeouts["retrieving"])
    assert machine.current_state is AppState.GENERATING
    assert find(actions, ActionType.RUN_LLM).data["degraded"] is True


def test_generation_timeout_speaks_friendly_fallback(machine, timeouts):
    put_in_state(machine, AppState.GENERATING, 0.0)
    actions = machine.tick(timeouts["generating"])
    assert machine.current_state is AppState.SPEAKING
    assert find(actions, ActionType.SPEAK_ANSWER).data["degraded"] is True


def test_waiting_timeout_ends_session_with_timeout_reason(machine, timeouts):
    put_in_state(machine, AppState.WAITING_FOR_INPUT, 0.0)
    actions = machine.tick(timeouts["waiting_for_input"])
    assert find(actions, ActionType.ARCHIVE_SESSION).data == {"reason": "TIMEOUT"}


def test_timeout_transition_is_logged_as_info(machine, timeouts, caplog):
    put_in_state(machine, AppState.COOLDOWN, 0.0)
    with caplog.at_level(logging.INFO, logger=sm_module.logger.name):
        machine.tick(timeouts["cooldown"])
    assert any("COOLDOWN -> IDLE" in r.getMessage() for r in caplog.records)


# =============================================================================
# 6. TARGET_LOST
# =============================================================================

@pytest.mark.parametrize("state", STATES_AFTER_LOCK, ids=lambda s: s.value)
def test_target_lost_from_any_state_after_lock_ends_session(machine, state):
    put_in_state(machine, state)
    actions = machine.handle(ev(EventType.TARGET_LOST, 5.0))

    assert machine.current_state is AppState.SESSION_ENDING
    assert find(actions, ActionType.ARCHIVE_SESSION).data == {"reason": "TARGET_LOST"}

    # Saat SESSION_ENDING selesai, lock dilepas dengan alasan TARGET_LOST
    actions = machine.handle(ev(EventType.CLEANUP_DONE, 6.0))
    assert find(actions, ActionType.RELEASE_LOCK).data == {"reason": "TARGET_LOST"}


@pytest.mark.parametrize(
    "state", [AppState.IDLE, AppState.DETECTING], ids=lambda s: s.value
)
def test_target_lost_before_lock_is_ignored(machine, state):
    put_in_state(machine, state)
    assert machine.handle(ev(EventType.TARGET_LOST, 5.0)) == []
    assert machine.current_state is state


# =============================================================================
# 7. Error dan pemulihan
# =============================================================================

@pytest.mark.parametrize(
    "state", [s for s in AppState if s is not AppState.ERROR_RECOVERY],
    ids=lambda s: s.value,
)
def test_error_from_any_state_goes_to_recovery(machine, state):
    put_in_state(machine, state)
    actions = machine.handle(
        ev(EventType.ERROR, 5.0, source="camera", message="kamera dicabut")
    )
    assert machine.current_state is AppState.ERROR_RECOVERY

    log_action = find(actions, ActionType.LOG_ERROR)
    assert log_action.data["source"] == "camera"
    assert log_action.data["from_state"] == state.value  # konteks error lengkap
    assert find(actions, ActionType.ATTEMPT_RECOVERY).data == {"attempt": 1}


def test_recovery_done_returns_to_idle(machine):
    put_in_state(machine, AppState.ERROR_RECOVERY)
    actions = machine.handle(ev(EventType.RECOVERY_DONE, 5.0))
    assert machine.current_state is AppState.IDLE
    assert ActionType.CLEAR_TARGET in action_types(actions)


def test_three_consecutive_recovery_failures_stop_the_application(machine):
    put_in_state(machine, AppState.ERROR_RECOVERY)

    # Gagal ke-1 dan ke-2: mencoba lagi
    a = machine.handle(ev(EventType.RECOVERY_FAILED, 1.0))
    assert machine.current_state is AppState.ERROR_RECOVERY
    assert find(a, ActionType.ATTEMPT_RECOVERY).data == {"attempt": 2}
    a = machine.handle(ev(EventType.RECOVERY_FAILED, 2.0))
    assert find(a, ActionType.ATTEMPT_RECOVERY).data == {"attempt": 3}
    assert machine.is_stopped is False

    # Gagal ke-3: berhenti dengan pesan jelas
    a = machine.handle(ev(EventType.RECOVERY_FAILED, 3.0))
    assert machine.is_stopped is True
    assert action_types(a) == [ActionType.STOP_APPLICATION]

    # Setelah berhenti, semua event diabaikan
    assert machine.handle(ev(EventType.RECOVERY_DONE, 4.0)) == []
    assert machine.tick(1_000_000.0) == []


def test_recovery_timeout_counts_as_failure(machine, timeouts):
    put_in_state(machine, AppState.ERROR_RECOVERY, 0.0)
    t = 0.0
    for _ in range(3):
        t += timeouts["error_recovery"]
        machine.tick(t)
    assert machine.is_stopped is True


def test_successful_recovery_resets_failure_counter(machine):
    put_in_state(machine, AppState.ERROR_RECOVERY)
    machine.handle(ev(EventType.RECOVERY_FAILED, 1.0))
    machine.handle(ev(EventType.RECOVERY_FAILED, 2.0))
    machine.handle(ev(EventType.RECOVERY_DONE, 3.0))  # berhasil -> hitungan nol

    machine.handle(ev(EventType.ERROR, 4.0, source="llm", message="x"))
    machine.handle(ev(EventType.RECOVERY_FAILED, 5.0))
    machine.handle(ev(EventType.RECOVERY_FAILED, 6.0))
    assert machine.is_stopped is False  # baru 2 kegagalan berturut-turut


def test_error_while_recovering_is_ignored(machine):
    put_in_state(machine, AppState.ERROR_RECOVERY)
    assert machine.handle(ev(EventType.ERROR, 5.0, source="x", message="y")) == []
    assert machine.current_state is AppState.ERROR_RECOVERY


# =============================================================================
# 8. Tidak ada state yang dapat macet selamanya
# =============================================================================

@pytest.mark.parametrize("state", list(AppState), ids=lambda s: s.value)
def test_no_state_can_hang_forever(machine, state):
    """Dari state mana pun, bila tidak ada event sama sekali dan waktu terus
    berjalan, mesin akhirnya berada di IDLE atau berhenti dengan jelas."""
    put_in_state(machine, state, 0.0)

    t = 0.0
    for _ in range(50):
        if machine.current_state is AppState.IDLE or machine.is_stopped:
            break
        t += 1000.0  # lompatan waktu jauh melebihi timeout mana pun
        machine.tick(t)

    assert machine.current_state is AppState.IDLE or machine.is_stopped


def test_every_state_is_reachable_from_idle():
    reachable = {AppState.IDLE}
    frontier = [AppState.IDLE]
    while frontier:
        current = frontier.pop()
        for (source, _), destination in TRANSITIONS.items():
            if source is current and destination not in reachable:
                reachable.add(destination)
                frontier.append(destination)
    assert reachable == set(AppState)


# =============================================================================
# 9. time_in_state dan can_transition
# =============================================================================

def test_time_in_state_follows_event_and_tick_time(machine):
    machine.handle(ev(EventType.PERSON_DETECTED, 5.0))  # masuk DETECTING pada t=5
    assert machine.time_in_state() == pytest.approx(0.0)
    machine.tick(7.0)
    assert machine.time_in_state() == pytest.approx(2.0)
    assert machine.time_in_state(now=9.5) == pytest.approx(4.5)


def test_can_transition(machine):
    assert machine.can_transition(AppState.DETECTING) is True
    assert machine.can_transition(AppState.GREETING) is False
    assert machine.can_transition(AppState.ERROR_RECOVERY) is True  # dari mana pun
    put_in_state(machine, AppState.WAITING_FOR_INPUT)
    assert machine.can_transition(AppState.RETRIEVING) is True
    assert machine.can_transition(AppState.SESSION_ENDING) is True
    assert machine.can_transition(AppState.IDLE) is False


# =============================================================================
# 10. Module bebas dependensi eksternal
# =============================================================================

def test_state_machine_module_has_no_external_dependencies():
    """Hanya library standar Python dan src.core yang boleh di-import."""
    source = Path(sm_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)

    for name in imported:
        top = name.split(".")[0]
        allowed = top in sys.stdlib_module_names or name.startswith("src.core")
        assert allowed, f"import terlarang di state_machine.py: {name}"
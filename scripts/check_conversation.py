"""
scripts/check_conversation.py

Uji LIVE CP09: seluruh pipeline CP02-CP08 (kamera -> ... -> greeting)
ditambah chat: InputThread -> StateMachine -> ConversationManager.

RAG (CP13), LLM (CP14), TTS (CP15) belum ada -- balasan dummy dan "bicara"
dianggap selesai seketika. Ini SENGAJA, supaya threading + siklus sesi bisa
diuji terpisah dari kebenaran jawaban.

Jalankan:  python scripts/check_conversation.py     (tekan 'q' untuk keluar)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from config.settings import ConfigError, load_settings
from src.application.conversation_manager import ConversationManager
from src.application.greeting_manager import GreetingManager
from src.application.session import is_farewell
from src.application.state_machine import Action, ActionType, StateMachine
from src.core.enums import AppState, LockPhase
from src.core.events import EventType, make_event
from src.core.exceptions import CameraError, DetectorError
from src.core.models import Event, LockState
from src.ui.input_thread import InputThread
from src.vision.camera import Camera
from src.vision.clothing_color import detect_clothing_color
from src.vision.color_stabilizer import ColorStabilizer
from src.vision.detector import PersonDetector
from src.vision.target_lock import TargetLock
from src.vision.target_selector import score_candidates, select_target
from src.vision.tracker import Tracker
from src.vision.visualizer import draw_lock_status, draw_tracked_people

_MAX_RECOVERY_FAILURES = 3
_MAX_SYNC_STEPS = 4

_STATES_AFTER_LOCK = (
    AppState.TARGET_LOCKED, AppState.COLOR_STABILIZING, AppState.GREETING,
    AppState.WAITING_FOR_INPUT, AppState.RETRIEVING, AppState.GENERATING,
    AppState.SPEAKING,
)
_GREETING_VISIBLE_STATES = (AppState.GREETING, AppState.WAITING_FOR_INPUT)

_WINDOW = "check_conversation  (q = keluar)"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_CYAN = (255, 255, 0)
_PANEL_BG = (30, 30, 30)
_MARGIN = 10
_LINE_HEIGHT = 26
_TEXT_SCALE = 0.6
_TEXT_THICKNESS = 2


@dataclass
class _Ui:
    greeting_text: str | None = None
    greeted_track_id: int | None = None
    greeted_color: str | None = None      # dipakai start_session() saat exit GREETING
    color_line: str = "Warna: -"
    last_answer: str = ""


# --------------------------------------------------------------------------
# Event dari kondisi TargetLock (sama seperti CP08)
# --------------------------------------------------------------------------

def _next_event(state: AppState, lock: LockState, now: float) -> Event | None:
    phase = lock.state
    if state is AppState.IDLE and phase in (LockPhase.CANDIDATE, LockPhase.LOCKED):
        return make_event(EventType.PERSON_DETECTED, now)
    if state is AppState.DETECTING:
        if phase is LockPhase.LOCKED:
            return make_event(EventType.CANDIDATE_STABLE, now, {"track_id": lock.track_id})
        if phase is LockPhase.IDLE:
            return make_event(EventType.CANDIDATE_LOST, now)
    if state is AppState.TARGET_LOCKED and phase is LockPhase.LOCKED:
        return make_event(EventType.LOCK_CONFIRMED, now, {"track_id": lock.track_id})
    if state in _STATES_AFTER_LOCK and phase is LockPhase.IDLE:
        return make_event(EventType.TARGET_LOST, now)
    return None


def _sync_machine(machine: StateMachine, lock: LockState, now: float) -> list[Action]:
    actions: list[Action] = []
    for _ in range(_MAX_SYNC_STEPS):
        event = _next_event(machine.current_state, lock, now)
        if event is None:
            break
        actions.extend(machine.handle(event))
    return actions


# --------------------------------------------------------------------------
# Menjalankan aksi (peran Orchestrator). Aksi CP09 bisa memicu event BARU
# (mis. RUN_RETRIEVAL dummy langsung "selesai" -> CONTEXT_READY), jadi
# fungsi ini mengembalikan Event susulan yang harus diproses lagi.
# --------------------------------------------------------------------------

def _execute(
    actions: list[Action],
    machine: StateMachine,
    greeter: GreetingManager,
    conv: ConversationManager,
    stabilizer: ColorStabilizer,
    lock: LockState,
    now: float,
    ui: _Ui,
) -> None:
    followups: list[Event] = []

    for action in actions:
        if action.type is ActionType.RESET_COLOR_STABILIZER:
            stabilizer.reset()
            ui.color_line = "Warna: -"

        elif action.type is ActionType.BUILD_GREETING:
            color = action.data.get("color")
            track_id = lock.track_id
            if track_id is not None and greeter.should_greet(track_id, now):
                ui.greeting_text = greeter.build(color, now)
                ui.greeted_track_id = track_id
                ui.greeted_color = color
                print(f"[SAPAAN] track {track_id} | warna={color or 'None -> cadangan'}\n         \"{ui.greeting_text}\"")
            else:
                ui.greeting_text = None
                ui.greeted_track_id = None

        elif action.type is ActionType.MARK_GREETED:
            if ui.greeted_track_id is not None:
                greeter.mark_greeted(ui.greeted_track_id, now)
                ui.greeted_track_id = None

        elif action.type is ActionType.START_SESSION:
            conv.start_session(lock.track_id, ui.greeted_color)
            print(f"[SESI] Dimulai untuk track {lock.track_id}")

        elif action.type is ActionType.TOUCH_ACTIVITY:
            conv.touch_activity()

        elif action.type is ActionType.RUN_RETRIEVAL:
            text = action.data.get("text", "")
            conv.receive_question(text)
            # Dummy: retrieval dianggap selesai seketika (RAG asli CP13)
            followups.append(make_event(EventType.CONTEXT_READY, now))

        elif action.type is ActionType.RUN_LLM:
            answer = conv.build_dummy_answer()
            ui.last_answer = answer
            # Dummy: LLM dianggap selesai seketika (LLM asli CP14)
            followups.append(make_event(EventType.ANSWER_READY, now, {"text": answer}))

        elif action.type is ActionType.SPEAK_ANSWER:
            text = action.data.get("text", "")
            print(f"[SISTEM] {text}")
            # Dummy: TTS dianggap selesai seketika (TTS asli CP15)
            followups.append(make_event(EventType.SPEECH_DONE, now))

        elif action.type is ActionType.ARCHIVE_SESSION:
            conv.archive_session(action.data.get("reason"))
            print(f"[SESI] Berakhir, alasan={action.data.get('reason')}")

        # Aksi lain (CLEAR_TARGET, RESET_DETECTION, PURGE_EXPIRED_COOLDOWNS,
        # REGISTER_COOLDOWN, RELEASE_LOCK, dst.) belum butuh tindakan visual
        # di skrip ini.

    # Proses event susulan sampai habis (maksimal beberapa langkah, mencegah
    # loop tak berujung kalau ada bug transisi).
    for _ in range(_MAX_SYNC_STEPS):
        if not followups:
            break
        event = followups.pop(0)
        new_actions = machine.handle(event)
        _execute(new_actions, machine, greeter, conv, stabilizer, lock, now, ui)


# --------------------------------------------------------------------------
# Tampilan (sama pola dengan CP08)
# --------------------------------------------------------------------------

def _wrap_text(text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        (width, _), _ = cv2.getTextSize(trial, _FONT, _TEXT_SCALE, _TEXT_THICKNESS)
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_panel(image, text: str):
    height, width = image.shape[:2]
    lines = _wrap_text(text, width - 2 * _MARGIN)
    panel_height = len(lines) * _LINE_HEIGHT + 2 * _MARGIN
    top = height - panel_height
    overlay = image.copy()
    cv2.rectangle(overlay, (0, top), (width, height), _PANEL_BG, thickness=-1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    for i, line in enumerate(lines):
        y = top + _MARGIN + (i + 1) * _LINE_HEIGHT - 8
        cv2.putText(image, line, (_MARGIN, y), _FONT, _TEXT_SCALE, _CYAN, _TEXT_THICKNESS)
    return image


def _show_state_changes() -> None:
    sm_logger = logging.getLogger("src.application.state_machine")
    sm_logger.setLevel(logging.INFO)
    sm_logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    sm_logger.addHandler(handler)


# --------------------------------------------------------------------------

def main() -> int:
    try:
        settings = load_settings()
        greeter = GreetingManager(settings.application.greeting, settings.vision.cooldown_sec)
    except (ConfigError, ValueError) as exc:
        print(f"Konfigurasi bermasalah: {exc}")
        return 1

    conv = ConversationManager(settings.application.max_history_turns)
    farewell_words = settings.application.farewell_words

    color_cfg = settings.vision.clothing_color
    detector = PersonDetector(settings.vision)
    tracker = Tracker()
    lock = TargetLock(settings.vision)
    stabilizer = ColorStabilizer(color_cfg)
    machine = StateMachine(
        settings.application.state_timeouts_sec, _MAX_RECOVERY_FAILURES, now=time.time()
    )
    ui = _Ui()

    _show_state_changes()

    input_thread = InputThread()
    input_thread.start()

    try:
        detector.load_model()
        detector.warmup()
        with Camera(settings.vision.camera) as cam:
            print("Kamera siap. Setelah disapa, ketik pertanyaan di terminal ini. Tekan 'q' di jendela video untuk keluar.")
            while True:
                frame = cam.read()
                if frame is None:
                    print("Kamera gagal dibaca (terputus?). Berhenti.")
                    break
                now = frame.timestamp

                # --- Vision ---
                detections = detector.detect_and_track(frame)
                people = tracker.update(detections)
                candidates = score_candidates(people, frame.width, frame.height, settings.vision)
                candidate_id = select_target(candidates)
                lock_state = lock.update(candidate_id, people, now)

                # --- Timeout & sinkronisasi state dari kondisi vision ---
                _execute(machine.tick(now), machine, greeter, conv, stabilizer, lock_state, now, ui)
                _execute(_sync_machine(machine, lock_state, now), machine, greeter, conv, stabilizer, lock_state, now, ui)

                # --- Warna (CP06/CP08, tidak berubah) ---
                if machine.current_state is AppState.COLOR_STABILIZING and lock_state.state is LockPhase.LOCKED and lock_state.bbox is not None:
                    result = detect_clothing_color(frame, lock_state.bbox, color_cfg, lock_state.mask_polygon)
                    stabilizer.add(result.color_name, now)
                    stable = stabilizer.get_stable_color()
                    raw = result.color_name.value if result.color_name else "None"
                    fixed = stable.color_name.value if stable.color_name else "-"
                    ui.color_line = f"Warna mentah:{raw}  stabil:{fixed} ({stable.ratio:.0%}, n={stable.sample_count})"
                    buffer_full = machine.time_in_state(now) >= color_cfg.stabilizer_buffer_sec
                    if buffer_full and stable.is_stable and stable.color_name is not None:
                        actions = machine.handle(make_event(EventType.COLOR_READY, now, {"color": stable.color_name.value}))
                        _execute(actions, machine, greeter, conv, stabilizer, lock_state, now, ui)

                # --- Input chat: hanya berarti saat menunggu jawaban user ---
                text = input_thread.get_nowait()
                if text is not None and machine.current_state is AppState.WAITING_FOR_INPUT:
                    if is_farewell(text, farewell_words):
                        actions = machine.handle(make_event(EventType.FAREWELL, now))
                    else:
                        actions = machine.handle(make_event(EventType.USER_MESSAGE, now, {"text": text}))
                    _execute(actions, machine, greeter, conv, stabilizer, lock_state, now, ui)
                elif text is not None:
                    print(f"[SISTEM] (diabaikan, belum siap menerima pertanyaan: state={machine.current_state.value})")

                # --- Tampilan ---
                highlight_id = lock_state.track_id if lock_state.track_id is not None else candidate_id
                display = draw_tracked_people(frame.image, people, candidates, highlight_id)
                display = draw_lock_status(display, lock_state)
                cv2.putText(display, f"FPS:{cam.current_fps():.1f}  App:{machine.current_state.value}",
                            (_MARGIN, 25), _FONT, 0.7, _WHITE, 2)
                cv2.putText(display, ui.color_line, (_MARGIN, 90), _FONT, 0.55, _WHITE, 2)
                panel_text = ui.last_answer if machine.current_state in (AppState.SPEAKING, AppState.WAITING_FOR_INPUT) else ui.greeting_text
                if panel_text and machine.current_state in _GREETING_VISIBLE_STATES + (AppState.SPEAKING,):
                    display = _draw_panel(display, panel_text)

                cv2.imshow(_WINDOW, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (CameraError, DetectorError) as exc:
        print(f"Gagal: {exc}")
        return 1
    finally:
        input_thread.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
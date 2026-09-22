"""
scripts/check_greeting_live.py

Uji LIVE CP08: kamera -> YOLO -> tracker -> selector -> lock -> warna ->
StateMachine -> GreetingManager. Sapaan tampil di jendela video dan konsol.

Skrip ini meniru peran Orchestrator (CP16) dalam bentuk paling sederhana.
TTS dan chat belum ada, jadi state setelah GREETING berjalan lewat timeout
StateMachine saja.

Jalankan:  python scripts/check_greeting_live.py     (tekan 'q' untuk keluar)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from config.settings import ConfigError, load_settings
from src.application.greeting_manager import GreetingManager
from src.application.state_machine import Action, ActionType, StateMachine
from src.core.enums import AppState, LockPhase
from src.core.events import EventType, make_event
from src.core.exceptions import CameraError, DetectorError
from src.core.models import Event, LockState
from src.vision.camera import Camera
from src.vision.clothing_color import detect_clothing_color
from src.vision.color_stabilizer import ColorStabilizer
from src.vision.detector import PersonDetector
from src.vision.target_lock import TargetLock
from src.vision.target_selector import score_candidates, select_target
from src.vision.tracker import Tracker
from src.vision.visualizer import draw_lock_status, draw_tracked_people

# Sementara: application.error_recovery di config.yaml belum dimuat oleh Settings
# (dibereskan di CP16). Skrip ini tidak pernah mengirim event ERROR.
_MAX_RECOVERY_FAILURES = 3

# Iterasi maksimum sinkronisasi per frame: IDLE -> DETECTING -> TARGET_LOCKED
# -> COLOR_STABILIZING adalah 3 langkah, +1 sebagai pengaman.
_MAX_SYNC_STEPS = 4

_STATES_AFTER_LOCK = (
    AppState.TARGET_LOCKED, AppState.COLOR_STABILIZING, AppState.GREETING,
    AppState.WAITING_FOR_INPUT, AppState.RETRIEVING, AppState.GENERATING,
    AppState.SPEAKING,
)
_GREETING_VISIBLE_STATES = (AppState.GREETING, AppState.WAITING_FOR_INPUT)

# Tampilan (kosmetik, bukan parameter perilaku)
_WINDOW = "check_greeting_live  (q = keluar)"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_CYAN = (255, 255, 0)
_PANEL_BG = (30, 30, 30)
_MARGIN = 10
_LINE_HEIGHT = 26
_GREETING_SCALE = 0.6
_GREETING_THICKNESS = 2


@dataclass
class _Ui:
    """Data tampilan/sapaan yang berubah antar-frame."""

    greeting_text: str | None = None
    greeted_track_id: int | None = None   # dari BUILD_GREETING, dipakai MARK_GREETED
    color_line: str = "Warna: -"


# --------------------------------------------------------------------------
# Event dari kondisi TargetLock (dicek tiap frame, bukan hanya saat berubah)
# --------------------------------------------------------------------------

def _next_event(state: AppState, lock: LockState, now: float) -> Event | None:
    """Event berikutnya yang sah untuk state mesin sekarang, atau None.

    Dibuat berbasis KONDISI (bukan hanya perubahan) supaya bila mesin sedang
    di COOLDOWN saat seseorang sudah terkunci, orang itu tetap diproses begitu
    mesin kembali ke IDLE.
    """
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
# Menjalankan aksi dari StateMachine (peran Orchestrator)
# --------------------------------------------------------------------------

def _execute(
    actions: list[Action],
    greeter: GreetingManager,
    stabilizer: ColorStabilizer,
    lock: LockState,
    now: float,
    ui: _Ui,
) -> None:
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
                label = color if color else "None -> sapaan cadangan"
                print(f"[SAPAAN] track {track_id} | warna={label}\n         \"{ui.greeting_text}\"")
            else:
                ui.greeting_text = None
                ui.greeted_track_id = None
                print(f"[SAPAAN] dilewati (track_id={track_id}, tidak ada / masih cooldown)")

        elif action.type is ActionType.MARK_GREETED:
            if ui.greeted_track_id is not None:
                greeter.mark_greeted(ui.greeted_track_id, now)
                ui.greeted_track_id = None
        # Aksi lain (START_SESSION, ARCHIVE_SESSION, RELEASE_LOCK, dst.) belum
        # punya padanan di CP08; state berikutnya berjalan lewat timeout mesin.


def _color_step(
    machine: StateMachine,
    lock: LockState,
    frame,
    now: float,
    color_cfg,
    stabilizer: ColorStabilizer,
    greeter: GreetingManager,
    ui: _Ui,
) -> None:
    """Saat COLOR_STABILIZING: ambil warna per frame, stabilkan, kirim COLOR_READY.

    COLOR_TIMEOUT diurus mesin sendiri (timeout state color_stabilizing).
    """
    if machine.current_state is not AppState.COLOR_STABILIZING:
        return
    if lock.state is not LockPhase.LOCKED or lock.bbox is None:
        return

    result = detect_clothing_color(frame, lock.bbox, color_cfg, lock.mask_polygon)
    stabilizer.add(result.color_name, now)
    stable = stabilizer.get_stable_color()

    raw = result.color_name.value if result.color_name else "None"
    fixed = stable.color_name.value if stable.color_name else "-"
    ui.color_line = (
        f"Warna mentah:{raw}  stabil:{fixed} "
        f"({stable.ratio:.0%}, n={stable.sample_count})"
    )

    # Jangan terima warna sebelum buffer terisi penuh (Failure Case Bagian 4.6:
    # "buffer belum penuh saat greeting hendak dipicu"). Satu sampel pertama
    # akan selalu terlihat "100% stabil".
    buffer_full = machine.time_in_state(now) >= color_cfg.stabilizer_buffer_sec
    if buffer_full and stable.is_stable and stable.color_name is not None:
        actions = machine.handle(
            make_event(EventType.COLOR_READY, now, {"color": stable.color_name.value})
        )
        _execute(actions, greeter, stabilizer, lock, now, ui)


# --------------------------------------------------------------------------
# Tampilan
# --------------------------------------------------------------------------

def _wrap_text(text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        (width, _), _ = cv2.getTextSize(trial, _FONT, _GREETING_SCALE, _GREETING_THICKNESS)
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_greeting(image, text: str):
    height, width = image.shape[:2]
    lines = _wrap_text(text, width - 2 * _MARGIN)
    panel_height = len(lines) * _LINE_HEIGHT + 2 * _MARGIN
    top = height - panel_height

    overlay = image.copy()
    cv2.rectangle(overlay, (0, top), (width, height), _PANEL_BG, thickness=-1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    for i, line in enumerate(lines):
        y = top + _MARGIN + (i + 1) * _LINE_HEIGHT - 8
        cv2.putText(image, line, (_MARGIN, y), _FONT, _GREETING_SCALE, _CYAN, _GREETING_THICKNESS)
    return image


def _show_state_changes() -> None:
    """Cetak setiap perpindahan state (dan peringatan mesin) ke konsol."""
    sm_logger = logging.getLogger("src.application.state_machine")
    sm_logger.setLevel(logging.INFO)
    sm_logger.propagate = False  # cegah cetak dobel bila logging pusat sudah aktif
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

    try:
        detector.load_model()
        detector.warmup()
        with Camera(settings.vision.camera) as cam:
            print("Kamera siap. Berdiri di depan kamera. Tekan 'q' di jendela video untuk keluar.")
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

                # --- Application: timeout, sinkron event, warna ---
                _execute(machine.tick(now), greeter, stabilizer, lock_state, now, ui)
                _execute(_sync_machine(machine, lock_state, now), greeter, stabilizer, lock_state, now, ui)
                _color_step(machine, lock_state, frame, now, color_cfg, stabilizer, greeter, ui)

                # --- Tampilan ---
                highlight_id = lock_state.track_id if lock_state.track_id is not None else candidate_id
                display = draw_tracked_people(frame.image, people, candidates, highlight_id)
                display = draw_lock_status(display, lock_state)
                cv2.putText(
                    display, f"FPS:{cam.current_fps():.1f}  App:{machine.current_state.value}",
                    (_MARGIN, 25), _FONT, 0.7, _WHITE, 2,
                )
                cv2.putText(display, ui.color_line, (_MARGIN, 90), _FONT, 0.55, _WHITE, 2)
                if ui.greeting_text and machine.current_state in _GREETING_VISIBLE_STATES:
                    display = _draw_greeting(display, ui.greeting_text)

                cv2.imshow(_WINDOW, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (CameraError, DetectorError) as exc:
        print(f"Gagal: {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
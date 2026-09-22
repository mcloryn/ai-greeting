"""
scripts/check_greeting.py

Simulasi CP08 TANPA kamera: event buatan -> StateMachine -> aksi -> GreetingManager.
Meniru peran Orchestrator (CP16) dalam bentuk paling sederhana.

Jalankan:  python scripts/check_greeting.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import load_settings
from src.application.greeting_manager import GreetingManager
from src.application.state_machine import ActionType, StateMachine
from src.core.enums import AppState
from src.core.events import EventType, make_event

T0 = datetime(2026, 9, 19, 9, 0).timestamp()  # pagi hari


def serve_person(machine, greeter, track_id, color, t):
    """Satu siklus penuh satu orang. Mengembalikan teks sapaan, atau None bila
    orang itu tidak disapa (masih cooldown)."""
    now = T0 + t
    if not greeter.should_greet(track_id, now):
        print(f"[t={t:>5.1f}s] track {track_id}: TIDAK disapa (masih cooldown)")
        return None

    def send(etype, dt, **payload):
        return machine.handle(make_event(etype, now + dt, payload))

    send(EventType.PERSON_DETECTED, 0.0)
    send(EventType.CANDIDATE_STABLE, 0.5, track_id=track_id)
    send(EventType.LOCK_CONFIRMED, 1.0, track_id=track_id)
    if color is not None:
        actions = send(EventType.COLOR_READY, 2.0, color=color)
    else:
        actions = send(EventType.COLOR_TIMEOUT, 3.0)  # warna gagal stabil

    text = None
    for action in actions:  # <- inilah yang nanti dikerjakan Orchestrator
        if action.type is ActionType.BUILD_GREETING:
            text = greeter.build(action.data["color"], now + 3.0)
        elif action.type is ActionType.MARK_GREETED:
            greeter.mark_greeted(track_id, now + 3.0)
    print(f"[t={t:>5.1f}s] track {track_id}: \"{text}\"")

    # Selesaikan sesi supaya mesin kembali ke IDLE
    send(EventType.SPEECH_DONE, 4.0)
    send(EventType.FAREWELL, 5.0)
    send(EventType.CLEANUP_DONE, 6.0)
    send(EventType.COOLDOWN_EXPIRED, 7.0)
    return text


def main() -> int:
    settings = load_settings()
    greeter = GreetingManager(settings.application.greeting, settings.vision.cooldown_sec)
    # Angka 3 = batas gagal pulih; tidak berpengaruh pada skrip ini.
    machine = StateMachine(settings.application.state_timeouts_sec, 3, now=T0)

    results = []

    def check(label, condition):
        results.append(condition)
        print(f"   -> {'OK  ' if condition else 'GAGAL'} {label}")

    print("--- Skenario 1: orang baru, warna merah ---")
    text = serve_person(machine, greeter, 7, "merah", t=0)
    check("disapa, menyebut 'merah'", text is not None and "merah" in text)
    check("mesin kembali ke IDLE", machine.current_state is AppState.IDLE)

    print("--- Skenario 2: orang yang sama kembali 10 detik kemudian ---")
    text = serve_person(machine, greeter, 7, "merah", t=10)
    check("TIDAK disapa ulang", text is None)

    print("--- Skenario 3: orang lain, warna gagal terdeteksi ---")
    text = serve_person(machine, greeter, 8, None, t=15)
    check("tetap disapa (sapaan cadangan tanpa warna)", text is not None and "baju" not in text)

    print("--- Skenario 4: orang pertama kembali setelah cooldown habis ---")
    text = serve_person(machine, greeter, 7, "biru", t=45)
    check("disapa lagi, menyebut 'biru'", text is not None and "biru" in text)

    print()
    print("SEMUA SKENARIO SESUAI" if all(results) else f"ADA {results.count(False)} SKENARIO GAGAL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
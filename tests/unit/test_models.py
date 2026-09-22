"""
tests/unit/test_models.py

Unit test untuk src/core/models.py -- memastikan setiap dataclass bisa
dibuat dan properti turunan (mis. BoundingBox.area) menghitung dengan benar.
"""

from __future__ import annotations

import numpy as np

from src.core.enums import ColorName, LockPhase, ReleaseReason
from src.core.models import (
    BoundingBox,
    ColorResult,
    ConversationSession,
    Detection,
    Event,
    Frame,
    LockState,
    StableColor,
    TargetCandidate,
    TrackedPerson,
)


def test_frame_bisa_dibuat():
    frame = Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=123.0,
        frame_id=1,
        width=640,
        height=480,
    )
    assert frame.width == 640
    assert frame.height == 480


def test_bounding_box_properti_turunan():
    box = BoundingBox(x1=100, y1=50, x2=200, y2=250)

    assert box.width == 100
    assert box.height == 200
    assert box.area == 100 * 200
    assert box.center == (150.0, 150.0)
    assert box.aspect_ratio == 100 / 200


def test_bounding_box_height_nol_tidak_crash():
    box = BoundingBox(x1=0, y1=0, x2=10, y2=0)
    assert box.aspect_ratio == 0.0


def test_detection_dan_tracked_person():
    box = BoundingBox(0, 0, 10, 10)
    detection = Detection(bbox=box, confidence=0.9, class_id=0, frame_id=1)
    assert detection.confidence == 0.9

    person = TrackedPerson(
        track_id=1, bbox=box, confidence=0.9,
        first_seen=0.0, last_seen=1.0, age_frames=30,
    )
    assert person.track_id == 1


def test_lock_state_default_values():
    lock = LockState(state=LockPhase.IDLE)
    assert lock.track_id is None
    assert lock.release_reason is None


def test_lock_state_dengan_release_reason():
    lock = LockState(state=LockPhase.LOST_GRACE, release_reason=ReleaseReason.TARGET_LOST)
    assert lock.release_reason == ReleaseReason.TARGET_LOST


def test_color_result_dan_stable_color():
    result = ColorResult(
        color_name=ColorName.BIRU, confidence=0.8,
        hsv_median=(110, 200, 180), pixel_count=5000, roi_valid=True,
    )
    assert result.color_name == ColorName.BIRU

    stable = StableColor(color_name=ColorName.BIRU, ratio=0.7, sample_count=15, is_stable=True)
    assert stable.is_stable is True


def test_target_candidate():
    candidate = TargetCandidate(
        track_id=1, score=0.75, area_score=0.8, center_score=0.6, is_eligible=True,
    )
    assert candidate.is_eligible is True


def test_conversation_session_default_history_kosong():
    session = ConversationSession(
        session_id="abc123", track_id=1, clothing_color=ColorName.HIJAU,
        started_at=0.0, last_activity_at=0.0,
    )
    assert session.history == []
    assert session.turn_count == 0


def test_event_bisa_dibuat():
    event = Event(type="PERSON_DETECTED", payload={"track_id": 1}, timestamp=0.0)
    assert event.type == "PERSON_DETECTED"

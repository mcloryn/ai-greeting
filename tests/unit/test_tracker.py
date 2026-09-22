"""
tests/unit/test_tracker.py

Unit test CP04 untuk Tracker. Murni memakai Detection buatan -- tidak
butuh kamera atau model YOLO.
"""

from __future__ import annotations

from src.core.models import BoundingBox, Detection
from src.vision.tracker import Tracker


def _detection(track_id, x1=0, y1=0, x2=100, y2=100, confidence=0.9, frame_id=1):
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=confidence,
        class_id=0,
        frame_id=frame_id,
        track_id=track_id,
    )


def test_detection_tanpa_track_id_dilewati_bukan_crash():
    tracker = Tracker()
    detections = [_detection(track_id=None), _detection(track_id=1)]

    tracked = tracker.update(detections)

    assert len(tracked) == 1
    assert tracked[0].track_id == 1


def test_age_frames_bertambah_selama_track_id_sama():
    tracker = Tracker()

    tracked_1 = tracker.update([_detection(track_id=1)])
    tracked_2 = tracker.update([_detection(track_id=1)])
    tracked_3 = tracker.update([_detection(track_id=1)])

    assert tracked_1[0].age_frames == 1
    assert tracked_2[0].age_frames == 2
    assert tracked_3[0].age_frames == 3


def test_track_id_hilang_lalu_muncul_lagi_dianggap_baru():
    tracker = Tracker()

    tracker.update([_detection(track_id=1)])
    tracker.update([])  # track_id 1 hilang satu frame
    tracked = tracker.update([_detection(track_id=1)])

    # age_frames reset ke 1 karena riwayat lama sudah dihapus saat hilang
    assert tracked[0].age_frames == 1


def test_dua_track_id_berbeda_tidak_saling_pengaruh():
    tracker = Tracker()

    tracked = tracker.update([_detection(track_id=1), _detection(track_id=2)])

    ids = {t.track_id for t in tracked}
    assert ids == {1, 2}
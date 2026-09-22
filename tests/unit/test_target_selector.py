"""
tests/unit/test_target_selector.py

Unit test CP04 untuk logika skor & pemilihan target. Murni memakai data
TrackedPerson buatan -- TIDAK butuh kamera atau model YOLO (syarat
eksplisit Build Plan: "Unit test dijalankan tanpa kamera").
"""

from __future__ import annotations

from config.settings import load_settings
from src.core.models import BoundingBox, TrackedPerson
from src.vision.target_selector import score_candidates, select_target

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _person(track_id, bbox, confidence=0.9, age_frames=20):
    return TrackedPerson(
        track_id=track_id, bbox=bbox, confidence=confidence,
        first_seen=0.0, last_seen=1.0, age_frames=age_frames,
    )


def test_tidak_ada_orang_tidak_ada_kandidat():
    settings = load_settings()
    candidates = score_candidates([], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    assert select_target(candidates) is None


def test_satu_orang_layak_langsung_terpilih():
    settings = load_settings()
    # Kotak besar di tengah frame -> pasti layak
    person = _person(1, BoundingBox(220, 100, 420, 400))
    candidates = score_candidates([person], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    assert select_target(candidates) == 1


def test_orang_dekat_kamera_menang_lawan_orang_jauh():
    settings = load_settings()
    # Orang dekat: kotak besar, dekat tengah
    dekat = _person(1, BoundingBox(200, 50, 440, 450))
    # Orang jauh di background: kotak kecil, di pinggir
    jauh = _person(2, BoundingBox(550, 200, 590, 260))

    candidates = score_candidates([dekat, jauh], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    winner = select_target(candidates)

    assert winner == 1


def test_orang_di_background_tidak_layak():
    settings = load_settings()
    # Kotak sangat kecil -> area_norm pasti di bawah MIN_AREA_RATIO (0.08)
    jauh = _person(1, BoundingBox(300, 200, 320, 230))

    candidates = score_candidates([jauh], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    assert candidates[0].is_eligible is False
    assert select_target(candidates) is None


def test_track_umur_kurang_dari_minimum_tidak_layak():
    settings = load_settings()
    # Kotak besar & di tengah (pasti lolos area & posisi), tapi track
    # baru muncul 2 frame -> di bawah MIN_AGE_FRAMES (15)
    person = _person(1, BoundingBox(220, 100, 420, 400), age_frames=2)

    candidates = score_candidates([person], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    assert candidates[0].is_eligible is False


def test_pemecah_seri_pilih_track_id_terkecil():
    settings = load_settings()
    # Dua kotak identik persis -> skor identik -> track_id terkecil menang
    box = BoundingBox(220, 100, 420, 400)
    a = _person(5, box)
    b = _person(2, box)

    candidates = score_candidates([a, b], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    assert select_target(candidates) == 2


def test_tiga_orang_hanya_satu_terpilih():
    settings = load_settings()
    orang1 = _person(1, BoundingBox(220, 100, 420, 400))   # besar, tengah
    orang2 = _person(2, BoundingBox(0, 0, 60, 80))          # kecil, pojok -- tidak layak
    orang3 = _person(3, BoundingBox(450, 150, 550, 350))    # sedang, agak pinggir

    candidates = score_candidates([orang1, orang2, orang3], FRAME_WIDTH, FRAME_HEIGHT, settings.vision)
    winner = select_target(candidates)

    eligible_ids = [c.track_id for c in candidates if c.is_eligible]
    assert winner in eligible_ids
    assert len(eligible_ids) >= 1
"""
tests/unit/test_target_lock.py

Unit test CP05: lima skenario eksplisit dari Build Plan Bagian
CHECKPOINT 05, memakai waktu yang disuntikkan (bukan time.time()
sungguhan / sleep), sehingga skenario 30 detik diuji dalam milidetik.
"""

from __future__ import annotations

from config.settings import load_settings
from src.core.enums import LockPhase, ReleaseReason
from src.core.models import BoundingBox, TrackedPerson

from src.vision.target_lock import TargetLock

BOX = BoundingBox(100, 100, 300, 400)


def _person(track_id: int) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id, bbox=BOX, confidence=0.9,
        first_seen=0.0, last_seen=0.0, age_frames=20,
    )


def _settings():
    return load_settings().vision


def test_a_tiga_orang_hanya_satu_terkunci():
    """(a) Tiga orang masuk, hanya satu terkunci."""
    settings = _settings()
    lock = TargetLock(settings)
    t = 1000.0

    # TargetSelector sudah memilih track_id=1 sebagai kandidat terbaik
    # (logika pemilihannya sendiri sudah diuji di test_target_selector.py)
    people = [_person(1), _person(2), _person(3)]

    # Kandidat harus stabil selama confirm_timer_sec (0.5s) sebelum terkunci
    state = lock.update(candidate_track_id=1, visible_people=people, now=t)
    assert state.state == LockPhase.CANDIDATE

    t += settings.confirm_timer_sec + 0.01
    state = lock.update(candidate_track_id=1, visible_people=people, now=t)

    assert state.state == LockPhase.LOCKED
    assert state.track_id == 1


def test_b_target_bergerak_kiri_kanan_lock_bertahan():
    """(b) Target berjalan ke kiri dan kanan (posisi bbox berubah-ubah),
    lock tetap bertahan -- TargetLock tidak peduli posisi, hanya peduli
    apakah track_id masih ada di daftar visible.
    """
    settings = _settings()
    lock = TargetLock(settings)
    t = 2000.0

    people = [_person(1)]
    lock.update(candidate_track_id=1, visible_people=people, now=t)
    t += settings.confirm_timer_sec + 0.01
    state = lock.update(candidate_track_id=1, visible_people=people, now=t)
    assert state.state == LockPhase.LOCKED

    # Simulasikan bbox berubah posisi beberapa kali (bergerak kiri-kanan)
    for i in range(5):
        t += 0.1
        moved_person = TrackedPerson(
            track_id=1, bbox=BoundingBox(100 + i * 20, 100, 300 + i * 20, 400),
            confidence=0.9, first_seen=0.0, last_seen=t, age_frames=20 + i,
        )
        state = lock.update(candidate_track_id=1, visible_people=[moved_person], now=t)
        assert state.state == LockPhase.LOCKED
        assert state.track_id == 1


def test_c_target_tertutup_satu_detik_lock_bertahan():
    """(c) Target tertutup selama satu detik (grace_period default 2.5s),
    lock harus bertahan -- BELUM melewati toleransi.
    """
    settings = _settings()
    lock = TargetLock(settings)
    t = 3000.0

    people = [_person(1)]
    lock.update(candidate_track_id=1, visible_people=people, now=t)
    t += settings.confirm_timer_sec + 0.01
    lock.update(candidate_track_id=1, visible_people=people, now=t)  # LOCKED

    # Target hilang dari daftar visible selama 1 detik
    t += 1.0
    state = lock.update(candidate_track_id=None, visible_people=[], now=t)
    assert state.state == LockPhase.LOST_GRACE
    assert state.track_id == 1

    # Target muncul lagi -- lock harus balik LOCKED, bukan hilang
    t += 0.1
    state = lock.update(candidate_track_id=1, visible_people=[_person(1)], now=t)
    assert state.state == LockPhase.LOCKED
    assert state.track_id == 1


def test_d_target_keluar_empat_detik_lock_lepas():
    """(d) Target keluar 4 detik (melewati grace_period 2.5s default),
    lock harus dilepas, dan orang lain boleh dikunci.
    """
    settings = _settings()
    lock = TargetLock(settings)
    t = 4000.0

    people = [_person(1)]
    lock.update(candidate_track_id=1, visible_people=people, now=t)
    t += settings.confirm_timer_sec + 0.01
    lock.update(candidate_track_id=1, visible_people=people, now=t)  # LOCKED

    t += 4.0  # lebih lama dari grace_period_sec (2.5s)
    state = lock.update(candidate_track_id=None, visible_people=[], now=t)

    assert state.state == LockPhase.IDLE
    assert state.release_reason == ReleaseReason.TARGET_LOST

    # Orang lain (track_id=2) sekarang boleh dikunci
    t += 0.1
    state = lock.update(candidate_track_id=2, visible_people=[_person(2)], now=t)
    assert state.state == LockPhase.CANDIDATE
    assert state.track_id == 2


def test_e_target_sama_kembali_dalam_10_detik_tidak_langsung_dikunci():
    """(e) Target yang sama kembali dalam 10 detik (di bawah cooldown
    30s default) TIDAK langsung dikunci ulang.
    """
    settings = _settings()
    lock = TargetLock(settings)
    t = 5000.0

    people = [_person(1)]
    lock.update(candidate_track_id=1, visible_people=people, now=t)
    t += settings.confirm_timer_sec + 0.01
    lock.update(candidate_track_id=1, visible_people=people, now=t)  # LOCKED

    t += 4.0  # lepas karena melewati grace period
    lock.update(candidate_track_id=None, visible_people=[], now=t)  # IDLE, cooldown mulai

    # Track_id=1 kembali 10 detik kemudian -- masih dalam cooldown 30s
    t += 10.0
    state = lock.update(candidate_track_id=1, visible_people=[_person(1)], now=t)

    assert state.state == LockPhase.IDLE  # tetap IDLE, TIDAK jadi CANDIDATE

    # Tapi setelah cooldown benar-benar habis (30s sejak dilepas), boleh lagi
    t += 21.0  # total ~31s sejak dilepas
    state = lock.update(candidate_track_id=1, visible_people=[_person(1)], now=t)
    assert state.state == LockPhase.CANDIDATE


def test_confirm_timer_belum_terlewati_tetap_candidate():
    """Kandidat yang baru muncul (belum 0.5 detik) tidak boleh langsung
    LOCKED -- mencegah orang yang cuma lewat sekilas langsung dikunci.
    """
    settings = _settings()
    lock = TargetLock(settings)
    t = 6000.0

    people = [_person(1)]
    state = lock.update(candidate_track_id=1, visible_people=people, now=t)
    assert state.state == LockPhase.CANDIDATE

    t += settings.confirm_timer_sec / 2  # baru separuh waktu confirm
    state = lock.update(candidate_track_id=1, visible_people=people, now=t)
    assert state.state == LockPhase.CANDIDATE  # belum LOCKED
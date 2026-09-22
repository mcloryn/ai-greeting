"""
src/vision/target_lock.py

Mesin keadaan kecil (state machine) yang mengubah "kandidat terbaik saat
ini" (dari TargetSelector, dihitung ulang tiap frame) menjadi KOMITMEN:
satu target dikunci dan dipertahankan meski ada kandidat lain yang
skornya lebih tinggi, dan tetap toleran bila target sesaat menghilang.

Tidak ada AI di sini -- murni logika kondisional dan perbandingan
timestamp (Bagian 9.5 Build Plan). Waktu SELALU diterima sebagai
parameter (`now`), tidak pernah dibaca sendiri lewat time.time() di
dalam class ini -- ini yang membuat seluruh logika bisa diuji dengan
waktu yang disuntikkan (injected clock), tanpa sleep sungguhan.

Fase (Bagian 5.6 / enums.LockPhase):
    IDLE       -- tidak ada kandidat atau target
    CANDIDATE  -- kandidat sedang "dipanaskan", menunggu confirm_timer_sec
    LOCKED     -- target resmi terkunci
    LOST_GRACE -- target terkunci tapi sedang tidak terlihat, dalam toleransi

Aturan tegas (poin 6 Implementation Tasks): SELAMA LOCKED atau LOST_GRACE,
kandidat lain (parameter candidate_track_id) diabaikan sepenuhnya --
hanya visible_people yang menentukan nasib lock.

CATATAN PERUBAHAN (Task seg-3): `mask_polygon` sekarang diteruskan dari
TrackedPerson ke LockState di setiap titik LockState dibentuk, PERSIS
mengikuti pola bbox yang sudah ada (mask_by_id dibentuk berdampingan
dengan bbox_by_id, dan dibawa lewat _continue_or_release yang sama
seperti self._state.bbox). Tanpa ini, ClothingColorDetector di titik
pemakaian (mis. check_greeting_live.py / Orchestrator) selalu menerima
mask_polygon=None, dan otomatis fallback diam-diam ke ROI rasio tetap
walau use_mask=true di config -- inilah akar masalah warna salah yang
kita telusuri.

Perbaikan lain (tidak terkait mask): method `_handle_lost_grace`
sebelumnya terdefinisi DUA KALI di file ini (isinya identik) --
definisi kedua menimpa yang pertama di Python, jadi definisi pertama
itu dead code. Sudah digabung jadi satu.
"""

from __future__ import annotations

from config.settings import VisionSettings
from src.core.enums import LockPhase, ReleaseReason
from src.core.logger import get_logger
from src.core.models import LockState, TrackedPerson

logger = get_logger(__name__)


class TargetLock:
    """Mengunci satu track_id sebagai target, dengan timer konfirmasi,
    grace period, dan cooldown per-track_id.

    Dipakai seperti ini (di dalam loop, setiap frame):

        lock = TargetLock(settings.vision)
        ...
        lock_state = lock.update(candidate_track_id, tracked_people, now)
    """

    def __init__(self, config: VisionSettings) -> None:
        self._config = config
        self._state = LockState(state=LockPhase.IDLE)

        # Kapan track_id yang sedang di-"panaskan" (fase CANDIDATE) mulai
        # dianggap kandidat -- dipakai untuk hitung confirm_timer_sec.
        self._candidate_track_id: int | None = None
        self._candidate_since: float | None = None

        # track_id -> timestamp kapan boleh dikunci lagi. Dicek & dibersihkan
        # setiap update() supaya tidak tumbuh tanpa batas (poin 5 Tasks).
        self._cooldown_until: dict[int, float] = {}

    @property
    def state(self) -> LockState:
        """State saat ini tanpa memicu transisi -- untuk dibaca visualizer."""
        return self._state

    def update(
        self,
        candidate_track_id: int | None,
        visible_people: list[TrackedPerson],
        now: float,
    ) -> LockState:
        """Perbarui state lock satu langkah, berdasarkan kandidat terbaik
        saat ini (dari TargetSelector) dan daftar orang yang masih
        terlihat di frame ini. Mengembalikan LockState terbaru.
        """
        self._clean_expired_cooldowns(now)

        visible_ids = {p.track_id for p in visible_people}
        bbox_by_id = {p.track_id: p.bbox for p in visible_people}
        # BARU -- berdampingan dengan bbox_by_id, sumbernya sama persis
        # (visible_people), supaya mask_polygon ikut mengalir ke LockState.
        mask_by_id = {p.track_id: p.mask_polygon for p in visible_people}

        if self._state.state == LockPhase.IDLE:
            self._state = self._handle_idle(candidate_track_id, now)
        elif self._state.state == LockPhase.CANDIDATE:
            self._state = self._handle_candidate(
                candidate_track_id, visible_ids, bbox_by_id, mask_by_id, now)
        elif self._state.state == LockPhase.LOCKED:
            self._state = self._handle_locked(visible_ids, bbox_by_id, mask_by_id, now)
        elif self._state.state == LockPhase.LOST_GRACE:
            self._state = self._handle_lost_grace(visible_ids, bbox_by_id, mask_by_id, now)

        return self._state

    # --- Transisi per fase ---------------------------------------------

    def _handle_idle(self, candidate_track_id: int | None, now: float) -> LockState:
        if candidate_track_id is None or candidate_track_id in self._cooldown_until:
            return LockState(state=LockPhase.IDLE)

        self._candidate_track_id = candidate_track_id
        self._candidate_since = now
        logger.debug("Kandidat baru mulai dipanaskan: track_id=%d", candidate_track_id)
        return LockState(state=LockPhase.CANDIDATE, track_id=candidate_track_id, last_seen=now)

    def _handle_candidate(
        self,
        candidate_track_id: int | None,
        visible_ids: set[int],
        bbox_by_id: dict,
        mask_by_id: dict,
        now: float,
    ) -> LockState:
        same_candidate_still_visible = (
            candidate_track_id == self._candidate_track_id
            and self._candidate_track_id in visible_ids
        )

        if not same_candidate_still_visible:
            # Kandidat berganti atau hilang -- mulai ulang dari IDLE
            # (dengan kandidat baru langsung, kalau ada dan tidak cooldown).
            self._candidate_track_id = None
            self._candidate_since = None
            return self._handle_idle(candidate_track_id, now)

        elapsed = now - self._candidate_since
        if elapsed >= self._config.confirm_timer_sec:
            logger.info(
                "Target terkunci: track_id=%d (stabil %.2fs)",
                self._candidate_track_id, elapsed,
            )
            return LockState(
                state=LockPhase.LOCKED,
                track_id=self._candidate_track_id,
                bbox=bbox_by_id.get(self._candidate_track_id),
                mask_polygon=mask_by_id.get(self._candidate_track_id),  # BARU
                locked_since=now,
                last_seen=now,
            )

        return LockState(
            state=LockPhase.CANDIDATE,
            track_id=self._candidate_track_id,
            last_seen=now,
        )

    def _handle_locked(
        self, visible_ids: set[int], bbox_by_id: dict, mask_by_id: dict, now: float,
    ) -> LockState:
        target_id = self._state.track_id

        if target_id in visible_ids:
            return LockState(
                state=LockPhase.LOCKED,
                track_id=target_id,
                bbox=bbox_by_id.get(target_id),
                mask_polygon=mask_by_id.get(target_id),  # BARU
                locked_since=self._state.locked_since,
                last_seen=now,
            )

        # Target tidak terlihat frame ini -- hitung LANGSUNG berapa lama
        # sebenarnya sudah hilang sejak last_seen terakhir (bukan asumsi
        # "baru saja hilang"), supaya lompatan waktu besar dalam satu
        # panggilan update() tetap terdeteksi dengan benar.
        last_seen = self._state.last_seen
        lost_duration = now - last_seen
        return self._continue_or_release(target_id, last_seen, lost_duration, now)

    def _handle_lost_grace(
        self, visible_ids: set[int], bbox_by_id: dict, mask_by_id: dict, now: float,
    ) -> LockState:
        target_id = self._state.track_id

        if target_id in visible_ids:
            logger.info("Target track_id=%d muncul lagi -- lock dipertahankan.", target_id)
            return LockState(
                state=LockPhase.LOCKED,
                track_id=target_id,
                bbox=bbox_by_id.get(target_id),
                mask_polygon=mask_by_id.get(target_id),  # BARU
                locked_since=self._state.locked_since,
                last_seen=now,
            )

        lost_duration = now - self._state.last_seen
        return self._continue_or_release(target_id, self._state.last_seen, lost_duration, now)

    def _continue_or_release(
        self, target_id: int, last_seen: float, lost_duration: float, now: float,
    ) -> LockState:
        """Logika bersama: lepas kalau sudah melewati grace_period_sec,
        atau tetap LOST_GRACE sambil mencatat durasi hilang saat ini.
        Dipakai baik saat baru pertama kali hilang (dari LOCKED) maupun
        saat sudah beberapa saat di LOST_GRACE -- supaya lompatan waktu
        besar dalam satu panggilan tetap terdeteksi benar (lihat CP05
        test skenario d).

        bbox dan mask_polygon di jalur LOST_GRACE diambil dari
        self._state (state SEBELUM panggilan ini) -- keduanya sama-sama
        "posisi/mask terakhir kali target terlihat", karena target
        sedang tidak terlihat sama sekali di frame ini.
        """
        if lost_duration >= self._config.grace_period_sec:
            logger.info(
                "Target track_id=%d dilepas (hilang %.2fs, melewati grace period).",
                target_id, lost_duration,
            )
            self._cooldown_until[target_id] = now + self._config.cooldown_sec
            self._candidate_track_id = None
            self._candidate_since = None
            return LockState(state=LockPhase.IDLE, release_reason=ReleaseReason.TARGET_LOST)

        return LockState(
            state=LockPhase.LOST_GRACE,
            track_id=target_id,
            bbox=self._state.bbox,
            mask_polygon=self._state.mask_polygon,  # BARU
            locked_since=self._state.locked_since,
            last_seen=last_seen,
            lost_duration=lost_duration,
        )

    def _clean_expired_cooldowns(self, now: float) -> None:
        expired = [tid for tid, until in self._cooldown_until.items() if until <= now]
        for tid in expired:
            del self._cooldown_until[tid]
            logger.debug("Cooldown track_id=%d berakhir.", tid)
"""
src/vision/tracker.py

Mengubah Detection (yang sudah membawa track_id dari tracker bawaan
Ultralytics, lihat PersonDetector.detect_and_track) menjadi TrackedPerson
lengkap dengan first_seen/last_seen/age_frames.

Sesuai Bagian 9.3 Build Plan: tracker sungguhan (algoritma pencocokan
kotak antar-frame) sudah dikerjakan oleh pustaka Ultralytics -- module
ini HANYA menjaga riwayat (kapan track_id pertama kali & terakhir
terlihat, sudah berapa lama) di atas hasil itu. Tidak menulis ulang
algoritma tracking dari nol.
"""

from __future__ import annotations

import time

from src.core.exceptions import DetectorError
from src.core.logger import get_logger
from src.core.models import Detection, TrackedPerson

logger = get_logger(__name__)


class Tracker:
    """Menjaga riwayat umur setiap track_id antar-frame.

    Dipakai seperti ini:

        tracker = Tracker()
        tracked_people = tracker.update(detections)
    """

    def __init__(self) -> None:
        # Menyimpan first_seen per track_id yang masih pernah terlihat.
        # Tidak pernah dihapus manual -- kalau track_id lama tidak
        # muncul lagi di update(), dia otomatis tidak masuk hasil, dan
        # kalau Ultralytics memakai ulang angka track_id (jarang, tapi
        # mungkin), first_seen akan diperbarui secara wajar sebagai
        # "kemunculan baru".
        self._first_seen: dict[int, float] = {}
        self._age_frames: dict[int, int] = {}

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        """Ubah detections (yang punya track_id) jadi list[TrackedPerson].

        Detection dengan track_id=None dilewati (bukan bikin crash) --
        ini kondisi normal saat tracker bawaan Ultralytics belum sempat
        mengonfirmasi identitas untuk objek yang baru saja muncul di
        frame. Objek itu akan mendapat track_id di frame-frame
        berikutnya begitu tracker mengonfirmasinya.
        """
        now = time.time()
        tracked: list[TrackedPerson] = []
        seen_this_frame: set[int] = set()

        for det in detections:
            if det.track_id is None:
                logger.debug(
                    "Deteksi tanpa track_id dilewati (belum dikonfirmasi tracker)."
                )
                continue

            track_id = det.track_id
            seen_this_frame.add(track_id)

            if track_id not in self._first_seen:
                self._first_seen[track_id] = now
                self._age_frames[track_id] = 0
                logger.debug("Track baru: track_id=%d", track_id)

            self._age_frames[track_id] += 1

            tracked.append(TrackedPerson(
                track_id=track_id,
                bbox=det.bbox,
                confidence=det.confidence,
                first_seen=self._first_seen[track_id],
                last_seen=now,
                age_frames=self._age_frames[track_id],
                mask_polygon=det.mask_polygon,
            ))

        stale_ids = set(self._first_seen) - seen_this_frame
        for stale_id in stale_ids:
            del self._first_seen[stale_id]
            del self._age_frames[stale_id]

        return tracked
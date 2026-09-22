"""
src/vision/target_selector.py

Dari banyak TrackedPerson, pilih SATU sebagai kandidat target utama.
Fungsi murni, tanpa state -- semua yang dibutuhkan diterima sebagai
argumen (Bagian 5 & 10 Build Plan).

Formula (Bagian 10.3):
    area_norm    = (bbox.width * bbox.height) / (frame_width * frame_height)
    center_dist  = |bbox.center_x - frame_center_x| / (frame_width / 2)
    center_score = 1 - min(center_dist, 1.0)
    score        = W_AREA * area_norm + W_CENTER * center_score

Syarat kelayakan (semua wajib):
    area_norm  >= MIN_AREA_RATIO
    age_frames >= MIN_AGE_FRAMES
    confidence >= CONF_THRESHOLD

Pemecah seri: selisih skor < 0.01 -> pilih track_id terkecil (deterministik).
"""

from __future__ import annotations

from config.settings import VisionSettings
from src.core.models import TargetCandidate, TrackedPerson

_TIE_BREAK_MARGIN = 0.01


def score_candidates(
    people: list[TrackedPerson],
    frame_width: int,
    frame_height: int,
    config: VisionSettings,
) -> list[TargetCandidate]:
    """Hitung skor & kelayakan setiap orang. Tidak mengurutkan atau memilih
    -- itu tugas select_target(). Dipisah supaya skor semua kandidat bisa
    ditampilkan di layar untuk kalibrasi (poin 8, Implementation Tasks CP04).
    """
    candidates: list[TargetCandidate] = []
    frame_area = frame_width * frame_height
    frame_center_x = frame_width / 2

    for person in people:
        area_norm = person.bbox.area / frame_area if frame_area > 0 else 0.0

        center_dist = abs(person.bbox.center[0] - frame_center_x) / (frame_width / 2) \
            if frame_width > 0 else 1.0
        center_score = 1 - min(center_dist, 1.0)

        score = (
            config.weight_area * area_norm
            + config.weight_center * center_score
        )

        is_eligible = (
            area_norm >= config.min_area_ratio
            and person.age_frames >= config.min_age_frames
            and person.confidence >= config.confidence_threshold
        )

        candidates.append(TargetCandidate(
            track_id=person.track_id,
            score=score,
            area_score=area_norm,
            center_score=center_score,
            is_eligible=is_eligible,
        ))

    return candidates


def select_target(candidates: list[TargetCandidate]) -> int | None:
    """Pilih satu track_id terbaik dari kandidat yang layak.

    Mengembalikan None kalau tidak ada kandidat layak sama sekali
    (Bagian 10.4: "Tidak ada orang -> Tidak ada kandidat").
    """
    eligible = [c for c in candidates if c.is_eligible]
    if not eligible:
        return None

    best = max(eligible, key=lambda c: c.score)

    # Pemecah seri deterministik: kalau ada kandidat lain yang skornya
    # nyaris sama (selisih < 0.01), track_id terkecil yang menang --
    # supaya hasil bisa direproduksi di unit test dan tidak "berkedip"
    # antar dua orang dengan skor mirip.
    near_ties = [c for c in eligible if abs(c.score - best.score) < _TIE_BREAK_MARGIN]
    if len(near_ties) > 1:
        best = min(near_ties, key=lambda c: c.track_id)

    return best.track_id
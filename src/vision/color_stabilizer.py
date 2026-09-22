"""
src/vision/color_stabilizer.py

Menyerap fluktuasi klasifikasi warna per-frame (yang bisa berubah-ubah
karena noise, pencahayaan sesaat, pergerakan) menjadi satu warna yang
stabil, lewat buffer waktu berjalan + voting mayoritas (Task 9).

Pola bufer waktu-berjalan ini SAMA dengan Camera._compute_fps() di CP02
-- deque yang menyimpan (timestamp, nilai), dibersihkan dari entri yang
lebih tua dari jendela waktu yang ditentukan.
"""

from __future__ import annotations

from collections import Counter, deque

from config.settings import ClothingColorSettings
from src.core.enums import ColorName
from src.core.models import StableColor


class ColorStabilizer:
    """Buffer warna berbasis waktu + voting mayoritas.

    Dipakai seperti ini:

        stabilizer = ColorStabilizer(settings.vision.clothing_color)
        stabilizer.add(color_result.color_name, now)
        stable = stabilizer.get_stable_color()
    """

    def __init__(self, config: ClothingColorSettings) -> None:
        self._config = config
        # Menyimpan (timestamp, ColorName | None) -- None disimpan juga
        # supaya ikut dihitung sebagai "sampel tidak yakin" saat voting,
        # bukan diam-diam dibuang (representasi jujur dari histori).
        self._buffer: deque[tuple[float, ColorName | None]] = deque()

    def add(self, color_name: ColorName | None, now: float) -> None:
        """Tambah satu sampel warna, lalu buang sampel yang sudah lebih
        tua dari stabilizer_buffer_sec dari sekarang.
        """
        self._buffer.append((now, color_name))
        cutoff = now - self._config.stabilizer_buffer_sec
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_stable_color(self) -> StableColor:
        """Hitung warna mayoritas di buffer saat ini. is_stable=True hanya
        kalau warna terbanyak mencapai ambang stabilizer_majority_ratio.
        """
        if not self._buffer:
            return StableColor(color_name=None, ratio=0.0, sample_count=0, is_stable=False)

        sample_count = len(self._buffer)
        counts = Counter(color_name for _, color_name in self._buffer)
        winner_color, winner_count = counts.most_common(1)[0]

        ratio = winner_count / sample_count
        is_stable = winner_color is not None and ratio >= self._config.stabilizer_majority_ratio

        return StableColor(
            color_name=winner_color if is_stable else None,
            ratio=ratio,
            sample_count=sample_count,
            is_stable=is_stable,
        )

    def reset(self) -> None:
        """Kosongkan buffer -- WAJIB dipanggil setiap kali target berganti
        (Task 10), supaya warna orang sebelumnya tidak terbawa ke orang
        baru (Failure Case eksplisit di Build Plan).
        """
        self._buffer.clear()
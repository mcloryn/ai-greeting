"""
src/application/greeting_manager.py

Merakit kalimat sapaan dan menjamin satu orang hanya disapa satu kali per
masa cooldown (Bagian 4.7 Build Plan).

Yang dilakukan module ini:
  - Memilih template (bergantian, supaya tidak monoton) lalu mengisi
    {waktu} dan {warna}.
  - Bila warna None / tidak dikenal -> memakai template cadangan tanpa warna.
  - Mencatat siapa yang sudah disapa dan kapan (daftar cooldown).

Yang TIDAK dilakukan module ini:
  - Tidak memutuskan KAPAN menyapa (itu tugas StateMachine).
  - Tidak bicara / memutar suara (itu CP15).
  - Tidak membaca jam sistem: waktu selalu diberikan pemanggil (`now`),
    supaya unit test bisa mengatur waktu sesukanya.
"""

from __future__ import annotations

import logging
from datetime import datetime
from string import Formatter

from config.settings import GreetingSettings
from src.core.enums import ColorName

logger = logging.getLogger(__name__)

_COLORED_FIELDS = {"warna", "waktu"}
_FALLBACK_FIELDS = {"waktu"}


class GreetingManager:
    """Perakit sapaan + pencatat cooldown.

    Contoh pemakaian (nanti dilakukan Orchestrator di CP16):

        greeter = GreetingManager(settings.application.greeting,
                                  settings.vision.cooldown_sec)
        if greeter.should_greet(track_id, now):
            ...                                  # StateMachine berjalan
            text = greeter.build(color, now)     # aksi BUILD_GREETING
            greeter.mark_greeted(track_id, now)  # aksi MARK_GREETED
    """

    def __init__(self, config: GreetingSettings, cooldown_sec: float) -> None:
        if cooldown_sec < 0:
            raise ValueError("cooldown_sec tidak boleh negatif")

        self._colored = list(config.templates)
        self._fallback = list(config.fallback_templates)
        self._validate_templates(self._colored, _COLORED_FIELDS, "templates", must_have="warna")
        self._validate_templates(self._fallback, _FALLBACK_FIELDS, "fallback_templates")
        self._display_names = self._validate_display_names(config.color_display_names)

        self._hours = config
        self._cooldown_sec = float(cooldown_sec)

        # track_id -> waktu terakhir disapa
        self._greeted_at: dict[int, float] = {}
        # Penunjuk giliran template (dipisah untuk berwarna dan cadangan)
        self._colored_index = 0
        self._fallback_index = 0

    # ------------------------------------------------------------ validasi

    @staticmethod
    def _validate_templates(
        templates: list[str],
        allowed: set[str],
        label: str,
        must_have: str | None = None,
    ) -> None:
        """Gagal cepat saat startup kalau ada placeholder yang tidak tersedia,
        bukan saat seseorang sedang berdiri menunggu disapa."""
        if not templates:
            raise ValueError(f"greeting.{label} tidak boleh kosong")
        for template in templates:
            try:
                fields = {f for _, f, _, _ in Formatter().parse(template) if f is not None}
            except ValueError as exc:
                raise ValueError(f"greeting.{label}: format tidak valid di {template!r} ({exc})") from exc
            unknown = fields - allowed
            if unknown:
                raise ValueError(
                    f"greeting.{label}: placeholder {sorted(unknown)} tidak tersedia "
                    f"di {template!r}. Yang boleh: {sorted(allowed)}"
                )
            if must_have and must_have not in fields:
                raise ValueError(f"greeting.{label}: template {template!r} wajib memuat {{{must_have}}}")

    @staticmethod
    def _validate_display_names(names: dict[str, str]) -> dict[str, str]:
        missing = [c.value for c in ColorName if c.value not in names]
        if missing:
            raise ValueError(f"greeting.color_display_names kekurangan warna: {missing}")
        return dict(names)

    # ------------------------------------------------------ waktu dan warna

    def time_of_day(self, now: float) -> str:
        """Kembalikan 'pagi' / 'siang' / 'sore' / 'malam' dari timestamp."""
        hour = datetime.fromtimestamp(now).hour
        h = self._hours
        if h.pagi_start_hour <= hour < h.siang_start_hour:
            return "pagi"
        if h.siang_start_hour <= hour < h.sore_start_hour:
            return "siang"
        if h.sore_start_hour <= hour < h.malam_start_hour:
            return "sore"
        return "malam"  # dari malam_start sampai lewat tengah malam, hingga pagi_start

    def _spoken_color(self, color: ColorName | str | None) -> str | None:
        """Nama warna untuk diucapkan, atau None bila tidak ada / tidak dikenal."""
        if color is None:
            return None
        try:
            return self._display_names[ColorName(color).value]
        except ValueError:
            logger.warning("Warna %r tidak dikenal, memakai sapaan cadangan", color)
            return None

    # ------------------------------------------------------ antarmuka publik

    def should_greet(self, track_id: int, now: float) -> bool:
        """True bila orang ini belum pernah disapa, atau cooldown-nya sudah habis."""
        greeted_at = self._greeted_at.get(track_id)
        if greeted_at is None:
            return True
        return (now - greeted_at) >= self._cooldown_sec

    def build(self, color: ColorName | str | None, now: float) -> str:
        """Rakit kalimat sapaan. Tidak menandai siapa pun sebagai 'sudah disapa'."""
        waktu = self.time_of_day(now)
        spoken = self._spoken_color(color)

        if spoken is None:
            template = self._fallback[self._fallback_index % len(self._fallback)]
            self._fallback_index += 1
            return template.format(waktu=waktu)

        template = self._colored[self._colored_index % len(self._colored)]
        self._colored_index += 1
        return template.format(warna=spoken, waktu=waktu)

    def mark_greeted(self, track_id: int, now: float) -> None:
        """Catat bahwa orang ini baru saja disapa (memulai cooldown)."""
        self._greeted_at[track_id] = now
        logger.info("Track %d ditandai sudah disapa", track_id)

    def purge_expired(self, now: float) -> int:
        """Buang entri yang cooldown-nya sudah habis. Mengembalikan jumlah yang dibuang.

        Dipanggil saat aksi PURGE_EXPIRED_COOLDOWNS, supaya daftar tidak
        tumbuh tanpa batas selama sistem menyala berjam-jam.
        """
        expired = [tid for tid, t in self._greeted_at.items() if (now - t) >= self._cooldown_sec]
        for tid in expired:
            del self._greeted_at[tid]
        return len(expired)

    @property
    def tracked_count(self) -> int:
        """Jumlah orang yang saat ini tercatat di daftar cooldown."""
        return len(self._greeted_at)
"""
src/knowledge/parsers/plain_text.py

Parser untuk file .txt dan .md.

Kenapa satu class untuk dua ekstensi? Di level Parser, keduanya sama
persis: baca isi file, jadikan teks mentah. Tanda markdown (#, ##, dst)
TIDAK diinterpretasi di sini -- itu tugas Chunker nanti (Task 8 CP10),
yang membaca tanda # untuk memutuskan di mana memotong. Parser hanya
bertanggung jawab mengekstrak teks, bukan memahami strukturnya
(Aturan #6, modular: setiap modul cuma tahu urusannya sendiri).
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import RawDocument, RawPage
from src.core.logger import get_logger
from src.knowledge.parsers.base import DocumentParser

logger = get_logger(__name__)


class PlainTextParser(DocumentParser):
    """Parser untuk file .txt dan .md.

    Tidak ada konsep halaman di format ini, jadi seluruh isi file
    dianggap satu RawPage dengan page_number=None (sesuai aturan yang
    didefinisikan di docstring RawPage, src/core/models.py).
    """

    def parse(self, file_path: Path) -> RawDocument:
        # encoding="utf-8" ditulis eksplisit, bukan dibiarkan default.
        # Kalau default OS berbeda (mis. Windows kadang pakai cp1252),
        # karakter seperti "é" atau tanda kutip pintar bisa rusak diam-diam.
        text = file_path.read_text(encoding="utf-8")

        # File kosong TETAP menghasilkan RawDocument yang valid (bukan
        # exception). Ini sengaja: Task 13/Testing CP10 mengharuskan
        # kasus "dokumen kosong" diuji, bukan dianggap error. Cleaner
        # atau Chunker nanti yang berhak memutuskan mengabaikan halaman
        # kosong -- Parser tidak boleh diam-diam membuang data.
        page = RawPage(text=text, page_number=None)

        logger.debug(
            f"Parsed {file_path.name}: {len(text)} karakter, "
            f"{'kosong' if not text.strip() else 'berisi teks'}"
        )

        return RawDocument(
            source_file=file_path.name,
            extension=file_path.suffix.lower(),
            pages=[page],
        )

    def supported_extensions(self) -> set[str]:
        return {".txt", ".md"}
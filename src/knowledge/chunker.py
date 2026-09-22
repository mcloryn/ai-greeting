"""
src/knowledge/chunker.py

Chunker: memotong CleanDocument menjadi list[DocumentChunk] siap embed.

Strategi (Task 8-9 CP10, Bagian 4.12 Build Plan):
1. Potong dulu berdasarkan STRUKTUR -- baris yang diawali "#" (markdown
   heading) jadi batas section. Ini kenapa Parser TIDAK boleh membuang
   tanda "#": Chunker inilah yang membutuhkannya.
2. Kalau satu section masih terlalu panjang (> chunk_size_words), potong
   lagi berdasarkan jumlah kata, dengan overlap antar-potongan supaya
   konteks di perbatasan chunk tidak hilang total.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from src.core.models import DocumentChunk
from src.core.logger import get_logger
from src.knowledge.cleaner import CleanDocument

logger = get_logger(__name__)

# Baris markdown heading: 1-6 tanda '#' diikuti spasi lalu teks judul.
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class Chunker:
    def __init__(self, chunk_size_words: int, chunk_overlap_words: int):
        if chunk_overlap_words >= chunk_size_words:
            # Validasi ini SENGAJA diulang di sini, bukan cuma di
            # config/settings.py. Alasan: Chunker adalah class yang
            # bisa dipakai/diuji sendiri (lihat test_chunker.py) tanpa
            # lewat load_settings() sama sekali -- kalau validasinya
            # hanya ada di settings.py, Chunker yang dipanggil langsung
            # dengan angka sembarangan (mis. dari unit test yang lupa)
            # akan diam-diam menghasilkan chunk kosong berulang, bukan
            # error yang jelas.
            raise ValueError(
                f"chunk_overlap_words ({chunk_overlap_words}) harus lebih kecil "
                f"dari chunk_size_words ({chunk_size_words})."
            )
        self._chunk_size_words = chunk_size_words
        self._chunk_overlap_words = chunk_overlap_words

    def chunk(self, document: CleanDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in document.pages:
            sections = self._split_by_heading(page.text)

            for section_title, section_text in sections:
                if not section_text.strip():
                    continue  # section kosong (mis. heading tanpa isi) dilewati

                word_pieces = self._split_by_length(section_text)

                for piece in word_pieces:
                    if not piece.strip():
                        continue

                    chunk_index += 1
                    chunk_id = self._make_chunk_id(
                        document.source_file, page.page_number, chunk_index
                    )
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=piece,
                            source_file=document.source_file,
                            page=page.page_number,
                            section=section_title,
                            char_count=len(piece),
                            ingested_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )

        logger.info(f"{document.source_file}: {chunk_index} chunk dihasilkan")
        return chunks

    def _split_by_heading(self, text: str) -> list[tuple[str | None, str]]:
        """Potong teks di tiap baris heading markdown.

        Mengembalikan list of (judul_section, isi_teks). Teks sebelum
        heading pertama (kalau ada) diberi judul None -- ini menangani
        dokumen .txt yang sama sekali tidak punya heading: seluruh
        isinya jadi satu section dengan section=None.
        """
        matches = list(_HEADING_PATTERN.finditer(text))

        if not matches:
            return [(None, text)]

        sections: list[tuple[str | None, str]] = []

        # Teks sebelum heading pertama (kalau ada isinya)
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((title, body))

        return sections

    def _split_by_length(self, text: str) -> list[str]:
        """Potong satu section jadi beberapa piece kalau lebih panjang dari
        chunk_size_words, dengan overlap antar-piece.

        Dipotong per KATA (bukan per karakter) supaya tidak pernah
        memutus kata atau angka penting di tengah -- salah satu Failure
        Case eksplisit di CP10: "chunk yang memotong angka penting di
        tengah".
        """
        words = text.split()

        if len(words) <= self._chunk_size_words:
            return [text]

        pieces: list[str] = []
        step = self._chunk_size_words - self._chunk_overlap_words
        start = 0

        while start < len(words):
            end = start + self._chunk_size_words
            piece_words = words[start:end]
            pieces.append(" ".join(piece_words))

            if end >= len(words):
                break
            start += step

        return pieces

    @staticmethod
    def _make_chunk_id(source_file: str, page_number: int | None, index: int) -> str:
        # Format: namafile_p<halaman>_c<urutan>. Halaman "x" dipakai untuk
        # dokumen tanpa konsep halaman (.txt/.md), supaya format chunk_id
        # tetap konsisten dan tidak perlu dua format berbeda.
        stem = source_file.rsplit(".", 1)[0]
        page_part = f"p{page_number}" if page_number is not None else "px"
        return f"{stem}_{page_part}_c{index:03d}"
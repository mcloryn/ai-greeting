"""
src/knowledge/cleaner.py

DocumentProcessor (Cleaner): mengubah RawDocument (teks mentah dari
Parser) menjadi CleanDocument (teks bersih siap dipotong Chunker).

Tanggung jawab (Bagian 4.11 Build Plan):
- Buang header/footer berulang, nomor halaman, spasi ganda, karakter sampah
- TIDAK memotong teks (itu tugas Chunker)
- TIDAK menginterpretasi heading (itu tugas Chunker)

Catatan arsitektur: dataclass CleanPage dan CleanDocument didefinisikan
di src/core/models.py (mengikuti konvensi yang sama seperti
ConversationSession/session.py) -- file ini TIDAK mendefinisikan ulang
strukturnya, hanya berisi logika yang mengoperasikannya.
"""

from __future__ import annotations

import re

from src.core.models import RawDocument, CleanDocument, CleanPage
from src.core.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Membersihkan teks mentah: komentar HTML, tabel markdown, spasi ganda, baris kosong berlebih, karakter sampah."""

    def process(self, raw: RawDocument) -> CleanDocument:
        clean_pages = [
            CleanPage(text=self.clean_text(page.text), page_number=page.page_number)
            for page in raw.pages
        ]
        return CleanDocument(
            source_file=raw.source_file,
            extension=raw.extension,
            pages=clean_pages,
        )

    def clean_text(self, text: str) -> str:
        """Bersihkan satu blok teks.

        Urutan operasi sengaja begini:
        0a. Buang komentar HTML (<!-- ... -->) DULUAN -- ini metadata
            dokumentasi (mis. catatan "dokumen dummy"), bukan konten,
            dan kalau dibiarkan bisa jadi chunk tersendiri yang tidak
            berguna buat RAG.
        0b. Konversi tabel markdown jadi kalimat natural, sebelum
            whitespace dirapikan -- supaya kalimat hasil konversi ikut
            kena normalisasi spasi/baris kosong di langkah berikutnya,
            bukan diproses terpisah dengan aturan whitespace sendiri.
        1. Normalisasi akhir baris (Windows \\r\\n -> \\n) dulu, supaya
           regex baris kosong di langkah berikutnya konsisten.
        2. Buang whitespace di akhir tiap baris (trailing spaces).
        3. Rapikan baris kosong berlebih (3+ baris kosong -> maksimal 2,
           supaya paragraf masih terpisah jelas tapi tidak ada jarak
           kosong berlebihan yang memakan "ruang" chunk nanti).
        4. Buang spasi ganda di dalam baris (bukan di awal baris, supaya
           indentasi list markdown seperti "  - item" tidak rusak).
        """
        if not text:
            return text

        text = self._strip_html_comments(text)
        text = self._convert_markdown_tables(text)

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"(?<=\S)  +", " ", text)

        return text.strip()

    def _strip_html_comments(self, text: str) -> str:
        """Buang komentar HTML/markdown '<!-- ... -->', termasuk yang
        multi-baris (re.DOTALL), karena isinya metadata dokumentasi
        (mis. penanda "dokumen dummy"), bukan konten yang perlu di-embed.
        """
        return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    
    def _convert_markdown_tables(self, text: str) -> str:
        """Deteksi blok tabel markdown (header + separator + baris data),
        ubah tiap baris data jadi kalimat natural: '<kolom1> memiliki
        <header2> <nilai2> dan <header3> <nilai3>.' Baris non-tabel dan
        blok yang tidak valid (tanpa separator '|---|---|') dibiarkan
        apa adanya.
        """
        lines = text.split("\n")
        output_lines: list[str] = []
        i = 0

        def is_table_row(line: str) -> bool:
            return line.strip().startswith("|") and line.strip().endswith("|")

        def is_separator_row(line: str) -> bool:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c)

        def parse_row(line: str) -> list[str]:
            return [c.strip() for c in line.strip().strip("|").split("|")]

        while i < len(lines):
            line = lines[i]

            if is_table_row(line):
                table_block = [line]
                j = i + 1
                while j < len(lines) and is_table_row(lines[j]):
                    table_block.append(lines[j])
                    j += 1

                if len(table_block) >= 3 and is_separator_row(table_block[1]):
                    headers = parse_row(table_block[0])
                    data_rows = table_block[2:]

                    for row_line in data_rows:
                        cells = parse_row(row_line)
                        if len(cells) != len(headers):
                            logger.warning(
                                "Baris tabel dilewati (jumlah kolom tidak cocok header): %r",
                                row_line,
                            )
                            continue

                        subject = cells[0]
                        parts = [
                            f"{h} {v}"
                            for h, v in zip(headers[1:], cells[1:])
                            if h and v
                        ]

                        if parts:
                            output_lines.append(f"{subject} memiliki " + " dan ".join(parts) + ".")
                        elif subject:
                            output_lines.append(subject + ".")

                    i = j
                    continue
                else:
                    output_lines.extend(table_block)
                    i = j
                    continue

            output_lines.append(line)
            i += 1

        return "\n".join(output_lines)
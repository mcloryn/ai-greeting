"""
scripts/ingest.py

Ingest TAHAP PERTAMA (Task 12 CP10): baca semua dokumen di
data/documents/sample/, proses lewat Loader -> Cleaner -> Chunker,
lalu cetak statistik dan simpan hasil ekstraksi bersih ke
data/documents/processed/ supaya bisa DIBACA MANUSIA (Task 7).

Embedding + penyimpanan ke vector store BELUM ada di sini -- itu
Task 10 CP11, ingest.py akan dilengkapi lagi di checkpoint itu.

Jalankan:  python scripts/ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ConfigError, load_settings
from src.knowledge.document_loader import DocumentLoader
from src.knowledge.parsers.plain_text import PlainTextParser
from src.knowledge.cleaner import DocumentProcessor
from src.knowledge.chunker import Chunker

SAMPLE_DOCS_DIR = Path("data/documents/sample")
PROCESSED_DIR = Path("data/documents/processed")


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Konfigurasi bermasalah: {exc}")
        return 1

    loader = DocumentLoader(parsers=[PlainTextParser()])
    cleaner = DocumentProcessor()
    chunker = Chunker(
        chunk_size_words=settings.knowledge.chunk_size_words,
        chunk_overlap_words=settings.knowledge.chunk_overlap_words,
    )

    print(f"Memindai folder: {SAMPLE_DOCS_DIR}")
    raw_documents = loader.load_all(SAMPLE_DOCS_DIR)

    if not raw_documents:
        print("Tidak ada dokumen yang berhasil dimuat. Berhenti.")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_lengths = []

    for raw_doc in raw_documents:
        clean_doc = cleaner.process(raw_doc)

        # Simpan hasil ekstraksi bersih ke file teks, supaya manusia bisa
        # membacanya langsung (Task 7 CP10) -- tanpa perlu baca database
        # atau lewat kode apa pun.
        processed_text = "\n\n".join(page.text for page in clean_doc.pages)
        out_path = PROCESSED_DIR / f"{raw_doc.source_file}.clean.txt"
        out_path.write_text(processed_text, encoding="utf-8")

        doc_chunks = chunker.chunk(clean_doc)
        all_chunks.extend(doc_chunks)
        chunk_lengths.extend(chunk.char_count for chunk in doc_chunks)

    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

    print("\n=== Statistik Ingest ===")
    print(f"Jumlah dokumen diproses : {len(raw_documents)}")
    print(f"Jumlah chunk dihasilkan : {len(all_chunks)}")
    print(f"Rata-rata panjang chunk : {avg_length:.0f} karakter")
    print(f"Chunk terpendek         : {min(chunk_lengths) if chunk_lengths else 0} karakter")
    print(f"Chunk terpanjang        : {max(chunk_lengths) if chunk_lengths else 0} karakter")
    print(f"\nHasil ekstraksi bersih disimpan di: {PROCESSED_DIR}/")
    print("Baca beberapa file di folder itu untuk verifikasi manual (Task 13 CP10).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
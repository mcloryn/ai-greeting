"""
tests/unit/test_chunker.py

Unit test untuk src/knowledge/chunker.py. Tidak butuh kamera, internet,
maupun file sungguhan di disk (Aturan #23, sesuai pola test_conversation_manager.py).
"""

import pytest

from src.knowledge.chunker import Chunker
from src.knowledge.cleaner import CleanDocument, CleanPage


def _make_document(text: str, page_number: int | None = None) -> CleanDocument:
    return CleanDocument(
        source_file="contoh.md",
        extension=".md",
        pages=[CleanPage(text=text, page_number=page_number)],
    )


def test_chunker_menolak_overlap_lebih_besar_dari_size():
    with pytest.raises(ValueError, match="overlap"):
        Chunker(chunk_size_words=50, chunk_overlap_words=100)


def test_dokumen_tanpa_heading_jadi_satu_section():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document("Ini teks polos tanpa heading sama sekali.")
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].section is None


def test_dokumen_dengan_heading_dipisah_per_section():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document(
        "# Judul Satu\nIsi bagian satu.\n\n# Judul Dua\nIsi bagian dua."
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].section == "Judul Satu"
    assert chunks[1].section == "Judul Dua"


def test_section_panjang_dipotong_dengan_overlap():
    # 100 kata dummy, chunk_size 20, overlap 5 -> step 15 kata per potongan
    words = " ".join(f"kata{i}" for i in range(100))
    chunker = Chunker(chunk_size_words=20, chunk_overlap_words=5)
    doc = _make_document(words)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    # overlap harus benar-benar terjadi: kata terakhir chunk pertama
    # muncul lagi di awal chunk kedua
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-1] in second_words


def test_chunk_id_formatnya_konsisten():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document("Teks singkat.", page_number=3)
    chunks = chunker.chunk(doc)

    assert chunks[0].chunk_id == "contoh_p3_c001"


def test_chunk_id_untuk_dokumen_tanpa_halaman():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document("Teks singkat.", page_number=None)
    chunks = chunker.chunk(doc)

    assert chunks[0].chunk_id == "contoh_px_c001"


def test_section_kosong_dilewati():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document("# Judul Kosong\n\n# Judul Berisi\nAda teks di sini.")
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].section == "Judul Berisi"


def test_dokumen_kosong_menghasilkan_nol_chunk():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document("")
    chunks = chunker.chunk(doc)

    assert chunks == []


def test_char_count_sesuai_panjang_teks():
    chunker = Chunker(chunk_size_words=500, chunk_overlap_words=50)
    doc = _make_document("Sepuluh karakter.")
    chunks = chunker.chunk(doc)

    assert chunks[0].char_count == len(chunks[0].text)
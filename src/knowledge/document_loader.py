"""
src/knowledge/document_loader.py

DocumentLoader: menemukan file dokumen di sebuah folder, lalu mengarahkan
tiap file ke parser yang tepat berdasarkan ekstensinya.

DocumentLoader TIDAK tahu detail cara kerja parser (Aturan #6). Ia hanya
menyimpan daftar parser yang tersedia dan bertanya ke tiap parser lewat
supported_extensions() -- persis kontrak yang didefinisikan di
DocumentParser (Bagian 4.10 Build Plan).
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import RawDocument
from src.core.logger import get_logger
from src.knowledge.parsers.base import DocumentParser

logger = get_logger(__name__)


class DocumentLoader:
    """Memindai folder dokumen dan memuat tiap file lewat parser yang sesuai."""

    def __init__(self, parsers: list[DocumentParser]):
        # Dibangun jadi dict {ekstensi: parser} sekali di awal, supaya
        # load_one() tidak perlu looping semua parser tiap kali dipanggil.
        self._parser_by_extension: dict[str, DocumentParser] = {}
        for parser in parsers:
            for ext in parser.supported_extensions():
                self._parser_by_extension[ext] = parser

    def supported_extensions(self) -> set[str]:
        """Seluruh ekstensi yang bisa ditangani, gabungan dari semua parser."""
        return set(self._parser_by_extension.keys())

    def load_one(self, path: Path) -> RawDocument:
        """Muat satu file lewat parser yang sesuai ekstensinya.

        Raises:
            ValueError: ekstensi file tidak didukung parser mana pun.
            FileNotFoundError: file tidak ditemukan.
        """
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        extension = path.suffix.lower()
        parser = self._parser_by_extension.get(extension)
        if parser is None:
            raise ValueError(
                f"Format '{extension}' tidak didukung (file: {path.name}). "
                f"Format yang didukung: {sorted(self.supported_extensions())}"
            )

        return parser.parse(path)

    def load_all(self, folder: Path) -> list[RawDocument]:
        """Muat semua file berformat didukung di dalam folder (tidak rekursif ke subfolder).

        File berekstensi tidak dikenal dilewati dengan peringatan, BUKAN
        menghentikan seluruh proses -- satu file rusak/format aneh tidak
        boleh menggagalkan ingest seluruh koleksi dokumen lain (Aturan #25).
        """
        if not folder.exists():
            raise FileNotFoundError(f"Folder dokumen tidak ditemukan: {folder}")

        documents: list[RawDocument] = []
        all_files = sorted(p for p in folder.iterdir() if p.is_file())

        if not all_files:
            logger.warning(f"Folder dokumen kosong: {folder}")
            return documents

        for file_path in all_files:
            extension = file_path.suffix.lower()
            if extension not in self._parser_by_extension:
                logger.warning(
                    f"Melewati file dengan format tidak didukung: {file_path.name}"
                )
                continue

            try:
                raw_document = self.load_one(file_path)
                documents.append(raw_document)
                logger.info(f"Berhasil memuat: {file_path.name}")
            except Exception as exc:
                # File rusak/terkunci tidak boleh menghentikan file lain
                # (Error Conditions Bagian 4.10: "file rusak; file terkunci").
                logger.error(f"Gagal memuat {file_path.name}: {exc}")
                continue

        return documents
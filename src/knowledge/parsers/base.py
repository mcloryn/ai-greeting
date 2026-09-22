"""
src/knowledge/parsers/base.py

Interface (kontrak) yang wajib dipenuhi SETIAP parser format dokumen.

Kenapa perlu interface ini (Aturan #6, Modular wajib):
DocumentLoader tidak boleh tahu detail cara kerja masing-masing parser.
Ia hanya perlu tahu bahwa setiap parser punya method parse() yang
menerima path file dan mengembalikan RawDocument -- bentuk yang SAMA,
apa pun format aslinya (.txt, .md, .pdf, dst).

Dengan begitu, menambah format baru (misal .docx di masa depan) cukup
membuat class parser baru yang mewarisi DocumentParser ini -- tidak
perlu mengubah DocumentLoader maupun Cleaner maupun Chunker sama sekali.
Ini persis prinsip di Bagian 21 Future Extensions Build Plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.models import RawDocument


class DocumentParser(ABC):
    """Kontrak dasar untuk semua parser format dokumen.

    ABC (Abstract Base Class) dipakai di sini, bukan sekadar konvensi
    lewat docstring, supaya Python SENDIRI yang menolak menjalankan
    program kalau ada parser baru yang lupa mengimplementasikan salah
    satu method wajib di bawah -- errornya muncul saat class dibuat,
    bukan baru ketahuan belakangan saat parser itu dipakai.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> RawDocument:
        """Baca satu file dan ubah jadi RawDocument.

        Args:
            file_path: path menuju file yang akan diparse.

        Returns:
            RawDocument berisi teks per halaman/bagian beserta metadata
            dasar (nama file, ekstensi). Belum dibersihkan (itu tugas
            Cleaner) dan belum dipotong-potong (itu tugas Chunker).

        Raises:
            Implementasi masing-masing parser boleh melempar exception
            sendiri (mis. FileNotFoundError, ValueError untuk file
            rusak) -- DocumentLoader yang akan menangkapnya nanti,
            sesuai Error Conditions di Bagian 4.10 Build Plan.
        """
        raise NotImplementedError

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Ekstensi file apa saja yang bisa ditangani parser ini.

        Contoh: {".txt"} untuk parser teks polos, {".md", ".markdown"}
        untuk parser markdown. DocumentLoader memakai ini untuk memilih
        parser yang tepat berdasarkan ekstensi file (Task 5 CP10).
        """
        raise NotImplementedError
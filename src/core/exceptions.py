"""
src/core/exceptions.py

Exception khusus project, supaya error dari komponen kita bisa dibedakan
dari error pustaka pihak ketiga (mis. cv2.error, requests.Timeout).

core/ tidak boleh meng-import module lain (Aturan #6 Build Plan).
"""

from __future__ import annotations


class AppError(Exception):
    """Kelas dasar untuk semua exception khusus project ini."""


class CameraError(AppError):
    """Gagal membuka, membaca, atau menutup perangkat kamera."""


class DetectorError(AppError):
    """Gagal memuat atau menjalankan model deteksi (YOLO)."""


class KnowledgeError(AppError):
    """Gagal memuat, memproses, atau mengindeks dokumen pengetahuan."""


class RetrievalError(AppError):
    """Gagal melakukan pencarian di vector store."""


class LLMError(AppError):
    """Gagal memanggil atau mendapat respons dari penyedia LLM."""


class TTSError(AppError):
    """Gagal mensintesis atau memutar audio hasil text-to-speech."""

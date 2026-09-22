"""
src/core/models.py

Seluruh dataclass "kontrak data" dari Bagian 5 Build Plan. Hanya bentuk
data (field) -- TIDAK ADA logika bisnis di file ini (lihat catatan di
Bagian 5 Build Plan: "definisi bentuk data, bukan implementasi").

core/ tidak boleh meng-import module lain (Aturan #6 Build Plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.enums import AppState, ColorName, LockPhase, ReleaseReason


# --------------------------------------------------------------------------
# Lapisan Vision (Bagian 5.1 - 5.8)
# --------------------------------------------------------------------------

@dataclass
class Frame:
    """Satu frame mentah dari kamera. Dibuat oleh Camera."""

    image: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int


@dataclass(frozen=True)
class BoundingBox:
    """Kotak posisi objek dalam piksel. Dibuat oleh PersonDetector."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(frozen=True)
class Detection:
    """Satu hasil deteksi YOLO mentah. Dibuat oleh PersonDetector.

    `mask_polygon` opsional -- hanya terisi kalau model yang dipakai
    adalah varian segmentasi (mis. yolo11n-seg.pt). Disimpan sebagai
    tuple[tuple[int, int], ...] (koordinat piksel absolut di frame),
    BUKAN np.ndarray: Detection ini frozen=True, jadi Python otomatis
    membuat __eq__/__hash__ dari semua field -- np.ndarray akan membuat
    perbandingan (`==`) menghasilkan array, bukan bool, dan meledak saat
    dipakai di set/dict atau dibandingkan. Tuple aman untuk itu.

    Modul yang tidak butuh mask (Tracker, TargetSelector, dst.) cukup
    mengabaikan field ini -- kontrak lama (bbox, confidence, dst.) sama
    sekali tidak berubah.
    """

    bbox: BoundingBox
    confidence: float
    class_id: int
    frame_id: int
    track_id: int | None = None
    mask_polygon: tuple[tuple[int, int], ...] | None = None


@dataclass
class TrackedPerson:
    track_id: int
    bbox: BoundingBox
    confidence: float
    first_seen: float
    last_seen: float
    age_frames: int
    mask_polygon: tuple[tuple[int, int], ...] | None = None  # BARU


@dataclass
class TargetCandidate:
    """Skor satu kandidat target. Dibuat oleh TargetSelector."""

    track_id: int
    score: float
    area_score: float
    center_score: float
    is_eligible: bool


@dataclass
class LockState:
    state: LockPhase
    track_id: int | None = None
    bbox: BoundingBox | None = None
    locked_since: float | None = None
    last_seen: float | None = None
    lost_duration: float = 0.0
    release_reason: ReleaseReason | None = None
    mask_polygon: tuple[tuple[int, int], ...] | None = None


@dataclass
class ColorResult:
    """Hasil klasifikasi warna satu frame. Dibuat oleh ClothingColorDetector."""

    color_name: ColorName | None
    confidence: float
    hsv_median: tuple[int, int, int]
    pixel_count: int
    roi_valid: bool


@dataclass
class StableColor:
    """Hasil warna yang sudah distabilkan lewat buffer. Dibuat oleh ColorStabilizer."""

    color_name: ColorName | None
    ratio: float
    sample_count: int
    is_stable: bool


# --------------------------------------------------------------------------
# Lapisan Knowledge (Bagian 5.9 - 5.10)
# --------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Satu potongan teks siap embed. Dibuat oleh Chunker."""

    chunk_id: str
    text: str
    source_file: str
    page: int | None
    section: str | None
    char_count: int
    ingested_at: str


@dataclass
class RetrievedChunk:
    """Satu hasil pencarian vector store. Dibuat oleh Retriever."""

    chunk: DocumentChunk
    score: float
    rank: int
    passed_threshold: bool

@dataclass
class CleanPage:
    """Satu halaman/bagian setelah dibersihkan Cleaner."""
    text: str
    page_number: int | None


@dataclass
class CleanDocument:
    """Dokumen yang teksnya sudah bersih, siap dipotong Chunker.

    Dibuat oleh: DocumentProcessor (Cleaner).
    Dipakai oleh: Chunker.
    """
    source_file: str
    extension: str
    pages: list[CleanPage]


# --------------------------------------------------------------------------
# Lapisan AI (Bagian 5.11 - 5.13)
# --------------------------------------------------------------------------

@dataclass
class RawPage:
    """Satu halaman/bagian teks mentah hasil parsing, sebelum dibersihkan.

    Untuk format tanpa konsep halaman (.txt, .md), page_number bernilai
    None dan seluruh isi file dianggap satu RawPage.

    Catatan: field ini BELUM ada di Bagian 5 Build Plan asli -- ditambah
    di CP10 karena Bagian 4.10/4.11 menyebut RawDocument sebagai output
    Parser/DocumentLoader tanpa merinci bentuknya. Keputusan: simpan teks
    per halaman (bukan satu string gabungan) supaya nomor halaman PDF
    (Task 4 CP10) tidak hilang sebelum sempat dipakai Chunker.
    """

    text: str
    page_number: int | None


@dataclass
class RawDocument:
    """Hasil satu file yang sudah diparse, sebelum masuk Cleaner.

    Dibuat oleh: Parser (lewat DocumentLoader).
    Dipakai oleh: DocumentProcessor (Cleaner).
    """

    source_file: str
    extension: str
    pages: list[RawPage]

    
@dataclass
class RAGContext:
    """Konteks yang dirakit RAGPipeline sebelum dikirim ke PromptBuilder."""

    question: str
    chunks: list[RetrievedChunk]
    history: list[dict]
    has_context: bool
    top_score: float


@dataclass
class LLMResponse:
    """Jawaban final dari LLMService / RAGPipeline."""

    text: str
    sources: list[str]
    used_context: bool
    is_fallback: bool
    latency_ms: int
    provider: str
    error: str | None = None


@dataclass
class TTSResponse:
    """Hasil sintesis suara dari TTSService."""

    audio_path: str | None
    duration_sec: float
    from_cache: bool
    success: bool
    error: str | None = None


# --------------------------------------------------------------------------
# Lapisan Application (Bagian 5.14 - 5.15)
# --------------------------------------------------------------------------

@dataclass
class ConversationSession:
    """Satu sesi percakapan dari sapaan sampai penutup. Dibuat oleh ConversationManager."""

    session_id: str
    track_id: int
    clothing_color: str | None
    started_at: float
    last_activity_at: float
    history: list[dict] = field(default_factory=list)
    turn_count: int = 0
    unanswered_count: int = 0
    end_reason: str | None = None


@dataclass
class Event:
    """Satu event yang mengalir ke StateMachine / Orchestrator."""

    type: str
    payload: dict
    timestamp: float
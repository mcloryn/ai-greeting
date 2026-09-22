"""
config/settings.py

Memuat konfigurasi dari dua sumber:
  - config/config.yaml  -> semua angka & pengaturan non-rahasia
  - .env                -> rahasia (API key)

lalu memvalidasi nilainya dan menyediakan SATU objek konfigurasi (`Settings`)
yang dipakai oleh seluruh module lain. Tidak ada module lain yang boleh
membaca config.yaml atau .env secara langsung -- semua lewat sini.

Ini memenuhi Aturan #9 dan #10 Build Plan: layanan eksternal & angka ajaib
dikonfigurasi di satu tempat, bukan tersebar di kode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Error khusus untuk kesalahan konfigurasi.

    Sengaja dibuat di sini (bukan di src/core/exceptions.py) karena
    settings.py harus bisa berdiri sendiri tanpa bergantung pada src/.
    """


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _require(mapping: dict[str, Any], dotted_key: str) -> Any:
    """Ambil nilai dari dict bersarang pakai key bertitik, mis. "vision.detection.model_name".

    Melempar ConfigError yang jelas (bukan KeyError mentah) kalau hilang.
    """
    node: Any = mapping
    parts = dotted_key.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            path_so_far = ".".join(parts[: i + 1])
            raise ConfigError(
                f"Konfigurasi wajib '{dotted_key}' tidak ditemukan "
                f"(berhenti di '{path_so_far}'). Periksa config/config.yaml."
            )
        node = node[part]
    return node


def _require_type(value: Any, expected_type: type, key: str) -> Any:
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"Konfigurasi '{key}' harus bertipe {expected_type.__name__}, "
            f"tapi ditemukan {type(value).__name__} (nilai: {value!r})."
        )
    return value


def _require_range(value: float, low: float, high: float, key: str) -> float:
    if not (low <= value <= high):
        raise ConfigError(
            f"Konfigurasi '{key}' harus berada di rentang [{low}, {high}], "
            f"tapi nilainya {value}."
        )
    return value
def _require_str_list(value: Any, key: str, min_items: int) -> list[str]:
    _require_type(value, list, key)
    if len(value) < min_items:
        raise ConfigError(f"Konfigurasi '{key}' minimal berisi {min_items} item, ditemukan {len(value)}.")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"Konfigurasi '{key}' hanya boleh berisi teks tidak kosong, ditemukan {item!r}.")
    return value

@dataclass(frozen=True)
class CameraSettings:
    device_index: int
    requested_width: int
    requested_height: int
    fps_target: int
    fps_window: int

@dataclass(frozen=True)
class ClothingColorSettings:
    palette: list[str]
    stabilizer_buffer_sec: float
    stabilizer_majority_ratio: float
    color_wait_timeout_sec: float
    use_mask: bool
    roi_top_ratio: float
    roi_bottom_ratio: float
    roi_left_ratio: float
    roi_right_ratio: float
    roi_min_pixel_count: int
    pixel_v_min: int
    pixel_v_max: int
    achromatic_v_black_max: int
    achromatic_s_gray_max: int
    hue_ranges: dict[str, list[tuple[int, int]]]
    shoulder_width_ratio: float          # BARU
    shoulder_top_margin_ratio: float     # BARU
    shoulder_torso_height_ratio: float   # BARU

@dataclass(frozen=True)
class VisionSettings:
    camera: CameraSettings
    clothing_color: ClothingColorSettings
    model_name: str
    person_class_id: int
    confidence_threshold: float
    min_age_frames: int
    weight_area: float
    weight_center: float
    min_area_ratio: float
    confirm_timer_sec: float
    grace_period_sec: float
    cooldown_sec: float

@dataclass(frozen=True)
class GreetingSettings:
    templates: list[str]
    fallback_templates: list[str]
    color_display_names: dict[str, str]
    pagi_start_hour: int
    siang_start_hour: int
    sore_start_hour: int
    malam_start_hour: int


@dataclass(frozen=True)
class ApplicationSettings:
    idle_timeout_sec: float
    max_history_turns: int
    farewell_words: list[str]
    state_timeouts_sec: dict[str, float]
    greeting: GreetingSettings

@dataclass(frozen=True)
class KnowledgeSettings:
    chunk_size_words: int
    chunk_overlap_words: int
    embedding_model_name: str
    vector_store_dir: str
    vector_store_collection: str


@dataclass(frozen=True)
class RAGSettings:
    top_k: int
    similarity_threshold: float


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_sec: float
    api_key: str | None  # dari .env, boleh None di CP01 (belum dipakai)


@dataclass(frozen=True)
class TTSSettings:
    provider: str
    voice: str
    cache_dir: str


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    log_file: str


@dataclass(frozen=True)
class Settings:
    """Satu objek konfigurasi tervalidasi untuk seluruh aplikasi."""

    vision: VisionSettings
    application: ApplicationSettings
    knowledge: KnowledgeSettings
    rag: RAGSettings
    llm: LLMSettings
    tts: TTSSettings
    logging: LoggingSettings
    project_root: Path = field(default=PROJECT_ROOT)

    def summary(self) -> str:
        """Ringkasan singkat untuk dicetak main.py -- tidak mencetak API key."""
        lines = [
            "=== Ringkasan Konfigurasi ===",
            f"Vision  : model={self.vision.model_name}, "
            f"conf_threshold={self.vision.confidence_threshold}, "
            f"palet_warna={len(self.vision.clothing_color.palette)} warna",
            f"App     : idle_timeout={self.application.idle_timeout_sec}s",
            f"RAG     : top_k={self.rag.top_k}, "
            f"threshold={self.rag.similarity_threshold}",
            f"LLM     : provider={self.llm.provider}, model={self.llm.model_name}, "
            f"api_key_ada={'ya' if self.llm.api_key else 'BELUM DIISI'}",
            f"TTS     : provider={self.tts.provider}, voice={self.tts.voice}",
            f"Logging : level={self.logging.level}, file={self.logging.log_file}",
        ]
        return "\n".join(lines)

def _parse_hue_ranges(raw_ranges: dict) -> dict[str, list[tuple[int, int]]]:
    """Ubah dict YAML {warna: [[min,max], ...]} jadi dict{warna: [(min,max), ...]},
    dengan validasi tipe supaya salah format ketahuan cepat, bukan crash
    samar saat dipakai nanti di clothing_color.py.
    """
    parsed: dict[str, list[tuple[int, int]]] = {}
    for color_name, ranges in raw_ranges.items():
        if not isinstance(ranges, list):
            raise ConfigError(
                f"vision.clothing_color.hue_ranges.{color_name} harus berupa "
                f"list rentang, mis. [[0, 10]]."
            )
        parsed_ranges = []
        for r in ranges:
            if not (isinstance(r, list) and len(r) == 2):
                raise ConfigError(
                    f"vision.clothing_color.hue_ranges.{color_name} berisi "
                    f"rentang tidak valid: {r!r}. Harus [min, max]."
                )
            parsed_ranges.append((int(r[0]), int(r[1])))
        parsed[color_name] = parsed_ranges
    return parsed

def load_settings() -> Settings:
    """Titik masuk utama: baca .env + config.yaml, validasi, kembalikan Settings.

    Melempar ConfigError dengan pesan jelas kalau ada yang salah -- sesuai
    Definition of Done CP01 ("pesan error yang jelas, bukan traceback mentah").
    """
    load_dotenv(dotenv_path=ENV_PATH)

    if not CONFIG_YAML_PATH.exists():
        raise ConfigError(
            f"File konfigurasi tidak ditemukan: {CONFIG_YAML_PATH}. "
            "Pastikan config/config.yaml ada."
        )

    with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml kosong atau formatnya tidak valid.")

    camera = CameraSettings(
        device_index=_require_type(_require(raw, "vision.camera.device_index"), int,
                                    "vision.camera.device_index"),
        requested_width=_require_type(_require(raw, "vision.camera.width"), int,
                                       "vision.camera.width"),
        requested_height=_require_type(_require(raw, "vision.camera.height"), int,
                                        "vision.camera.height"),
        fps_target=_require_type(_require(raw, "vision.camera.fps_target"), int,
                                  "vision.camera.fps_target"),
        fps_window=_require_type(_require(raw, "vision.camera.fps_window"), int,
                                  "vision.camera.fps_window"),
    )

    clothing_color = ClothingColorSettings(
        palette=_require_type(_require(raw, "vision.clothing_color.palette"), list,
                               "vision.clothing_color.palette"),
        stabilizer_buffer_sec=_require_type(
            _require(raw, "vision.clothing_color.stabilizer_buffer_sec"), float,
            "vision.clothing_color.stabilizer_buffer_sec"),
        stabilizer_majority_ratio=_require_range(
            _require_type(_require(raw, "vision.clothing_color.stabilizer_majority_ratio"), float,
                          "vision.clothing_color.stabilizer_majority_ratio"),
            0.0, 1.0, "vision.clothing_color.stabilizer_majority_ratio"),
        color_wait_timeout_sec=_require_type(
            _require(raw, "vision.clothing_color.color_wait_timeout_sec"), float,
            "vision.clothing_color.color_wait_timeout_sec"),
        use_mask=_require_type(_require(raw, "vision.clothing_color.use_mask"), bool,
                                "vision.clothing_color.use_mask"),
        roi_top_ratio=_require_type(_require(raw, "vision.clothing_color.roi.top_ratio"), float,
                                     "vision.clothing_color.roi.top_ratio"),
        roi_bottom_ratio=_require_type(_require(raw, "vision.clothing_color.roi.bottom_ratio"), float,
                                        "vision.clothing_color.roi.bottom_ratio"),
        roi_left_ratio=_require_type(_require(raw, "vision.clothing_color.roi.left_ratio"), float,
                                      "vision.clothing_color.roi.left_ratio"),
        roi_right_ratio=_require_type(_require(raw, "vision.clothing_color.roi.right_ratio"), float,
                                       "vision.clothing_color.roi.right_ratio"),
        roi_min_pixel_count=_require_type(_require(raw, "vision.clothing_color.roi.min_pixel_count"), int,
                                           "vision.clothing_color.roi.min_pixel_count"),
        pixel_v_min=_require_type(_require(raw, "vision.clothing_color.pixel_filter.v_min"), int,
                                   "vision.clothing_color.pixel_filter.v_min"),
        pixel_v_max=_require_type(_require(raw, "vision.clothing_color.pixel_filter.v_max"), int,
                                   "vision.clothing_color.pixel_filter.v_max"),
        achromatic_v_black_max=_require_type(
            _require(raw, "vision.clothing_color.achromatic.v_black_max"), int,
            "vision.clothing_color.achromatic.v_black_max"),
        achromatic_s_gray_max=_require_type(
            _require(raw, "vision.clothing_color.achromatic.s_gray_max"), int,
            "vision.clothing_color.achromatic.s_gray_max"),
        hue_ranges=_parse_hue_ranges(
            _require_type(_require(raw, "vision.clothing_color.hue_ranges"), dict,
                          "vision.clothing_color.hue_ranges")),
                shoulder_width_ratio=_require_type(
            _require(raw, "vision.clothing_color.shoulder.width_ratio"), float,
            "vision.clothing_color.shoulder.width_ratio"),
        shoulder_top_margin_ratio=_require_type(
            _require(raw, "vision.clothing_color.shoulder.top_margin_ratio"), float,
            "vision.clothing_color.shoulder.top_margin_ratio"),
        shoulder_torso_height_ratio=_require_type(
            _require(raw, "vision.clothing_color.shoulder.torso_height_ratio"), float,
            "vision.clothing_color.shoulder.torso_height_ratio"),
    )
        
    vision = VisionSettings(
        camera=camera,
        clothing_color=clothing_color,
        model_name=_require_type(_require(raw, "vision.detection.model_name"), str,
                                  "vision.detection.model_name"),
        person_class_id=_require_type(_require(raw, "vision.detection.person_class_id"), int,
                                       "vision.detection.person_class_id"),
        confidence_threshold=_require_range(
            _require_type(_require(raw, "vision.detection.confidence_threshold"), float,
                          "vision.detection.confidence_threshold"),
            0.0, 1.0, "vision.detection.confidence_threshold"),
        min_age_frames=_require_type(_require(raw, "vision.tracker.min_age_frames"), int,
                                      "vision.tracker.min_age_frames"),
        weight_area=_require_type(_require(raw, "vision.target_selector.weight_area"), float,
                                   "vision.target_selector.weight_area"),
        weight_center=_require_type(_require(raw, "vision.target_selector.weight_center"), float,
                                     "vision.target_selector.weight_center"),
        min_area_ratio=_require_type(_require(raw, "vision.target_selector.min_area_ratio"), float,
                                     "vision.target_selector.min_area_ratio"),
        confirm_timer_sec=_require_type(_require(raw, "vision.target_lock.confirm_timer_sec"), float,
                                         "vision.target_lock.confirm_timer_sec"),
        grace_period_sec=_require_type(_require(raw, "vision.target_lock.grace_period_sec"), float,
                                        "vision.target_lock.grace_period_sec"),
        cooldown_sec=_require_type(_require(raw, "vision.target_lock.cooldown_sec"), (int, float),
                                    "vision.target_lock.cooldown_sec"),
    )

    display_names = _require_type(_require(raw, "application.greeting.color_display_names"), dict,
                                  "application.greeting.color_display_names")
    for name, spoken in display_names.items():
        if not isinstance(name, str) or not isinstance(spoken, str) or not spoken.strip():
            raise ConfigError(
                f"application.greeting.color_display_names tidak valid pada {name!r}: {spoken!r}."
            )

    def _hour(name: str) -> int:
        key = f"application.greeting.time_of_day.{name}_start_hour"
        return _require_range(_require_type(_require(raw, key), int, key), 0, 23, key)

    greeting = GreetingSettings(
        templates=_require_str_list(_require(raw, "application.greeting.templates"),
                                    "application.greeting.templates", 3),
        fallback_templates=_require_str_list(_require(raw, "application.greeting.fallback_templates"),
                                             "application.greeting.fallback_templates", 2),
        color_display_names=dict(display_names),
        pagi_start_hour=_hour("pagi"),
        siang_start_hour=_hour("siang"),
        sore_start_hour=_hour("sore"),
        malam_start_hour=_hour("malam"),
    )
    if not (greeting.pagi_start_hour < greeting.siang_start_hour
            < greeting.sore_start_hour < greeting.malam_start_hour):
        raise ConfigError(
            "application.greeting.time_of_day harus berurutan: "
            "pagi < siang < sore < malam."
        )

    application = ApplicationSettings(
        greeting=greeting,
        idle_timeout_sec=_require_type(_require(raw, "application.conversation.idle_timeout_sec"),
                                        (int, float), "application.conversation.idle_timeout_sec"),
        max_history_turns=_require_type(_require(raw, "application.conversation.max_history_turns"),
                                         int, "application.conversation.max_history_turns"),
        farewell_words=_require_str_list(_require(raw, "application.conversation.farewell_words"),
                                  "application.conversation.farewell_words", 2),
        state_timeouts_sec=_require_type(_require(raw, "application.state_timeouts_sec"), dict,
                                          "application.state_timeouts_sec"),
    )

    knowledge = KnowledgeSettings(
        chunk_size_words=_require_type(_require(raw, "knowledge.chunking.chunk_size_words"), int,
                                        "knowledge.chunking.chunk_size_words"),
        chunk_overlap_words=_require_type(_require(raw, "knowledge.chunking.chunk_overlap_words"), int,
                                           "knowledge.chunking.chunk_overlap_words"),
        embedding_model_name=_require_type(_require(raw, "knowledge.embedding.model_name"), str,
                                            "knowledge.embedding.model_name"),
        vector_store_dir=_require_type(_require(raw, "knowledge.vector_store.persist_directory"), str,
                                        "knowledge.vector_store.persist_directory"),
        vector_store_collection=_require_type(_require(raw, "knowledge.vector_store.collection_name"), str,
                                               "knowledge.vector_store.collection_name"),
    )

    if knowledge.chunk_overlap_words >= knowledge.chunk_size_words:
        raise ConfigError(
            "knowledge.chunking.chunk_overlap_words harus lebih kecil dari "
            "chunk_size_words (overlap tidak boleh >= ukuran chunk)."
        )

    rag = RAGSettings(
        top_k=_require_type(_require(raw, "rag.top_k"), int, "rag.top_k"),
        similarity_threshold=_require_range(
            _require_type(_require(raw, "rag.similarity_threshold"), float, "rag.similarity_threshold"),
            0.0, 1.0, "rag.similarity_threshold"),
    )

    llm = LLMSettings(
        provider=_require_type(_require(raw, "llm.provider"), str, "llm.provider"),
        model_name=_require_type(_require(raw, "llm.model_name"), str, "llm.model_name"),
        temperature=_require_type(_require(raw, "llm.temperature"), float, "llm.temperature"),
        max_tokens=_require_type(_require(raw, "llm.max_tokens"), int, "llm.max_tokens"),
        timeout_sec=_require_type(_require(raw, "llm.timeout_sec"), (int, float), "llm.timeout_sec"),
        api_key=os.getenv("GEMINI_API_KEY") or None,
    )

    tts = TTSSettings(
        provider=_require_type(_require(raw, "tts.provider"), str, "tts.provider"),
        voice=_require_type(_require(raw, "tts.voice"), str, "tts.voice"),
        cache_dir=_require_type(_require(raw, "tts.cache_directory"), str, "tts.cache_directory"),
    )

    logging_settings = LoggingSettings(
        level=_require_type(_require(raw, "logging.level"), str, "logging.level"),
        log_file=_require_type(_require(raw, "logging.log_file"), str, "logging.log_file"),
    )

    return Settings(
        vision=vision,
        application=application,
        knowledge=knowledge,
        rag=rag,
        llm=llm,
        tts=tts,
        logging=logging_settings,
    )
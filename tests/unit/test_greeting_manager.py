"""
tests/unit/test_greeting_manager.py

Pengujian GreetingManager (CP08). Tanpa kamera, tanpa internet, tanpa sleep:
waktu selalu disuntikkan.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from config.settings import GreetingSettings, load_settings
from src.application import greeting_manager as gm_module
from src.application.greeting_manager import GreetingManager
from src.core.enums import ColorName

COOLDOWN = 30.0

COLORED = [
    "Selamat {waktu}! Baju {warna} A",
    "Halo, {waktu}! Baju {warna} B",
    "Hai {waktu}! Baju {warna} C",
]
FALLBACK = ["Selamat {waktu}! X", "Halo {waktu}! Y"]


def ts(hour: int, minute: int = 30) -> float:
    """Timestamp pada jam tertentu di hari yang tetap."""
    return datetime(2026, 9, 19, hour, minute).timestamp()


@pytest.fixture
def cfg() -> GreetingSettings:
    return GreetingSettings(
        templates=list(COLORED),
        fallback_templates=list(FALLBACK),
        color_display_names={c.value: c.value for c in ColorName},
        pagi_start_hour=4, siang_start_hour=11, sore_start_hour=15, malam_start_hour=18,
    )


@pytest.fixture
def greeter(cfg) -> GreetingManager:
    return GreetingManager(cfg, COOLDOWN)


# --------------------------------------------------------------- waktu hari

@pytest.mark.parametrize(
    "hour,expected",
    [(0, "malam"), (3, "malam"), (4, "pagi"), (10, "pagi"), (11, "siang"),
     (14, "siang"), (15, "sore"), (17, "sore"), (18, "malam"), (23, "malam")],
)
def test_time_of_day_boundaries(greeter, hour, expected):
    assert greeter.time_of_day(ts(hour)) == expected


# ---------------------------------------------------------------- membangun

def test_build_with_color_fills_all_placeholders(greeter):
    text = greeter.build(ColorName.MERAH, ts(9))
    assert "merah" in text and "pagi" in text
    assert "{" not in text and "}" not in text


def test_build_accepts_plain_string_color(greeter):
    assert "biru" in greeter.build("biru", ts(9))


def test_build_uses_display_name(cfg):
    names = dict(cfg.color_display_names, putih_abu="putih keabu-abuan")
    g = GreetingManager(replace(cfg, color_display_names=names), COOLDOWN)
    assert "putih keabu-abuan" in g.build(ColorName.PUTIH_ABU, ts(9))


def test_build_without_color_uses_fallback(greeter):
    text = greeter.build(None, ts(20))
    assert text in {t.format(waktu="malam") for t in FALLBACK}
    assert "Baju" not in text


def test_unknown_color_falls_back_instead_of_crashing(greeter):
    text = greeter.build("pink", ts(9))
    assert text in {t.format(waktu="pagi") for t in FALLBACK}


def test_colored_templates_rotate_and_wrap(greeter):
    texts = [greeter.build("merah", ts(9)) for _ in range(len(COLORED) + 1)]
    assert len(set(texts[:3])) == 3          # tiga sapaan berbeda berturut-turut
    assert texts[3] == texts[0]              # lalu kembali ke template pertama


def test_fallback_rotation_is_independent_of_colored(greeter):
    first = greeter.build(None, ts(9))
    greeter.build("merah", ts(9))            # tidak boleh menggeser giliran cadangan
    second = greeter.build(None, ts(9))
    assert first != second


def test_build_does_not_mark_anyone_greeted(greeter):
    greeter.build("merah", ts(9))
    assert greeter.tracked_count == 0


# ---------------------------------------------------------------- cooldown

def test_new_person_should_be_greeted(greeter):
    assert greeter.should_greet(1, 100.0) is True


def test_greeted_person_is_not_greeted_again_within_cooldown(greeter):
    greeter.mark_greeted(1, 100.0)
    assert greeter.should_greet(1, 100.0) is False
    assert greeter.should_greet(1, 110.0) is False
    assert greeter.should_greet(1, 129.9) is False


def test_person_is_greeted_again_after_cooldown(greeter):
    greeter.mark_greeted(1, 100.0)
    assert greeter.should_greet(1, 130.0) is True


def test_cooldown_is_per_person(greeter):
    greeter.mark_greeted(1, 100.0)
    assert greeter.should_greet(2, 101.0) is True


def test_marking_again_restarts_cooldown(greeter):
    greeter.mark_greeted(1, 100.0)
    greeter.mark_greeted(1, 140.0)
    assert greeter.should_greet(1, 150.0) is False
    assert greeter.should_greet(1, 170.0) is True


def test_purge_removes_only_expired_entries(greeter):
    greeter.mark_greeted(1, 100.0)
    greeter.mark_greeted(2, 125.0)
    removed = greeter.purge_expired(131.0)   # id 1 kedaluwarsa, id 2 belum
    assert removed == 1
    assert greeter.tracked_count == 1
    assert greeter.should_greet(2, 131.0) is False


def test_cooldown_list_does_not_grow_forever(greeter):
    for i in range(500):
        greeter.mark_greeted(i, float(i))
    greeter.purge_expired(1000.0)
    assert greeter.tracked_count == 0


# ------------------------------------------------------- validasi konfigurasi

def test_unknown_placeholder_is_rejected_at_startup(cfg):
    bad = replace(cfg, templates=COLORED[:2] + ["Halo {nama}, baju {warna}"])
    with pytest.raises(ValueError, match="nama"):
        GreetingManager(bad, COOLDOWN)


def test_fallback_template_cannot_use_color_placeholder(cfg):
    bad = replace(cfg, fallback_templates=["Halo baju {warna}", "Halo {waktu}"])
    with pytest.raises(ValueError, match="warna"):
        GreetingManager(bad, COOLDOWN)


def test_colored_template_must_contain_color(cfg):
    bad = replace(cfg, templates=["Selamat {waktu}"] * 3)
    with pytest.raises(ValueError, match="wajib memuat"):
        GreetingManager(bad, COOLDOWN)


def test_empty_positional_placeholder_is_rejected(cfg):
    bad = replace(cfg, fallback_templates=["Halo {}", "Halo {waktu}"])
    with pytest.raises(ValueError):
        GreetingManager(bad, COOLDOWN)


def test_missing_color_display_name_is_rejected(cfg):
    names = dict(cfg.color_display_names)
    del names["ungu"]
    with pytest.raises(ValueError, match="ungu"):
        GreetingManager(replace(cfg, color_display_names=names), COOLDOWN)


def test_negative_cooldown_is_rejected(cfg):
    with pytest.raises(ValueError):
        GreetingManager(cfg, -1.0)


# ------------------------------------------------------------- config asli

def test_real_config_builds_a_working_manager():
    """config.yaml yang ASLI valid dan menghasilkan sapaan tanpa placeholder mentah."""
    settings = load_settings()
    g = GreetingManager(settings.application.greeting, settings.vision.cooldown_sec)
    for color in list(ColorName) + [None]:
        text = g.build(color, ts(9))
        assert text and "{" not in text


# ----------------------------------------------------------- aturan impor

def test_greeting_manager_does_not_import_vision_or_rag():
    source = Path(gm_module.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for name in imported:
        assert not name.startswith(("src.vision", "src.rag", "src.llm", "src.tts")), name
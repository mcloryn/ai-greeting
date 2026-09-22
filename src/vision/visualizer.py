"""
src/vision/visualizer.py

Menggambar hasil deteksi (bounding box + confidence) di atas frame, murni
untuk keperluan visual debugging manusia. Module ini TIDAK mengubah data
apa pun -- hanya menghasilkan gambar baru untuk ditampilkan.

Dipakai oleh script pengujian manual (scripts/check_detection.py dan
seterusnya), bukan oleh alur aplikasi utama saat sudah production-ready
tanpa jendela video.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.models import Detection

_BOX_COLOR = (0, 255, 0)       # hijau, format BGR (OpenCV)
_TEXT_COLOR = (0, 255, 0)
_BOX_THICKNESS = 2


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Gambar kotak + label confidence untuk setiap deteksi.

    Mengembalikan SALINAN gambar (tidak mengubah `image` asli), supaya
    frame asli tetap bersih kalau dibutuhkan module lain nanti.
    """
    output = image.copy()

    for detection in detections:
        bbox = detection.bbox
        cv2.rectangle(
            output,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            _BOX_COLOR,
            _BOX_THICKNESS,
        )

        label = f"person {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (bbox.x1, max(bbox.y1 - 10, 0)),  # max(...,0) cegah teks keluar layar ke atas
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _TEXT_COLOR,
            2,
        )

    return output

def draw_mask_overlay(
    image: np.ndarray,
    detections: list,
    color: tuple[int, int, int] = (255, 0, 255),
    alpha: float = 0.35,
) -> np.ndarray:
    """Gambar overlay semi-transparan untuk mask_polygon tiap Detection
    (kalau ada -- Detection dari model non-seg punya mask_polygon=None
    dan otomatis dilewati, jadi fungsi ini aman dipanggil kapan pun
    tanpa perlu cek dulu model apa yang sedang dipakai).

    Murni untuk debug visual manusia di jendela OpenCV produksi -- TIDAK
    dipakai oleh logika klasifikasi warna yang sesungguhnya di
    clothing_color.py.
    """
    output = image.copy()
    overlay = image.copy()
    any_mask_drawn = False

    for detection in detections:
        mask_polygon = getattr(detection, "mask_polygon", None)
        if not mask_polygon:
            continue
        points = np.array(mask_polygon, dtype=np.int32)
        if points.shape[0] < 3:
            continue
        cv2.fillPoly(overlay, [points], color)
        any_mask_drawn = True

    if any_mask_drawn:
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def draw_tracked_people(
    image: np.ndarray,
    tracked_people: list,
    candidates: list,
    target_id: int | None,
) -> np.ndarray:
    """Gambar kotak untuk setiap TrackedPerson, dengan track_id + skor,
    dan warna berbeda untuk target yang terpilih (kuning) vs bukan (hijau).
    """
    output = image.copy()
    score_by_id = {c.track_id: c for c in candidates}

    for person in tracked_people:
        bbox = person.bbox
        candidate = score_by_id.get(person.track_id)
        is_target = person.track_id == target_id

        color = (0, 255, 255) if is_target else (0, 255, 0)  # kuning vs hijau (BGR)
        thickness = 3 if is_target else 2

        cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)

        score_text = f"ID:{person.track_id} score:{candidate.score:.2f}" if candidate else f"ID:{person.track_id}"
        cv2.putText(output, score_text, (bbox.x1, max(bbox.y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output

def draw_lock_status(image: np.ndarray, lock_state) -> np.ndarray:
    """Gambar fase lock + track_id + durasi di pojok kanan atas, sesuai
    poin 9 Implementation Tasks CP05 ("Tampilkan fase lock, track_id,
    dan durasi di layar").
    """
    output = image.copy()

    phase_text = f"Lock: {lock_state.state.value}"
    if lock_state.track_id is not None:
        phase_text += f"  ID:{lock_state.track_id}"

    if lock_state.state.value == "LOCKED" and lock_state.locked_since is not None:
        phase_text += f"  durasi:{lock_state.last_seen - lock_state.locked_since:.1f}s"
    elif lock_state.state.value == "LOST_GRACE":
        phase_text += f"  hilang:{lock_state.lost_duration:.1f}s"

    color = {
        "IDLE": (128, 128, 128),
        "CANDIDATE": (0, 200, 255),
        "LOCKED": (0, 255, 0),
        "LOST_GRACE": (0, 165, 255),
    }.get(lock_state.state.value, (255, 255, 255))

    cv2.putText(output, phase_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    return output
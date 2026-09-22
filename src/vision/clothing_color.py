"""
src/vision/clothing_color.py

Menentukan satu nama warna pakaian dari ROI torso seseorang, memakai
HSV + rule-based -- BUKAN model AI (Bagian D5 Blueprint: "tidak semua
masalah memerlukan model AI"). Klasifikasi bertingkat: cek dulu apakah
warna tidak-berwarna (hitam/putih-abu) lewat S & V, baru tentukan warna
lewat H (Task 7).

Kontrak: Input Frame + BoundingBox -> Output ColorResult.

CATATAN PERUBAHAN (Task seg-2, perbaikan false-positive kulit wajah):
Sebelumnya ROI torso SELALU dihitung dari rasio tetap terhadap bbox
(roi_top_ratio dst). Ini rapuh: kalau kepala menunduk, jendela rasio
tetap itu bergeser ke wilayah wajah alih-alih kain (terbukti lewat
debug_color_detection.py -- median HSV wajah kena terklasifikasi
"oranye"). Sekarang, KALAU mask tersedia, batas atas torso dicari dari
bentuk mask itu sendiri (baris tempat lebar mask melebar drastis = garis
bahu), bukan tebakan rasio. Kalau garis bahu gagal terdeteksi (mask
tipis/noise), tetap fallback ke rasio lama -- tidak ada regresi untuk
kasus yang sebelumnya sudah benar.

classify_color() dan _classify_hsv() SAMA SEKALI TIDAK DIUBAH.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.settings import ClothingColorSettings
from src.core.enums import ColorName
from src.core.models import BoundingBox, ColorResult, Frame

_NAME_TO_COLOR_ENUM = {
    "merah": ColorName.MERAH,
    "oranye": ColorName.ORANYE,
    "kuning": ColorName.KUNING,
    "hijau": ColorName.HIJAU,
    "biru": ColorName.BIRU,
    "ungu": ColorName.UNGU,
    "hitam": ColorName.HITAM,
    "putih_abu": ColorName.PUTIH_ABU,
}


def _compute_roi_bounds(
    frame_image: np.ndarray, bbox: BoundingBox, config: ClothingColorSettings,
) -> tuple[int, int, int, int] | None:
    """Hitung batas (y1, y2, x1, x2) wilayah dada relatif ke bbox lewat
    RASIO TETAP (Task 1) -- ini jalur LAMA, sekarang dipakai sebagai
    fallback saat mask tidak tersedia atau garis bahu gagal terdeteksi
    (lihat _find_shoulder_offset dan _compute_roi_bounds_from_mask).
    Sudah dijaga supaya tidak keluar frame (Task 2). Mengembalikan None
    kalau hasilnya tidak valid (lebar/tinggi <= 0).
    """
    frame_height, frame_width = frame_image.shape[:2]

    roi_y1 = bbox.y1 + int(bbox.height * config.roi_top_ratio)
    roi_y2 = bbox.y1 + int(bbox.height * config.roi_bottom_ratio)
    roi_x1 = bbox.x1 + int(bbox.width * config.roi_left_ratio)
    roi_x2 = bbox.x1 + int(bbox.width * config.roi_right_ratio)

    roi_y1 = max(0, min(roi_y1, frame_height))
    roi_y2 = max(0, min(roi_y2, frame_height))
    roi_x1 = max(0, min(roi_x1, frame_width))
    roi_x2 = max(0, min(roi_x2, frame_width))

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return None

    return roi_y1, roi_y2, roi_x1, roi_x2


def _find_shoulder_offset(
    full_mask: np.ndarray, bbox: BoundingBox, config: ClothingColorSettings,
) -> int | None:
    """Cari offset-y (relatif ke bbox.y1) tempat lebar mask melebar
    drastis -- penanda transisi leher -> bahu.

    Algoritma: untuk tiap baris di dalam bbox, hitung lebar mask di baris
    itu (jarak piksel-mask paling kiri ke paling kanan). Baris PERTAMA
    dari atas yang lebarnya >= shoulder_width_ratio * lebar_terlebar
    dianggap garis bahu. Kepala jauh lebih sempit daripada bahu+badan,
    jadi transisi ini biasanya tajam.

    Mengembalikan None kalau mask di dalam bbox kosong/terlalu tipis
    untuk dipercaya (pemanggil WAJIB fallback ke rasio tetap saat None).
    """
    y1, y2, x1, x2 = bbox.y1, bbox.y2, bbox.x1, bbox.x2
    region = full_mask[y1:y2, x1:x2]
    if region.size == 0:
        return None

    row_widths = np.zeros(region.shape[0], dtype=np.int32)
    for i, row in enumerate(region):
        cols = np.flatnonzero(row)
        if cols.size > 0:
            row_widths[i] = int(cols[-1] - cols[0] + 1)

    width_max = int(row_widths.max())
    if width_max <= 0:
        return None

    threshold = config.shoulder_width_ratio * width_max
    candidates = np.flatnonzero(row_widths >= threshold)
    if candidates.size == 0:
        return None

    return int(candidates[0])


def _compute_roi_bounds_from_mask(
    frame_image: np.ndarray,
    bbox: BoundingBox,
    full_mask: np.ndarray,
    config: ClothingColorSettings,
) -> tuple[int, int, int, int] | None:
    """Hitung batas ROI torso dari garis bahu (Task seg-2), bukan rasio
    tetap. Mengembalikan None kalau garis bahu gagal terdeteksi ATAU
    hasil batasnya tidak valid -- pemanggil WAJIB fallback ke
    _compute_roi_bounds() (rasio tetap) saat None.
    """
    shoulder_offset = _find_shoulder_offset(full_mask, bbox, config)
    if shoulder_offset is None:
        return None

    frame_height, frame_width = frame_image.shape[:2]
    margin = int(bbox.height * config.shoulder_top_margin_ratio)
    torso_span = int(bbox.height * config.shoulder_torso_height_ratio)

    roi_y1 = bbox.y1 + shoulder_offset + margin
    roi_y2 = roi_y1 + torso_span
    roi_x1 = bbox.x1
    roi_x2 = bbox.x2

    roi_y1 = max(0, min(roi_y1, frame_height))
    roi_y2 = max(0, min(roi_y2, frame_height))
    roi_x1 = max(0, min(roi_x1, frame_width))
    roi_x2 = max(0, min(roi_x2, frame_width))

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return None

    return roi_y1, roi_y2, roi_x1, roi_x2


def extract_torso_roi(
    frame_image: np.ndarray, bbox: BoundingBox, config: ClothingColorSettings,
) -> np.ndarray | None:
    """Potong wilayah dada dari frame berdasarkan proporsi relatif ke bbox
    (Task 1) -- jalur RASIO TETAP murni, dipakai skrip lama/uji A-B.
    Tidak diubah dari versi sebelumnya.
    """
    bounds = _compute_roi_bounds(frame_image, bbox, config)
    if bounds is None:
        return None
    roi_y1, roi_y2, roi_x1, roi_x2 = bounds

    roi = frame_image[roi_y1:roi_y2, roi_x1:roi_x2]

    pixel_count = roi.shape[0] * roi.shape[1]
    if pixel_count < config.roi_min_pixel_count:
        return None

    return roi


def extract_torso_pixels(
    frame_image: np.ndarray,
    bbox: BoundingBox,
    config: ClothingColorSettings,
    mask_polygon: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray | None:
    """Ambil piksel torso.

    Urutan keputusan (Task seg-2):
      1. Kalau mask tersedia & diaktifkan (`use_mask`): cari garis bahu
         dari BENTUK mask itu sendiri -> tentukan jendela ROI dari situ
         (bukan rasio tetap). Ini yang memperbaiki kasus kepala menunduk.
      2. Kalau garis bahu gagal terdeteksi (mask tipis/noise): fallback
         ke jendela ROI rasio tetap (perilaku lama, TIDAK berubah).
      3. Di jendela ROI manapun yang dipakai (dari langkah 1 atau 2):
         kalau piksel mask di dalam jendela itu cukup (>= min_pixel_count),
         pakai HANYA piksel mask (buang latar belakang yang ikut
         terpotong rectangle). Kalau tidak cukup, pakai seluruh rectangle
         jendela itu apa adanya.

    Mengembalikan piksel dalam bentuk (N, 1, 3) BGR (piksel mask) atau
    (H, W, 3) BGR (rectangle) -- classify_color() tidak peduli bentuk
    asal array selama channel terakhirnya 3, jadi TIDAK ADA perubahan
    di classify_color()/_classify_hsv().
    """
    use_mask = getattr(config, "use_mask", False) and mask_polygon is not None

    full_mask = None
    dynamic_bounds = None
    if use_mask:
        frame_height, frame_width = frame_image.shape[:2]
        full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        polygon_points = np.array(mask_polygon, dtype=np.int32)
        if polygon_points.shape[0] >= 3:
            cv2.fillPoly(full_mask, [polygon_points], 255)
        dynamic_bounds = _compute_roi_bounds_from_mask(frame_image, bbox, full_mask, config)

    if dynamic_bounds is not None:
        roi_y1, roi_y2, roi_x1, roi_x2 = dynamic_bounds
    else:
        # Garis bahu tidak terdeteksi (atau mask tidak tersedia sama
        # sekali) -- balik ke jendela rasio tetap, dan JANGAN pura-pura
        # masih di "mode mask" untuk jendela ini.
        bounds = _compute_roi_bounds(frame_image, bbox, config)
        if bounds is None:
            return None
        roi_y1, roi_y2, roi_x1, roi_x2 = bounds
        use_mask = False

    rect_roi = frame_image[roi_y1:roi_y2, roi_x1:roi_x2]
    if rect_roi.shape[0] * rect_roi.shape[1] < config.roi_min_pixel_count:
        return None

    if not use_mask:
        return rect_roi

    roi_mask = full_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    masked_pixel_count = int(np.count_nonzero(roi_mask))

    if masked_pixel_count < config.roi_min_pixel_count:
        # Mask tidak menutupi cukup area di jendela ini -- mundur jujur
        # ke rectangle penuh jendela yang sama (bukan jendela rasio lama).
        return rect_roi

    selected_pixels = rect_roi[roi_mask.astype(bool)]  # (N, 3)
    return selected_pixels.reshape(-1, 1, 3)


def classify_color(roi: np.ndarray, config: ClothingColorSettings) -> ColorResult:
    """TIDAK DIUBAH dari versi sebelumnya. Klasifikasikan warna dominan
    dari ROI. Mengembalikan ColorResult dengan color_name=None (BUKAN
    menebak) kalau tidak cukup yakin.
    """
    if roi is None or roi.size == 0:
        return ColorResult(
            color_name=None, confidence=0.0, hsv_median=(0, 0, 0),
            pixel_count=0, roi_valid=False,
        )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].flatten()
    s_channel = hsv[:, :, 1].flatten()
    v_channel = hsv[:, :, 2].flatten()

    total_pixel_count = len(v_channel)

    valid_mask = (v_channel >= config.pixel_v_min) & (v_channel <= config.pixel_v_max)
    valid_pixel_count = int(valid_mask.sum())

    if valid_pixel_count == 0:
        h_valid, s_valid, v_valid = h_channel, s_channel, v_channel
        valid_pixel_count = total_pixel_count
        confidence = 0.5
    else:
        h_valid = h_channel[valid_mask]
        s_valid = s_channel[valid_mask]
        v_valid = v_channel[valid_mask]
        confidence = valid_pixel_count / total_pixel_count

    median_h = int(np.median(h_valid))
    median_s = int(np.median(s_valid))
    median_v = int(np.median(v_valid))

    color_name = _classify_hsv(median_h, median_s, median_v, config)

    return ColorResult(
        color_name=color_name,
        confidence=confidence,
        hsv_median=(median_h, median_s, median_v),
        pixel_count=total_pixel_count,
        roi_valid=True,
    )


def _classify_hsv(h: int, s: int, v: int, config: ClothingColorSettings) -> ColorName | None:
    """TIDAK DIUBAH. Klasifikasi bertingkat (Task 7): tidak-berwarna
    dulu, baru hue.
    """
    if v <= config.achromatic_v_black_max:
        return ColorName.HITAM

    if s <= config.achromatic_s_gray_max:
        return ColorName.PUTIH_ABU

    for color_name_str, ranges in config.hue_ranges.items():
        for h_min, h_max in ranges:
            if h_min <= h <= h_max:
                return _NAME_TO_COLOR_ENUM[color_name_str]

    return None


def detect_clothing_color(
    frame: Frame,
    bbox: BoundingBox,
    config: ClothingColorSettings,
    mask_polygon: tuple[tuple[int, int], ...] | None = None,
) -> ColorResult:
    """TIDAK DIUBAH secara kontrak. Fungsi gabungan: ambil piksel torso
    (mask+garis bahu kalau tersedia & aktif, fallback rasio tetap) lalu
    klasifikasikan.
    """
    pixels = extract_torso_pixels(frame.image, bbox, config, mask_polygon)
    if pixels is None:
        return ColorResult(
            color_name=None, confidence=0.0, hsv_median=(0, 0, 0),
            pixel_count=0, roi_valid=False,
        )
    return classify_color(pixels, config)
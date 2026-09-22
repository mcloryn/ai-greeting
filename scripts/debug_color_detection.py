"""
scripts/debug_color_detection.py

Script debug MANDIRI -- tidak mengubah satu baris pun kode produksi
(detector.py, clothing_color.py, settings.py semua dipakai APA ADANYA,
hanya diimpor). Tujuannya cuma satu: menampilkan angka-angka yang lewat
di sepanjang pipeline, dari deteksi sampai klasifikasi warna, supaya
kamu bisa lihat PERSIS di titik mana hasilnya menyimpang dari dugaan.

Cara pakai:
    python scripts/debug_color_detection.py               # webcam
    python scripts/debug_color_detection.py --image foto.jpg
    python scripts/debug_color_detection.py --no-mask      # paksa fallback rectangle (A/B test)

Tekan 'q' untuk keluar (mode webcam), atau tekan tombol apa saja
(mode --image, karena cv2.imshow butuh waitKey untuk menggambar).

Yang dicetak PER ORANG per frame:
    1. Bounding box + confidence dari detector (angka mentah YOLO)
    2. Apakah mask_polygon ada, dan berapa titik polygon-nya
    3. use_mask_config (dibaca langsung dari objek config yang sungguhan
       dipakai -- ini yang paling penting untuk konfirmasi/gugurkan
       dugaan "use_mask tidak pernah nyampe ke config")
    4. Jalur piksel yang akhirnya dipakai: MASK atau RECTANGLE (dan kenapa,
       kalau fallback)
    5. Jumlah piksel yang dipakai (mask vs rectangle, ditampilkan dua-duanya
       kalau mask tersedia, supaya kelihatan bedanya)
    6. HSV median hasil akhir (H, S, V terpisah) SEBELUM masuk _classify_hsv
    7. Hasil klasifikasi bertingkat: apakah kena cabang hitam / putih_abu /
       hue tertentu / None -- dicetak alasan cabang mana yang menang
    8. ColorResult final (color_name, confidence, pixel_count)

Overlay visual (jendela OpenCV):
    - bbox hijau + confidence (deteksi biasa)
    - outline mask kuning kalau mask_polygon ada (supaya kamu lihat
      langsung apakah maskny masuk akal secara visual)
    - teks HSV + nama warna di atas bbox
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# WAJIB sebelum import config/src -- script ini ada di scripts/, jadi root
# project (yang berisi folder config/ dan src/) harus ditambahkan ke
# sys.path dulu. Sama persis dengan pola di scripts/check_detection.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# --- import APA ADANYA dari kode produksi -- tidak ada logika baru di sini,
# script ini cuma "penonton" yang mencetak apa yang terjadi.
from config.settings import load_settings
from src.vision.camera import Camera
from src.vision.detector import PersonDetector
from src.vision import clothing_color as cc
from src.core.models import Frame


def _classify_with_trace(h: int, s: int, v: int, config) -> tuple[str, str]:
    """Reimplementasi READ-ONLY dari urutan pengecekan _classify_hsv,
    HANYA untuk mencetak alasan cabang mana yang menang -- tidak dipakai
    untuk hasil final (hasil final tetap dari cc._classify_hsv yang asli,
    ini murni untuk narasi debug).
    """
    if v <= config.achromatic_v_black_max:
        return "hitam", f"v({v}) <= achromatic_v_black_max({config.achromatic_v_black_max})"
    if s <= config.achromatic_s_gray_max:
        return "putih_abu", f"s({s}) <= achromatic_s_gray_max({config.achromatic_s_gray_max})"
    for color_name_str, ranges in config.hue_ranges.items():
        for h_min, h_max in ranges:
            if h_min <= h <= h_max:
                return color_name_str, f"h({h}) dalam rentang {color_name_str} [{h_min},{h_max}]"
    return "None", f"h({h}) tidak masuk rentang hue manapun"


def debug_one_frame(
    frame_bgr: np.ndarray,
    frame_id: int,
    detector: PersonDetector,
    cc_config,
    force_no_mask: bool,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    frame = Frame(image=frame_bgr, frame_id=frame_id, timestamp=time.time(),
                  width=width, height=height)
    detections = detector.detect(frame)

    print(f"\n===== frame_id={frame_id}  jumlah_deteksi={len(detections)} =====")
    print(f"[config] use_mask attr on config object: "
          f"{getattr(cc_config, 'use_mask', 'MISSING')!r}")

    overlay = frame_bgr.copy()

    for idx, det in enumerate(detections):
        bbox = det.bbox
        mask_polygon = None if force_no_mask else det.mask_polygon

        print(f"\n-- deteksi #{idx} --")
        print(f"  bbox=({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2})  "
              f"confidence={det.confidence:.3f}  track_id={det.track_id}")
        print(f"  mask_polygon: "
              f"{'None (fallback wajib)' if mask_polygon is None else f'{len(mask_polygon)} titik polygon'}")

        # --- ambil piksel rectangle DAN mask (kalau ada) secara terpisah,
        # murni untuk perbandingan angka -- tidak memengaruhi hasil final.
        rect_roi = cc.extract_torso_roi(frame_bgr, bbox, cc_config)
        rect_pixel_count = 0 if rect_roi is None else rect_roi.shape[0] * rect_roi.shape[1]
        print(f"  rectangle ROI: "
              f"{'INVALID/None' if rect_roi is None else f'{rect_roi.shape[1]}x{rect_roi.shape[0]} = {rect_pixel_count}px'}")

        # --- panggil fungsi produksi ASLI (tidak diubah) untuk ambil piksel
        # final yang benar-benar dipakai pipeline
        final_pixels = cc.extract_torso_pixels(frame_bgr, bbox, cc_config, mask_polygon)
        if final_pixels is None:
            print("  -> extract_torso_pixels() = None (ROI ditolak, ColorResult akan kosong)")
            continue

        used_mask = (
            not force_no_mask
            and mask_polygon is not None
            and getattr(cc_config, "use_mask", False)
            and final_pixels.shape != rect_roi.shape
        )
        final_pixel_count = final_pixels.shape[0] * (final_pixels.shape[1] if final_pixels.ndim > 1 else 1)
        print(f"  jalur piksel terpakai: {'MASK' if used_mask else 'RECTANGLE (fallback atau use_mask nonaktif)'}"
              f"  ({final_pixel_count} px dipakai)")

        # --- panggil classify_color() ASLI -- ini fungsi produksi yang sama
        # persis dipakai live, tidak dimodifikasi.
        result = cc.classify_color(final_pixels, cc_config)
        h, s, v = result.hsv_median
        print(f"  HSV median SEBELUM klasifikasi: H={h}  S={s}  V={v}")

        branch, reason = _classify_with_trace(h, s, v, cc_config)
        print(f"  cabang klasifikasi: {branch}   (alasan: {reason})")
        print(f"  ColorResult final: color_name={result.color_name}  "
              f"confidence={result.confidence:.3f}  pixel_count={result.pixel_count}  "
              f"roi_valid={result.roi_valid}")

        # --- tentukan jendela ROI yang SUNGGUHAN dipakai, dengan urutan
        # keputusan PERSIS SAMA dengan extract_torso_pixels() produksi:
        # coba garis-bahu-dari-mask dulu, fallback ke rasio tetap kalau
        # gagal. Ini PENTING -- sebelumnya overlay ini selalu menggambar
        # jendela rasio lama, jadi tidak mencerminkan perbaikan Task seg-2
        # sama sekali walau clothing_color.py sudah diupdate.
        roi_bounds_used = None
        roi_source = "RECTANGLE (rasio tetap)"
        full_mask_for_vis = None

        if used_mask and mask_polygon is not None:
            fh, fw = frame_bgr.shape[:2]
            full_mask_for_vis = np.zeros((fh, fw), dtype=np.uint8)
            poly_pts = np.array(mask_polygon, dtype=np.int32)
            if poly_pts.shape[0] >= 3:
                cv2.fillPoly(full_mask_for_vis, [poly_pts], 255)

            shoulder_offset = cc._find_shoulder_offset(full_mask_for_vis, bbox, cc_config)
            print(f"  shoulder_offset: {shoulder_offset if shoulder_offset is not None else 'TIDAK TERDETEKSI (fallback rasio)'}")

            dyn_bounds = cc._compute_roi_bounds_from_mask(frame_bgr, bbox, full_mask_for_vis, cc_config)
            if dyn_bounds is not None:
                roi_bounds_used = dyn_bounds
                roi_source = "MASK (garis bahu)"

        if roi_bounds_used is None:
            roi_bounds_used = cc._compute_roi_bounds(frame_bgr, bbox, cc_config)

        print(f"  jendela ROI dipakai: {roi_source}"
              f"  bounds={roi_bounds_used}")

        if roi_bounds_used is not None:
            roi_y1, roi_y2, roi_x1, roi_x2 = roi_bounds_used
            if used_mask and full_mask_for_vis is not None:
                roi_mask_vis = full_mask_for_vis[roi_y1:roi_y2, roi_x1:roi_x2]
                region = overlay[roi_y1:roi_y2, roi_x1:roi_x2]
                highlight = np.zeros_like(region)
                highlight[:, :] = (255, 0, 255)  # magenta, BGR -- piksel yang BENAR dipakai
                mask_bool = roi_mask_vis.astype(bool)
                region[mask_bool] = cv2.addWeighted(region, 0.4, highlight, 0.6, 0)[mask_bool]
                overlay[roi_y1:roi_y2, roi_x1:roi_x2] = region
            # Kotak cyan = jendela ROI yang sungguhan dipakai (mask atau rasio)
            cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

        # --- overlay visual ---
        cv2.rectangle(overlay, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
        if mask_polygon is not None:
            pts = np.array(mask_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        label = f"HSV({h},{s},{v}) {result.color_name}"
        cv2.putText(overlay, label, (bbox.x1, max(bbox.y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug angka detection -> klasifikasi warna")
    parser.add_argument("--image", type=str, default=None,
                         help="Path ke satu gambar statis. Kalau tidak diisi, pakai webcam.")
    parser.add_argument("--no-mask", action="store_true",
                         help="Paksa abaikan mask_polygon (paksa jalur rectangle), untuk A/B compare manual.")
    args = parser.parse_args()

    settings = load_settings()
    cc_config = settings.vision.clothing_color

    detector = PersonDetector(settings.vision)
    detector.load_model()
    detector.warmup()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Gagal baca gambar: {args.image}", file=sys.stderr)
            sys.exit(1)
        overlay = debug_one_frame(img, frame_id=0, detector=detector,
                                   cc_config=cc_config, force_no_mask=args.no_mask)
        cv2.imshow("debug_color_detection", overlay)
        print("\nTekan tombol apa saja di jendela gambar untuk keluar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Pakai class Camera produksi (sama seperti scripts/check_detection.py)
    # supaya tidak perlu menebak nama field CameraSettings (width/height/
    # device_index) -- Camera sudah tahu cara baca config-nya sendiri.
    frame_id = 0
    print("Webcam aktif -- tekan 'q' di jendela video untuk keluar.")
    with Camera(settings.vision.camera) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                print("Frame tidak terbaca -- kamera mungkin terputus. Berhenti.")
                break

            overlay = debug_one_frame(frame.image, frame_id, detector, cc_config, args.no_mask)
            cv2.imshow("debug_color_detection", overlay)
            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
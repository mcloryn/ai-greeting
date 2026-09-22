"""
scripts/bench_seg_model.py

Skrip sekali-pakai untuk MENGUKUR kombinasi model + imgsz di laptop ini
secara nyata, sebelum memutuskan pindah arsitektur ke segmentasi. Bukan
bagian permanen checkpoint manapun -- boleh dihapus setelah dipakai.

Menguji yolo11n.pt (baseline) dan yolo11n-seg.pt (segmentasi), masing-masing
di tiga ukuran inferensi (imgsz): 640 (default), 480, 416.

Tiga hal yang diukur, bukan cuma "seberapa cepat":

  1. LIVE FPS MURNI      -- kecepatan model.predict() saja, dengan kamera asli.
  2. LIVE FPS + EKSTRAKSI -- ditambah biaya nyata memotong ROI torso dan
                             (untuk model seg) menerapkan mask -- ini FPS
                             yang benar-benar akan terjadi di pipeline.
  3. CAKUPAN MASK DI ROI  -- proksi akurasi: dari seluruh piksel di ROI
                             persegi yang dipakai desain SEKARANG, berapa
                             persen yang menurut mask segmentasi benar-benar
                             piksel orang (bukan latar belakang). Rendah =
                             bukti kuat rectangle ROI banyak memuat noise.
                             Hanya berlaku untuk model seg (baseline tidak
                             punya mask untuk dibandingkan).

Jalankan (perbandingan cepat semua kombinasi):
    python scripts/bench_seg_model.py

Jalankan sekaligus soak test kandidat tertentu selama 5 menit:
    python scripts/bench_seg_model.py --soak-seconds 300 \
        --soak-model yolo11n-seg.pt --soak-imgsz 480
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from config.settings import ConfigError, load_settings
from src.vision.camera import Camera

try:
    import psutil
except ImportError:
    psutil = None  # soak test tetap jalan, hanya tanpa data CPU/suhu

_WARMUP_RUNS = 3
_TIMED_RUNS = 20
_LIVE_TEST_SECONDS = 6
_IMGSZ_VARIANTS = [640, 480, 416]
_MODELS = ["yolo11n.pt", "yolo11n-seg.pt"]
_FPS_THRESHOLD = 15.0

# Sama persis dengan rentang ROI torso di Bagian 11.1 dokumen arsitektur --
# supaya angka cakupan mask ini mengukur ROI yang BENAR-BENAR dipakai
# ClothingColorDetector sekarang, bukan ROI karangan skrip ini sendiri.
_TORSO_HEIGHT_RANGE = (0.22, 0.52)
_TORSO_WIDTH_RANGE = (0.28, 0.72)


@dataclass
class ConfigResult:
    model_name: str
    imgsz: int
    dummy_ms: float
    live_fps_inference_only: float
    live_fps_with_extraction: float
    extraction_ms: float
    avg_mask_coverage: float | None  # None untuk model tanpa segmentasi
    detection_rate: float  # proporsi frame live test yang menemukan orang
    frames_tested: int = 0


def _bench_dummy(model, imgsz: int) -> float:
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(_WARMUP_RUNS):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    start = time.perf_counter()
    for _ in range(_TIMED_RUNS):
        model.predict(dummy, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    return (elapsed / _TIMED_RUNS) * 1000  # ms/frame


def _torso_rect(bbox: np.ndarray, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Rect (y1, y2, x1, x2) -- ROI torso yang sama persis dengan yang
    dipakai ClothingColorDetector di pipeline asli (Bagian 11.1)."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    ty1 = int(y1 + _TORSO_HEIGHT_RANGE[0] * bh)
    ty2 = int(y1 + _TORSO_HEIGHT_RANGE[1] * bh)
    tx1 = int(x1 + _TORSO_WIDTH_RANGE[0] * bw)
    tx2 = int(x1 + _TORSO_WIDTH_RANGE[1] * bw)
    return max(0, ty1), min(h, ty2), max(0, tx1), min(w, tx2)


def _largest_detection_index(boxes_xyxy: np.ndarray) -> int | None:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return int(np.argmax(areas))


def _polygon_mask(shape_hw: tuple[int, int], polygon_xy: np.ndarray | None) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if polygon_xy is None or len(polygon_xy) == 0:
        return mask.astype(bool)
    pts = polygon_xy.astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _extract_roi_stats(
    result, frame_image: np.ndarray, is_seg: bool
) -> tuple[float, float | None, bool]:
    """Simulasikan langkah yang akan dilakukan ClothingColorDetector:
    ambil bbox terbesar -> potong ROI torso -> (kalau seg) terapkan mask.

    Return: (waktu_ekstraksi_ms, cakupan_mask_atau_None, ada_deteksi)
    """
    t0 = time.perf_counter()

    boxes_xyxy = (
        result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else None
    )
    idx = _largest_detection_index(boxes_xyxy)
    if idx is None:
        return (time.perf_counter() - t0) * 1000, None, False

    bbox = boxes_xyxy[idx]
    ty1, ty2, tx1, tx2 = _torso_rect(bbox, frame_image.shape)
    rect_area = max(1, (ty2 - ty1) * (tx2 - tx1))

    coverage = None
    if is_seg and result.masks is not None and idx < len(result.masks.xy):
        polygon = result.masks.xy[idx]
        mask = _polygon_mask(frame_image.shape[:2], polygon)
        rect_mask = mask[ty1:ty2, tx1:tx2]
        coverage = float(rect_mask.sum()) / rect_area if rect_mask.size else None

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, coverage, True


def _bench_live(model, imgsz: int, settings, is_seg: bool) -> ConfigResult:
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, verbose=False)  # warmup

    frame_count = 0
    detections = 0
    extraction_times: list[float] = []
    coverages: list[float] = []
    inference_seconds = 0.0
    extraction_seconds = 0.0

    with Camera(settings.vision.camera) as cam:
        start = time.time()
        while time.time() - start < _LIVE_TEST_SECONDS:
            frame = cam.read()
            if frame is None:
                break

            t_inf_start = time.perf_counter()
            results = model.predict(
                frame.image, imgsz=imgsz, conf=0.5, classes=[0], verbose=False
            )
            inference_seconds += time.perf_counter() - t_inf_start

            extraction_ms, coverage, found = _extract_roi_stats(
                results[0], frame.image, is_seg
            )
            extraction_seconds += extraction_ms / 1000
            extraction_times.append(extraction_ms)
            if found:
                detections += 1
            if coverage is not None:
                coverages.append(coverage)

            frame_count += 1
        elapsed = time.time() - start

    live_fps_inference_only = frame_count / inference_seconds if inference_seconds > 0 else 0.0
    total_seconds = inference_seconds + extraction_seconds
    live_fps_with_extraction = frame_count / total_seconds if total_seconds > 0 else 0.0

    return ConfigResult(
        model_name="",  # diisi caller
        imgsz=imgsz,
        dummy_ms=0.0,  # diisi caller
        live_fps_inference_only=live_fps_inference_only,
        live_fps_with_extraction=live_fps_with_extraction,
        extraction_ms=float(np.mean(extraction_times)) if extraction_times else 0.0,
        avg_mask_coverage=float(np.mean(coverages)) if coverages else None,
        detection_rate=detections / frame_count if frame_count else 0.0,
        frames_tested=frame_count,
    )


def _run_quick_comparison(settings) -> list[ConfigResult]:
    from ultralytics import YOLO

    print("=" * 88)
    print("BENCHMARK CEPAT: model x imgsz")
    print(f"(tiap kombinasi live: {_LIVE_TEST_SECONDS} detik -- BERDIRI di depan kamera)")
    print("=" * 88)

    results: list[ConfigResult] = []

    for model_name in _MODELS:
        is_seg = "seg" in model_name
        print(f"\nMemuat '{model_name}'...")
        model = YOLO(model_name)

        for imgsz in _IMGSZ_VARIANTS:
            print(f"\n  >> {model_name}  imgsz={imgsz}")
            dummy_ms = _bench_dummy(model, imgsz)
            print(f"     dummy: {dummy_ms:.1f} ms/frame (~{1000/dummy_ms:.1f} FPS teoretis)")

            print(f"     live kamera {_LIVE_TEST_SECONDS}s ...", end=" ", flush=True)
            res = _bench_live(model, imgsz, settings, is_seg)
            res.model_name = model_name
            res.dummy_ms = dummy_ms

            marker = "OK" if res.live_fps_with_extraction >= _FPS_THRESHOLD else "kurang"
            cov_str = (
                f"{res.avg_mask_coverage * 100:.0f}%" if res.avg_mask_coverage is not None else "N/A"
            )
            print(
                f"FPS(inference)={res.live_fps_inference_only:.1f}  "
                f"FPS(+ekstraksi)={res.live_fps_with_extraction:.1f}  "
                f"cakupan_ROI={cov_str}  deteksi={res.detection_rate*100:.0f}%  [{marker}]"
            )

            if res.detection_rate < 0.5:
                print(
                    "     [PERINGATAN] Kurang dari separuh frame menemukan orang -- "
                    "pastikan Anda berdiri di depan kamera saat pengujian ini berjalan."
                )

            results.append(res)

    return results


def _print_summary(results: list[ConfigResult]) -> None:
    print("\n" + "=" * 88)
    print("RINGKASAN (diurutkan dari FPS+ekstraksi tertinggi -- ini FPS realistis pipeline)")
    print("=" * 88)
    header = (
        f"{'Model':<18}{'imgsz':<7}{'dummy ms':<10}{'FPS infer':<11}"
        f"{'FPS+ekstr':<11}{'ekstr ms':<10}{'cakupan ROI':<13}{'deteksi':<9}{'Status'}"
    )
    print(header)
    print("-" * 88)
    for r in sorted(results, key=lambda r: -r.live_fps_with_extraction):
        status = "OK" if r.live_fps_with_extraction >= _FPS_THRESHOLD else "kurang"
        cov_str = f"{r.avg_mask_coverage*100:.0f}%" if r.avg_mask_coverage is not None else "N/A"
        print(
            f"{r.model_name:<18}{r.imgsz:<7}{r.dummy_ms:<10.1f}{r.live_fps_inference_only:<11.1f}"
            f"{r.live_fps_with_extraction:<11.1f}{r.extraction_ms:<10.2f}{cov_str:<13}"
            f"{r.detection_rate*100:<8.0f}%{status}"
        )
    print("=" * 88)


def _print_worth_it_verdict(results: list[ConfigResult]) -> None:
    print("\n" + "=" * 88)
    print("INDIKATOR WORTH-IT (bukan keputusan otomatis -- baca angkanya sendiri)")
    print("=" * 88)

    baselines = [r for r in results if "seg" not in r.model_name and r.live_fps_with_extraction >= _FPS_THRESHOLD]
    seg_candidates = [r for r in results if "seg" in r.model_name and r.live_fps_with_extraction >= _FPS_THRESHOLD]

    if not seg_candidates:
        print(
            "Tidak ada kombinasi model segmentasi yang mencapai ambang "
            f"{_FPS_THRESHOLD:.0f} FPS (sudah termasuk biaya ekstraksi ROI) di laptop ini.\n"
            "-> Secara FPS saja, migrasi ke segmentasi belum layak di hardware ini."
        )
        return

    best_seg = max(seg_candidates, key=lambda r: r.live_fps_with_extraction)
    print(
        f"Kandidat seg terbaik yang lolos ambang: {best_seg.model_name} @ imgsz={best_seg.imgsz} "
        f"-> {best_seg.live_fps_with_extraction:.1f} FPS (dengan ekstraksi ROI nyata)."
    )

    if baselines:
        best_baseline = max(baselines, key=lambda r: r.live_fps_with_extraction)
        drop_pct = (
            (best_baseline.live_fps_with_extraction - best_seg.live_fps_with_extraction)
            / best_baseline.live_fps_with_extraction
            * 100
        )
        print(
            f"Dibanding baseline terbaik ({best_baseline.model_name} @ imgsz={best_baseline.imgsz}, "
            f"{best_baseline.live_fps_with_extraction:.1f} FPS): turun {drop_pct:.0f}%."
        )

    if best_seg.avg_mask_coverage is None:
        print("Tidak ada data cakupan mask (tidak ada orang terdeteksi cukup lama saat uji).")
    else:
        cov_pct = best_seg.avg_mask_coverage * 100
        print(f"Rata-rata cakupan ROI torso oleh mask nyata: {cov_pct:.0f}%.")
        if cov_pct >= 90:
            print(
                "-> Cakupan tinggi: ROI persegi yang dipakai sekarang sudah cukup bersih "
                "di kondisi pencahayaan/pose saat uji ini. Manfaat akurasi dari migrasi ke "
                "segmentasi kemungkinan kecil -- pertimbangkan apakah biaya FPS di atas sepadan."
            )
        elif cov_pct >= 70:
            print(
                "-> Cakupan sedang: sebagian ROI persegi memang bukan piksel torso. "
                "Migrasi ke segmentasi kemungkinan membantu, tapi coba juga alternatif lebih "
                "murah (K-Means di dalam ROI yang ada) sebelum memutuskan."
            )
        else:
            print(
                "-> Cakupan rendah: sebagian besar ROI persegi BUKAN piksel torso -- ini bukti "
                "kuat kenapa median warna sering kebobolan latar belakang. Migrasi ke segmentasi "
                "kemungkinan besar memberi manfaat akurasi nyata, sepadan dengan biaya FPS di atas."
            )

    print(
        "\nCatatan: cakupan mask ini proksi, bukan ukuran akurasi warna akhir -- tetap perlu "
        "dibandingkan hasil klasifikasi warna aktual (rectangle vs mask) di beberapa baju uji."
    )


def _run_soak_test(model_name: str, imgsz: int, settings, seconds: int) -> None:
    from ultralytics import YOLO

    is_seg = "seg" in model_name
    print("\n" + "=" * 88)
    print(f"SOAK TEST: {model_name} @ imgsz={imgsz}, durasi {seconds} detik")
    print("Berdiri di depan kamera selama pengujian ini berjalan.")
    if psutil is None:
        print("(psutil tidak terpasang -- data CPU/suhu tidak akan ditampilkan)")
    print("=" * 88)

    model = YOLO(model_name)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, verbose=False)  # warmup

    window_seconds = 30
    window_frames = 0
    window_start = time.time()

    with Camera(settings.vision.camera) as cam:
        test_start = time.time()
        while time.time() - test_start < seconds:
            frame = cam.read()
            if frame is None:
                break
            results = model.predict(
                frame.image, imgsz=imgsz, conf=0.5, classes=[0], verbose=False
            )
            _extract_roi_stats(results[0], frame.image, is_seg)
            window_frames += 1

            now = time.time()
            if now - window_start >= window_seconds:
                fps = window_frames / (now - window_start)
                elapsed_total = now - test_start
                cpu_info = ""
                if psutil is not None:
                    cpu_pct = psutil.cpu_percent(interval=None)
                    cpu_info = f"  CPU={cpu_pct:.0f}%"
                    temps = getattr(psutil, "sensors_temperatures", lambda: {})()
                    if temps:
                        first_sensor = next(iter(temps.values()))
                        if first_sensor:
                            cpu_info += f"  suhu={first_sensor[0].current:.0f}C"
                print(f"  t={elapsed_total:6.0f}s  FPS jendela 30s = {fps:5.1f}{cpu_info}")
                window_frames = 0
                window_start = now

    print("Soak test selesai. Bandingkan FPS jendela awal vs akhir -- penurunan tajam")
    print("mengindikasikan thermal throttling, bukan cuma variasi normal.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Lewati soak test, hanya jalankan perbandingan cepat."
    )
    parser.add_argument(
        "--soak-seconds",
        type=int,
        default=0,
        help="Durasi soak test dalam detik (0 = tidak dijalankan). Contoh: 300 untuk 5 menit.",
    )
    parser.add_argument(
        "--soak-model",
        type=str,
        default=None,
        help="Model untuk soak test. Default: kandidat seg terbaik dari perbandingan cepat.",
    )
    parser.add_argument(
        "--soak-imgsz",
        type=int,
        default=None,
        help="imgsz untuk soak test. Default: imgsz dari kandidat seg terbaik.",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[KONFIGURASI SALAH] {exc}", file=sys.stderr)
        return 1

    results = _run_quick_comparison(settings)
    _print_summary(results)
    _print_worth_it_verdict(results)

    if args.quick or (args.soak_seconds <= 0 and not args.soak_model):
        return 0

    soak_model = args.soak_model
    soak_imgsz = args.soak_imgsz
    if soak_model is None or soak_imgsz is None:
        seg_candidates = [r for r in results if "seg" in r.model_name and r.live_fps_with_extraction >= _FPS_THRESHOLD]
        if not seg_candidates:
            print(
                "\nTidak ada kandidat seg yang lolos ambang FPS -- soak test dilewati. "
                "Jalankan ulang dengan --soak-model/--soak-imgsz untuk memaksa kombinasi tertentu."
            )
            return 0
        best = max(seg_candidates, key=lambda r: r.live_fps_with_extraction)
        soak_model = soak_model or best.model_name
        soak_imgsz = soak_imgsz or best.imgsz

    seconds = args.soak_seconds if args.soak_seconds > 0 else 300
    _run_soak_test(soak_model, soak_imgsz, settings, seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
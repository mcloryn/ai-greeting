"""
src/vision/detector.py

Membungkus model YOLO (Ultralytics) menjadi satu tanggung jawab tunggal:
mengubah satu Frame menjadi daftar Detection orang. TIDAK ADA logika
tracking, target selection, atau apa pun lain di sini (Aturan #6 Build
Plan -- vision/ tidak boleh tahu soal StateMachine atau lapisan lain).

Prinsip pengikat (Bagian 1.6 Build Plan): "YOLO adalah sensor, bukan otak."
Module ini hanya melapor apa yang terlihat -- tidak memutuskan apa pun.

Kontrak (Bagian 4.2 Build Plan):
    Input  : Frame
    Output : list[Detection]
    Public : load_model(), detect(frame) -> list[Detection], warmup()
"""

from __future__ import annotations

from config.settings import VisionSettings
from src.core.exceptions import DetectorError
from src.core.logger import get_logger
from src.core.models import BoundingBox, Detection, Frame

logger = get_logger(__name__)


class PersonDetector:
    """Deteksi orang di satu frame memakai YOLO pretrained.

    Dipakai seperti ini:

        detector = PersonDetector(settings.vision)
        detector.load_model()
        detector.warmup()
        detections = detector.detect(frame)
    """

    def __init__(self, config: VisionSettings) -> None:
        self._config = config
        self._model = None  # diisi saat load_model()

    def load_model(self) -> None:
        """Muat file bobot YOLO. Melempar DetectorError kalau gagal."""
        try:
            # Import ditunda ke sini (bukan di top-level file) supaya
            # module ini tetap bisa di-import untuk unit test tanpa
            # memaksa ultralytics terinstal di setiap environment test.
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "Package 'ultralytics' belum terinstal. Jalankan: "
                "pip install ultralytics"
            ) from exc

        try:
            self._model = YOLO(self._config.model_name)
        except Exception as exc:
            raise DetectorError(
                f"Gagal memuat model YOLO '{self._config.model_name}'. "
                "Periksa koneksi internet (untuk unduhan pertama) atau "
                "nama file model di config.yaml."
            ) from exc

        logger.info("Model YOLO '%s' berhasil dimuat.", self._config.model_name)

    def warmup(self) -> None:
        """Jalankan satu inferensi dummy supaya inferensi PERTAMA yang
        sungguhan (saat kamera sudah hidup) tidak lambat.

        YOLO melakukan inisialisasi internal (alokasi memori, kompilasi
        graph) pada panggilan pertama -- kalau itu terjadi saat orang
        sungguhan sedang berdiri di depan kamera, sapaan jadi telat.
        """
        if self._model is None:
            raise DetectorError("warmup() dipanggil sebelum load_model().")

        import numpy as np

        dummy_image = np.zeros((480, 640, 3), dtype="uint8")
        self._model.predict(dummy_image, verbose=False)
        logger.info("Warmup selesai -- model siap untuk inferensi real-time.")

    def detect(self, frame: Frame) -> list[Detection]:
        """Deteksi orang di satu frame. Mengembalikan list kosong kalau
        tidak ada orang terdeteksi -- BUKAN None, supaya pemanggil tidak
        perlu cek None di setiap tempat yang memakai hasil ini.

        Method ini dipakai bila hanya butuh deteksi tanpa identitas
        antar-frame (mis. untuk unit test murni CP03). Untuk identitas
        yang bertahan antar-frame, pakai detect_and_track().
        """
        if self._model is None:
            raise DetectorError("detect() dipanggil sebelum load_model().")

        results = self._model.predict(
            frame.image,
            conf=self._config.confidence_threshold,
            classes=[self._config.person_class_id],
            verbose=False,
        )

        return _parse_yolo_results(results, frame.frame_id)

    def detect_and_track(self, frame: Frame) -> list[Detection]:
        """Sama seperti detect(), tapi mengaktifkan tracker bawaan
        Ultralytics (persist=True) sehingga setiap Detection membawa
        track_id yang konsisten antar-frame -- dipakai oleh Tracker.

        `persist=True` WAJIB: tanpa ini, tracker Ultralytics membuat
        instance baru setiap panggilan dan track_id akan reset setiap
        frame (Failure Case yang eksplisit disebut Build Plan Bagian 9.3).
        """
        if self._model is None:
            raise DetectorError("detect_and_track() dipanggil sebelum load_model().")

        results = self._model.track(
            frame.image,
            conf=self._config.confidence_threshold,
            classes=[self._config.person_class_id],
            persist=True,
            verbose=False,
        )

        return _parse_yolo_results(results, frame.frame_id)

def _parse_yolo_results(results: list, frame_id: int) -> list[Detection]:
    """Ubah objek hasil mentah Ultralytics menjadi list[Detection] milik kita.

    Dipisah jadi fungsi murni (bukan method) supaya bisa diuji dengan data
    hasil YOLO buatan (fake), tanpa perlu model sungguhan ter-load.
    """
    detections: list[Detection] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        # result.masks hanya ada kalau model yang dimuat adalah varian
        # segmentasi (mis. yolo11n-seg.pt). Untuk model deteksi biasa
        # (yolo11n.pt), ini None -- dan mask_polygon setiap Detection
        # otomatis tetap None, sama seperti sebelum perubahan ini.
        # masks.xy sejajar indeksnya dengan boxes karena keduanya berasal
        # dari result yang sama (sudah difilter conf/class yang sama).
        masks_xy = None
        if getattr(result, "masks", None) is not None:
            masks_xy = result.masks.xy

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            # box.id hanya ada saat memakai .track() (bukan .predict());
            # None berarti belum ada track_id (dipakai jalur CP03 lama).
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None

            bbox = BoundingBox(
                x1=int(xyxy[0]), y1=int(xyxy[1]),
                x2=int(xyxy[2]), y2=int(xyxy[3]),
            )

            mask_polygon = None
            if masks_xy is not None and i < len(masks_xy):
                # masks_xy[i]: array Nx2 float, koordinat piksel absolut
                # di frame asli (bukan relatif ke bbox).
                mask_polygon = tuple(
                    (int(x), int(y)) for x, y in masks_xy[i]
                )

            detections.append(Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                frame_id=frame_id,
                track_id=track_id,
                mask_polygon=mask_polygon,
            ))

    return detections
"""
tests/unit/test_detector.py

Unit test CP03. TIDAK memuat model YOLO sungguhan -- hanya menguji fungsi
murni _parse_yolo_results() memakai objek hasil YOLO tiruan (fake).

Kenapa fake, bukan model asli? Supaya test tetap cepat (<1 detik) dan
tidak butuh unduhan/GPU/koneksi internet setiap kali `pytest` dijalankan --
sama seperti test_camera_config.py yang tidak butuh webcam.
"""

from __future__ import annotations

from src.vision.detector import _parse_yolo_results


class _FakeTensor:
    """Tiruan minimal dari tensor PyTorch -- cukup punya .tolist() atau
    bisa di-index dan di-float()-kan, sesuai yang dipakai kode kita.
    """

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def __getitem__(self, idx):
        return self._values[idx]

    def __float__(self):
        return float(self._values[0])

    def __int__(self):
        return int(self._values[0])


class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [_FakeTensor(xyxy)]
        self.conf = [conf]
        self.cls = [cls]


class _FakeBoxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = _FakeBoxes(boxes) if boxes is not None else None


def test_parse_yolo_results_kosong():
    results = [_FakeResult(boxes=[])]
    detections = _parse_yolo_results(results, frame_id=1)
    assert detections == []


def test_parse_yolo_results_boxes_none_tidak_crash():
    # Beberapa versi Ultralytics bisa mengembalikan boxes=None kalau
    # tidak ada deteksi sama sekali -- ini kondisi tepi yang wajib aman.
    results = [_FakeResult(boxes=None)]
    detections = _parse_yolo_results(results, frame_id=1)
    assert detections == []


def test_parse_yolo_results_satu_deteksi():
    fake_box = _FakeBox(xyxy=[100.0, 50.0, 200.0, 250.0], conf=0.87, cls=0)
    results = [_FakeResult(boxes=[fake_box])]

    detections = _parse_yolo_results(results, frame_id=42)

    assert len(detections) == 1
    det = detections[0]
    assert det.bbox.x1 == 100
    assert det.bbox.y1 == 50
    assert det.bbox.x2 == 200
    assert det.bbox.y2 == 250
    assert det.confidence == 0.87
    assert det.class_id == 0
    assert det.frame_id == 42


def test_parse_yolo_results_banyak_deteksi():
    box_a = _FakeBox(xyxy=[0.0, 0.0, 50.0, 100.0], conf=0.9, cls=0)
    box_b = _FakeBox(xyxy=[60.0, 0.0, 110.0, 100.0], conf=0.75, cls=0)
    results = [_FakeResult(boxes=[box_a, box_b])]

    detections = _parse_yolo_results(results, frame_id=5)

    assert len(detections) == 2
    assert detections[0].confidence == 0.9
    assert detections[1].confidence == 0.75


def test_parse_yolo_results_koordinat_float_dibulatkan_ke_int():
    # YOLO mengembalikan koordinat float (mis. 100.6) -- BoundingBox kita
    # bertipe int, jadi harus dipotong (truncated), bukan crash.
    fake_box = _FakeBox(xyxy=[100.6, 50.2, 200.9, 250.4], conf=0.5, cls=0)
    results = [_FakeResult(boxes=[fake_box])]

    detections = _parse_yolo_results(results, frame_id=1)

    assert detections[0].bbox.x1 == 100
    assert detections[0].bbox.y2 == 250
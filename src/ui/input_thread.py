"""
InputThread membaca ketikan user di thread terpisah.

Kenapa perlu thread? Fungsi input() Python itu "blocking" -- artinya
program berhenti total sampai user menekan Enter. Kalau ini dipanggil
langsung di loop utama (tempat video diproses setiap frame), video akan
membeku selama user berpikir mau mengetik apa. Dengan menaruhnya di
thread terpisah, loop utama bebas terus memproses video sementara
thread ini menunggu ketikan di "latar belakang".
"""

import threading
import queue

from src.core.logger import get_logger

logger = get_logger(__name__)


class InputThread:
    """Membaca input user tanpa memblokir thread utama."""

    def __init__(self):
        # Queue adalah "kotak surat" aman antar-thread: thread ini
        # menaruh pesan masuk, loop utama mengambilnya kapan pun siap.
        # Aman dipakai dua thread sekaligus tanpa perlu lock manual.
        self._queue: queue.Queue[str] = queue.Queue()

        # Event ini seperti saklar. Selama False, thread terus berjalan.
        # Saat di-set True, thread tahu ia harus berhenti.
        self._stop_event = threading.Event()

        # daemon=True: kalau program utama mati mendadak, thread ini
        # ikut mati otomatis, tidak menggantung.
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Mulai thread pembaca input."""
        self._thread.start()
        logger.info("InputThread dimulai")

    def _run(self) -> None:
        """Loop internal thread: terus menunggu ketikan user."""
        while not self._stop_event.is_set():
            try:
                # input() memblokir DI SINI, tapi karena ini thread
                # terpisah, video di thread utama tidak terpengaruh.
                text = input()
                if text.strip():
                    self._queue.put(text.strip())
            except EOFError:
                # Terjadi kalau stdin ditutup (misalnya saat testing
                # otomatis). Jangan crash, cukup berhenti.
                break

    def get_nowait(self) -> str | None:
        """
        Dipanggil dari loop utama setiap iterasi.
        Mengembalikan teks jika ada, atau None jika belum ada ketikan baru.
        Tidak pernah menunggu (non-blocking) -- inilah kuncinya.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Minta thread berhenti. Thread akan berhenti di ketikan berikutnya."""
        self._stop_event.set()
        logger.info("InputThread diminta berhenti")
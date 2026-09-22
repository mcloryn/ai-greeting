"""
Tampilan console sederhana untuk chat.
Hanya urusan cetak-mencetak ke layar -- tidak menyimpan state apa pun.
"""


def print_greeting(text: str) -> None:
    print(f"\n[SISTEM] {text}")


def print_prompt() -> None:
    print("Anda: ", end="", flush=True)


def print_assistant_reply(text: str) -> None:
    print(f"[SISTEM] {text}")
    print_prompt()


def print_session_ended(reason: str) -> None:
    print(f"\n[SISTEM] Sesi berakhir ({reason}). Menunggu orang berikutnya...\n")
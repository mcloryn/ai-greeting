from src.application.conversation_manager import ConversationManager
from src.core.enums import EndReason
from src.application.session import is_farewell


def test_start_session_creates_active_session():
    conv = ConversationManager(max_history_turns=3)
    conv.start_session(track_id=1, clothing_color="biru")
    assert conv.active_session is not None
    assert conv.active_session.track_id == 1
    assert conv.active_session.clothing_color == "biru"


def test_receive_question_adds_to_history():
    conv = ConversationManager(max_history_turns=3)
    conv.start_session(track_id=1, clothing_color="biru")
    conv.receive_question("Apa saja jurusan yang ada?")
    assert len(conv.active_session.history) == 1
    assert conv.active_session.history[0]["role"] == "user"


def test_build_dummy_answer_adds_to_history_and_returns_text():
    conv = ConversationManager(max_history_turns=3)
    conv.start_session(track_id=1, clothing_color="biru")
    conv.receive_question("Apa saja jurusan yang ada?")
    answer = conv.build_dummy_answer()
    assert "Apa saja jurusan yang ada?" in answer
    assert conv.active_session.turn_count == 1
    assert len(conv.active_session.history) == 2


def test_archive_session_closes_and_clears_active_session():
    conv = ConversationManager(max_history_turns=3)
    conv.start_session(track_id=1, clothing_color="biru")
    conv.archive_session(EndReason.FAREWELL.value)
    assert conv.active_session is None


def test_archive_session_with_unknown_reason_falls_back_to_manual(caplog):
    conv = ConversationManager(max_history_turns=3)
    conv.start_session(track_id=1, clothing_color="biru")
    conv.archive_session("BUKAN_ALASAN_VALID")
    assert conv.active_session is None


def test_history_is_trimmed_to_max_turns():
    conv = ConversationManager(max_history_turns=2)
    conv.start_session(track_id=1, clothing_color="biru")
    for i in range(5):
        conv.receive_question(f"pertanyaan {i}")
        conv.build_dummy_answer()
    assert len(conv.active_session.history) <= 4

def test_is_farewell_terdeteksi():
    assert is_farewell("terima kasih ya", ["terima kasih", "bye"]) is True


def test_is_farewell_tidak_terdeteksi():
    assert is_farewell("berapa biaya kuliah?", ["terima kasih", "bye"]) is False
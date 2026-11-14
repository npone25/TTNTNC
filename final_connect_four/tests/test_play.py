"""Test phần logic của demo (không mở cửa sổ). Phụ trách: Nguyễn Trọng Phúc."""

from play import GameController
from utils.arena import random_agent
from conftest import set_seed


def test_nguoi_dau_may_dung_luot():
    set_seed(0)
    game = GameController(("Người chơi", None), ("Máy", random_agent))
    assert game.is_human_turn()
    # máy chưa tới lượt nên chưa có gì thay đổi; người đi cột 4
    row, col, color = game.human_move(4)
    assert (row, col, color) == (0, 4, "red")
    assert not game.is_human_turn()
    # người bấm thêm khi chưa tới lượt thì bị bỏ qua
    assert game.human_move(3) is None
    row, col, color = game.ai_move()
    assert color == "yellow"
    assert game.is_human_turn()


def test_nguoi_bam_cot_day_hoac_ngoai_ban():
    game = GameController(("Người 1", None), ("Người 2", None))
    for _ in range(6):
        game.human_move(1)
    assert 1 not in game.env.validinputs
    assert game.human_move(1) is None
    assert game.human_move(None) is None


def test_may_dau_may_den_het_van_va_thong_bao():
    set_seed(3)
    game = GameController(("Máy A", random_agent), ("Máy B", random_agent))
    while not game.env.done:
        game.ai_move()
    text = game.status_text()
    assert ("thắng" in text) or (text == "Hòa!")

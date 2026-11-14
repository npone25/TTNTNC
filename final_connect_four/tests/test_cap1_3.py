"""Test cấp 1, 2 (rules.py) và cấp 3 (alphabeta.py). Phụ trách: Phạm Thiên Phúc."""

import pytest

from utils.conn_simple_env import conn
from conftest import (check_takes_win, check_blocks, check_never_illegal,
                      check_reproducible, set_seed)


# Cấp 1 và cấp 2: luật (rules.py)

from utils import rules


def test_cap1_khong_di_nuoc_sai_va_tai_lap():
    check_never_illegal(rules.conn_random)
    check_reproducible(rules.conn_random)


def test_cap2_thang_ngay_va_chan():
    check_takes_win(rules.conn_think3)
    check_blocks(rules.conn_think3)


def test_cap2_khong_di_nuoc_sai_va_tai_lap():
    check_never_illegal(rules.conn_think3)
    check_reproducible(rules.conn_think3)


def test_cap2_thang_ngay_truoc_khi_chiem_cot_giua():
    # cột 4 còn trống nhưng đỏ thắng dọc ở cột 1: luật thắng phải đứng trước luật cột giữa
    env = conn.from_moves([1, 2, 1, 2, 1, 3])
    assert rules.conn_think3(env) == 1
    assert rules.LAST_INFO["rule"] == "win"


@pytest.mark.parametrize("moves, trap", [
    # vàng sắp đi: đỏ có 3 quân ở hàng 2 (cột 1, 3, 4), thiếu ô cột 2 hàng 2.
    # Vàng thả cột 2 thì đỏ thả lên ngay trên và thắng. Bản của sách bỏ sót trường hợp này.
    ([1, 3, 1, 1, 3, 4, 4], 2),
    # đỏ sắp đi: thả cột 3 thì vàng thắng bằng ô ngay phía trên
    ([2, 4, 1, 1, 1, 1, 2, 2], 3),
])
def test_to_avoid_dung_cho_ca_hai_mau(moves, trap):
    env = conn.from_moves(moves)
    assert trap in rules.to_avoid(env)
    # tác tử cấp 2 không được đi vào cột bệ đỡ
    for seed in range(3):
        set_seed(seed)
        assert rules.conn_think3(env) != trap


# Cấp 3: Minimax Alpha-Beta (alphabeta.py)

from utils import alphabeta


def cap3_agent(env):
    return alphabeta.MiniMax_conn(env, depth=3)


def test_cap3_thang_ngay_va_chan():
    check_takes_win(cap3_agent)
    check_blocks(cap3_agent)


def test_cap3_khong_di_nuoc_sai_va_tai_lap():
    check_never_illegal(cap3_agent)
    check_reproducible(cap3_agent)


def test_cap3_cat_tia_khong_doi_ket_qua_nhung_duyet_it_nut_hon():
    env = conn.from_moves([4, 4, 3, 5])
    scores = {}
    nodes = {}
    for prune in (True, False):
        search = alphabeta.Search(prune=prune)
        scores[prune] = search.root_scores(env, 3)
        nodes[prune] = search.nodes
    assert scores[True] == scores[False]
    assert nodes[True] < nodes[False]


def test_cap3_ham_luong_gia_doi_xung():
    # cùng một bàn, đổi người sắp đi thì điểm đổi dấu
    env = conn.from_moves([4, 3, 4])
    red_view = env.clone()
    red_view.turn = "red"
    yellow_view = env.clone()
    yellow_view.turn = "yellow"
    assert alphabeta.evaluate(red_view) == pytest.approx(-alphabeta.evaluate(yellow_view))
    assert -0.9 <= alphabeta.evaluate(red_view) <= 0.9


def test_cap3_che_do_thoi_gian():
    alphabeta.MiniMax_conn(conn.from_moves([4, 4]), time_limit=0.2)
    assert alphabeta.LAST_INFO["depth"] >= 2
    assert alphabeta.LAST_INFO["seconds"] < 0.5

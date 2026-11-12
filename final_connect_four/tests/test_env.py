"""Test engine: luật, thắng ngang/dọc/chéo, hòa, nước sai, undo, clone, perft. Phụ trách: Nguyễn Trọng Phúc."""

import random
import numpy as np
import pytest

from utils.conn_simple_env import conn, InvalidMoveError
from conftest import slow_winner, count_windows

# Ván thật lấy từ chơi ngẫu nhiên có seed: kết thúc ở nước 42 bằng một nước THẮNG
WIN_ON_MOVE_42 = [5, 6, 7, 5, 5, 5, 4, 5, 2, 3, 4, 3, 6, 4, 1, 7, 2, 2, 5, 7, 1, 3, 3, 4,
                  1, 6, 7, 1, 4, 2, 2, 1, 1, 6, 4, 7, 2, 3, 7, 3, 6, 6]
# Ván đầy bàn không ai thắng
DRAW_GAME = [5, 4, 7, 1, 2, 5, 6, 6, 2, 2, 6, 1, 2, 7, 1, 2, 6, 6, 2, 1, 5, 7, 4, 3,
             7, 7, 1, 5, 7, 6, 3, 1, 5, 3, 5, 3, 3, 3, 4, 4, 4, 4]


def test_khoi_tao():
    env = conn()
    assert env.turn == "red"
    assert env.legal_actions() == [1, 2, 3, 4, 5, 6, 7]
    assert env.state.shape == (7, 6)
    assert np.all(env.state == 0)
    assert env.is_terminal() is False
    assert env.result() is None


def test_trong_luc_va_doi_luot():
    env = conn()
    env.step(4)
    env.step(4)
    # quân đỏ ở đáy cột 4, quân vàng ngay trên
    assert env.state[3][0] == 1
    assert env.state[3][1] == -1
    assert env.turn == "red"


def test_cot_day_bi_loai():
    env = conn.from_moves([1, 1, 1, 1, 1, 1])
    assert 1 not in env.legal_actions()
    with pytest.raises(InvalidMoveError):
        env.step(1)

# Gắn dữ liệu chạy này cho hàm ngay bên dưới 
@pytest.mark.parametrize("moves, winner", [ 
    ([1, 7, 2, 7, 3, 6, 4], 1),          # đỏ ngang
    ([1, 2, 1, 2, 1, 3, 1], 1),          # đỏ dọc
    ([1, 2, 2, 3, 3, 4, 3, 4, 4, 7, 4], 1),   # đỏ chéo /
    ([1, 4, 1, 5, 2, 6, 7, 3], -1),      # vàng ngang
    ([1, 2, 3, 2, 5, 2, 7, 2], -1),      # vàng dọc
    ([7, 6, 6, 5, 5, 4, 5, 4, 4, 1, 1, 3], -1),   # vàng chéo \
])
def test_thang_bon_huong(moves, winner):
    env = conn.from_moves(moves)
    assert env.is_terminal()
    assert env.result() == winner
    assert slow_winner(env.state) == winner
    # ván đã kết thúc thì không còn nước hợp lệ, lượt vẫn là người vừa đi
    assert env.legal_actions() == []
    expected_turn = "red" if winner == 1 else "yellow"
    assert env.turn == expected_turn


def test_thang_o_canh_bien():
    # đỏ thắng hàng dọc ở cột 7, từ hàng 3 lên tới hàng trên cùng
    env = conn.from_moves([1, 7, 2, 7, 7, 1, 7, 2, 7, 1, 7])
    assert env.is_terminal()
    assert env.result() == 1
    assert env.state[6][5] == 1
    assert slow_winner(env.state) == 1


def test_hoa():
    env = conn.from_moves(DRAW_GAME)
    assert env.is_terminal()
    assert env.result() == 0
    assert env.winner() == "draw"
    assert slow_winner(env.state) == 0


def test_thang_o_nuoc_42_van_la_thang():
    env = conn.from_moves(WIN_ON_MOVE_42)
    assert len(env.history) == 42
    assert env.is_terminal()
    assert env.result() != 0
    assert slow_winner(env.state) == env.result()


@pytest.mark.parametrize("bad", [0, 8, -1, 2.5, "3", None, True])
def test_nuoc_sai_bi_chan_va_khong_doi_trang_thai(bad):
    env = conn.from_moves([4, 4, 3])
    state_before = env.state.copy()
    history_before = list(env.history)
    with pytest.raises(InvalidMoveError):
        env.step(bad)
    assert np.array_equal(env.state, state_before)
    assert env.history == history_before
    assert env.turn == "yellow"


def test_khong_di_duoc_sau_khi_ket_thuc():
    env = conn.from_moves([1, 7, 2, 7, 3, 6, 4])
    with pytest.raises(InvalidMoveError):
        env.step(5)


def test_chap_nhan_so_nguyen_numpy():
    env = conn()
    env.step(np.int64(4))
    assert env.state[3][0] == 1


def test_undo_tung_buoc():
    random.seed(7)
    env = conn()
    snapshots = []
    while not env.done:
        snapshots.append((env.state.copy(), list(env.validinputs), env.turn, env.done, env.reward))
        env.step(random.choice(env.validinputs))
    # hoàn tác ngược lại đến đầu ván, so từng bước
    while len(env.history) > 0:
        env.undo()
        state, valid, turn, done, reward = snapshots.pop()
        assert np.array_equal(env.state, state)
        assert env.validinputs == valid
        assert env.turn == turn
        assert env.done == done
        assert env.reward == reward
    with pytest.raises(InvalidMoveError):
        env.undo()


def test_undo_sau_nuoc_thang():
    env = conn.from_moves([1, 7, 2, 7, 3, 6, 4])
    env.undo()
    assert env.is_terminal() is False
    assert env.result() is None
    assert env.turn == "red"
    assert env.state[3][0] == 0    # quân Đỏ ở cột 4 đã bị gỡ
    assert env.legal_actions() == [1, 2, 3, 4, 5, 6, 7]

def test_clone_doc_lap():
    env = conn.from_moves([4, 4, 3])
    copy_env = env.clone()
    copy_env.step(2)
    assert len(env.history) == 3
    assert env.state[1][0] == 0
    env.step(5)
    assert copy_env.state[4][0] == 0
    assert copy_env.turn == "red"


def test_render():
    env = conn.from_moves([4, 4, 3])
    text = env.render()
    lines = text.split("\n")
    assert lines[-1] == "1 2 3 4 5 6 7"
    assert lines[-2] == ". . X X . . ."
    assert lines[-3] == ". . . O . . ."


def test_so_cua_so_bon_o_la_69():
    assert count_windows() == 69

# tổng kiểu đi != tổng thế cờ -> nhiều kiểu đi có thể ra cùng 1 thế cờ 
# tính số dãy nước đi có thể thực hiện nếu đi trong depth nước, chuẩn khi env là rỗng
def count_sequences(env, depth):
    """Đếm số dãy nước đi hợp lệ dài depth, dùng step và undo."""
    if depth == 0:
        return 1
    total = 0
    for move in env.legal_actions():
        env.step(move)
        total += count_sequences(env, depth - 1)
        env.undo()
    return total


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_perft(depth):
    # trước nước thứ 7 chưa ai thắng được nên số dãy nước đúng bằng 7^depth
    env = conn()
    assert count_sequences(env, depth) == 7 ** depth
    # sau khi duyệt xong, bàn cờ phải trở về trống
    assert len(env.history) == 0
    assert np.all(env.state == 0)


@pytest.mark.slow
def test_perft_do_sau_6():
    assert count_sequences(conn(), 6) == 7 ** 6


def test_doi_chieu_voi_bo_kiem_tra_doc_lap():
    random.seed(2026)
    for _ in range(500):
        env = conn()
        while True:
            env.step(random.choice(env.validinputs))
            expected = slow_winner(env.state)
            if expected != 0:
                # bộ kiểm tra độc lập thấy có người thắng: engine phải dừng đúng người đó
                assert env.is_terminal() and env.result() == expected
                break
            if env.is_terminal():
                # engine dừng mà không ai thắng thì phải là hòa, đủ 42 nước
                assert env.result() == 0 and len(env.history) == 42
                break


def test_van_ngau_nhien_ket_thuc_hop_le():
    random.seed(11)
    for _ in range(1000):
        env = conn()
        while not env.done:
            env.step(random.choice(env.validinputs))
        assert 7 <= len(env.history) <= 42

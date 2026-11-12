"""Hàm hỗ trợ test dùng chung cho cả nhóm. Phụ trách: Nguyễn Trọng Phúc.

Gồm hai phần:
1. slow_winner(): bộ kiểm tra thắng CHẬM, viết độc lập với engine, để đối chiếu.
2. Các hàm kiểm tra tác tử dùng chung. Mỗi người chỉ cần gọi cho tác tử của mình, ví dụ:

    from conftest import check_never_illegal, check_takes_win, check_blocks, check_reproducible

    def test_cap2_thang_ngay():
        check_takes_win(conn_think3)

Tác tử là một hàm nhận env và trả về một cột 1-7 (quy ước của sách).
Tác tử cần thêm tham số thì bọc lại: check_takes_win(lambda env: MiniMax_conn(env, depth=3))
"""

import random
import numpy as np

from utils.conn_simple_env import conn


def slow_winner(state):
    """Quét toàn bộ 69 cửa sổ 4 ô của bàn (7, 6), độc lập với engine.

    Trả về 1 nếu đỏ có 4 quân liên tiếp, -1 nếu vàng có, 0 nếu chưa ai có.
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for col in range(7):
        for row in range(6):
            for d_col, d_row in directions:
                end_col = col + 3 * d_col
                end_row = row + 3 * d_row
                if end_col < 0 or end_col > 6 or end_row < 0 or end_row > 5:
                    continue
                first = state[col][row]
                if first == 0:
                    continue
                same = True
                for k in range(1, 4):
                    if state[col + k * d_col][row + k * d_row] != first:
                        same = False
                if same:
                    return int(first)
    return 0


# Thế cờ mà người sắp đi thắng được ngay bằng đúng một cột.
# Mỗi phần tử: (dãy nước đi tạo thế cờ, cột thắng duy nhất, mô tả)
WIN_IN_ONE = [
    ([1, 7, 2, 7, 3, 6], 4, "đỏ thắng hàng ngang"),
    ([1, 2, 1, 2, 1, 3], 1, "đỏ thắng hàng dọc"),
    ([1, 4, 1, 5, 2, 6, 7], 3, "vàng thắng hàng ngang"),
    ([1, 2, 3, 2, 5, 2, 7], 2, "vàng thắng hàng dọc"),
    ([1, 2, 2, 3, 3, 4, 3, 4, 4, 7], 4, "đỏ thắng chéo /, đối thủ cũng đang dọa"),
    ([7, 6, 6, 5, 5, 4, 5, 4, 4, 1, 1], 3, "vàng thắng chéo \\, đối thủ cũng đang dọa"),
]


# Thế cờ mà người sắp đi không thắng ngay được nhưng phải chặn đúng một cột.
MUST_BLOCK = [
    ([1, 7, 2, 7, 3], 4, "vàng phải chặn hàng ngang của đỏ"),
    ([1, 2, 3, 2, 5, 2], 2, "đỏ phải chặn hàng dọc của vàng"),
]


def set_seed(seed):
    """Đặt seed cho cả random và numpy (sách dùng cả hai)."""
    random.seed(seed)
    np.random.seed(seed)


def snapshot(env):
    """Chụp lại trạng thái env để kiểm tra tác tử không sửa env."""
    return (env.state.copy(), list(env.history), env.turn, env.done, list(env.validinputs))


def same_snapshot(a, b):
    return (np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
            and a[3] == b[3] and a[4] == b[4])


def check_takes_win(agent, trials=3):
    """Tác tử phải chọn nước thắng ngay ở mọi thế trong WIN_IN_ONE."""
    for moves, best, note in WIN_IN_ONE:
        for t in range(trials):
            set_seed(t)
            env = conn.from_moves(moves)
            move = agent(env)
            assert move == best, f"{note}: chọn {move}, đúng phải là {best}"


def check_blocks(agent, trials=3):
    """Tác tử phải chặn nước thắng ngay của đối thủ ở mọi thế trong MUST_BLOCK."""
    for moves, best, note in MUST_BLOCK:
        for t in range(trials):
            set_seed(t)
            env = conn.from_moves(moves)
            move = agent(env)
            assert move == best, f"{note}: chọn {move}, đúng phải là {best}"


def check_never_illegal(agent, n_games=6, seed=0):
    """Đấu với tác tử ngẫu nhiên ở cả hai màu: không bao giờ đi nước sai, không sửa env."""
    set_seed(seed)
    for g in range(n_games):
        env = conn()
        agent_color = "red" if g % 2 == 0 else "yellow"
        while not env.done:
            if env.turn == agent_color:
                before = snapshot(env)
                move = agent(env)
                after = snapshot(env)
                assert same_snapshot(before, after), "tác tử đã làm thay đổi env"
                assert move in env.validinputs, f"nước đi không hợp lệ: {move}"
            else:
                move = random.choice(env.validinputs)
            env.step(move)


def play_one_game_moves(agent, seed):
    """Một ván tác tử (đỏ) đấu ngẫu nhiên (vàng) với seed cố định, trả về dãy nước."""
    set_seed(seed)
    env = conn()
    while not env.done:
        if env.turn == "red":
            move = agent(env)
        else:
            move = random.choice(env.validinputs)
        env.step(move)
    return list(env.history)


def check_reproducible(agent, seed=123):
    """Cùng seed thì cùng dãy nước đi."""
    first = play_one_game_moves(agent, seed)
    second = play_one_game_moves(agent, seed)
    assert first == second, "cùng seed nhưng ra hai ván khác nhau"

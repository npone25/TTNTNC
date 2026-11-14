"""Test cấp 4 (MCTS/UCT dạng cây) và cấp 7 (PUCT kết hợp mạng). Phụ trách: Nguyễn Trọng Phúc.

Cấp 7 dùng mạng giả hoặc mạng chưa huấn luyện: chỉ kiểm tra tìm kiếm đúng, không đo sức mạnh.
"""

import random
import pytest

from utils.conn_simple_env import conn
from conftest import (check_takes_win, check_blocks, check_never_illegal,
                      check_reproducible, set_seed)


# Cấp 4: MCTS/UCT dạng cây (Trọng Phúc)

from utils import mcts_uct as cap4


def cap4_agent(env):
    return cap4.mcts_uct(env, num_simulations=300)


def test_cap4_thang_ngay():
    check_takes_win(cap4_agent)


def test_cap4_chan_nuoc_thua():
    check_blocks(cap4_agent)


def test_cap4_khong_di_nuoc_sai_va_khong_sua_env():
    check_never_illegal(lambda env: cap4.mcts_uct(env, num_simulations=50), n_games=4)


def test_cap4_tai_lap_theo_seed():
    check_reproducible(lambda env: cap4.mcts_uct(env, num_simulations=50))


def test_cap4_dau_doan_dung_gia_tri():
    # thế có nước thắng ngay ở cột 4: giá trị của cột đó phải gần +1 và được thăm nhiều nhất
    set_seed(0)
    env = conn.from_moves([1, 7, 2, 7, 3, 6])
    cap4.mcts_uct(env, num_simulations=500)
    visits = cap4.LAST_INFO["visits"]
    values = cap4.LAST_INFO["values"]
    assert max(visits, key=visits.get) == 4
    assert values[4] > 0.99


def test_cap4_dung_so_mo_phong():
    set_seed(0)
    cap4.mcts_uct(conn(), num_simulations=200)
    assert cap4.LAST_INFO["simulations"] == 200
    # lần mô phỏng đầu tiên dùng để mở rộng gốc nên tổng lượt thăm của các con là 199
    assert sum(cap4.LAST_INFO["visits"].values()) == 199


def test_cap4_gioi_han_thoi_gian():
    cap4.mcts_uct(conn(), time_limit=0.1)
    assert 0.1 <= cap4.LAST_INFO["seconds"] < 0.3
    assert cap4.LAST_INFO["simulations"] > 1


@pytest.mark.slow
def test_cap4_thang_ngau_nhien():
    set_seed(1)
    wins = 0
    for g in range(40):
        env = conn()
        color = "red" if g % 2 == 0 else "yellow"
        while not env.done:
            if env.turn == color:
                move = cap4.mcts_uct(env, num_simulations=200)
            else:
                move = random.choice(env.validinputs)
            env.step(move)
        if (env.reward == 1 and color == "red") or (env.reward == -1 and color == "yellow"):
            wins += 1
    assert wins >= 38


# Cấp 7: PUCT trên cây kết hợp mạng (Trọng Phúc)
# Dùng mạng chưa huấn luyện hoặc mạng giả: chỉ kiểm tra tính đúng của tìm kiếm, không đo sức mạnh.


class FakePolicy:
    """Mạng policy giả: dồn xác suất strength vào một cột, phần còn lại chia đều."""

    def __init__(self, favorite, strength=0.9):
        self.favorite = favorite
        self.strength = strength
        self.inputs = [None]

    def __call__(self, x, training=False):
        probs = []
        for col in range(1, 8):
            if col == self.favorite:
                probs.append(self.strength)
            else:
                probs.append((1 - self.strength) / 6)
        return [probs]


class FakeValue:
    """Mạng value giả: luôn cho rằng thế cờ là hòa (giá trị 0)."""

    inputs = [None]

    def __call__(self, x, training=False):
        return [[1.0, 0.0, 0.0]]


def test_cap7_thang_ngay_du_tien_nghiem_sai():
    # mạng policy dồn 0.9 vào cột 7, nhưng nước thắng là cột 4: tìm kiếm phải sửa được
    from utils import alphago_tree as cap7
    set_seed(0)
    env = conn.from_moves([1, 7, 2, 7, 3, 6])
    move = cap7.alphago_tree(env, FakePolicy(7), FakeValue(), None, num_simulations=300, mix=0.0)
    assert move == 4


def test_cap7_thang_ngay_va_chan():
    from utils import alphago_tree as cap7

    def agent(env):
        return cap7.alphago_tree(env, FakePolicy(1, 0.3), FakeValue(), None, num_simulations=300, mix=0.5)
    check_takes_win(agent, trials=2)
    check_blocks(agent, trials=2)


def test_cap7_tien_nghiem_co_tac_dung():
    # thế mở đầu, ít mô phỏng: cột được mạng policy ưu tiên phải được thăm nhiều nhất
    from utils import alphago_tree as cap7
    set_seed(0)
    cap7.alphago_tree(conn(), FakePolicy(2, 0.9), FakeValue(), None, num_simulations=30, mix=0.0)
    visits = cap7.LAST_INFO["visits"]
    assert max(visits, key=visits.get) == 2


def test_cap7_khong_chon_cot_day():
    # mạng giả dồn xác suất vào cột 4 đã đầy: không bao giờ được chọn cột 4
    from utils import alphago_tree as cap7
    env = conn.from_moves([4, 4, 4, 4, 4, 4])
    for t in range(5):
        set_seed(t)
        move = cap7.alphago_tree(env, FakePolicy(4, 0.99), FakeValue(), None, num_simulations=50)
        assert move != 4
    assert 4 not in cap7.LAST_INFO["priors"]


def test_cap7_voi_mang_keras_that():
    pytest.importorskip("tensorflow")
    from utils import nets
    from utils import alphago_tree as cap7
    policy = nets.build_policy_net()
    value = nets.build_value_net()
    check_never_illegal(lambda env: cap7.alphago_tree(env, policy, value, policy, num_simulations=10),
                        n_games=2)
    cap7.alphago_tree(conn(), policy, value, None, time_limit=0.2)
    assert cap7.LAST_INFO["simulations"] >= 1


def test_cap7_bon_cau_hinh_ablation_deu_chay_dung():
    # cả 4 cấu hình đều phải thắng ngay và chặn đúng; cấu hình nào tắt mạng thì truyền None
    from utils import alphago_tree as cap7
    configs = [
        (None, None, False, False),            # MCTS thuần
        (FakePolicy(1, 0.3), None, True, False),   # thêm prior
        (None, FakeValue(), False, True),      # thêm value
        (FakePolicy(1, 0.3), FakeValue(), True, True),   # đầy đủ
    ]
    for prior, value, use_prior, use_value in configs:
        def agent(env):
            return cap7.alphago_tree(env, prior, value, None, num_simulations=300, mix=0.0,
                                     use_prior=use_prior, use_value=use_value)
        check_takes_win(agent, trials=1)
        check_blocks(agent, trials=1)


def test_cap7_tat_prior_thi_tien_nghiem_deu():
    from utils import alphago_tree as cap7
    set_seed(0)
    cap7.alphago_tree(conn(), FakePolicy(2, 0.9), FakeValue(), None, num_simulations=30,
                      mix=0.0, use_prior=False)
    priors = cap7.LAST_INFO["priors"]
    for p in priors.values():
        assert p == pytest.approx(1 / 7, abs=1e-3)

"""Test tập dữ liệu cho mạng: lật gương bàn cờ.

Phụ trách: Nguyễn Trọng Phúc.
"""

import numpy as np

from utils import data
from utils.conn_simple_env import conn


def test_lat_guong_doi_cot_va_nhan_policy():
    # quân đỏ ở cột 1: lật gương thì quân sang cột 7, nhãn cột 2 (chỉ số 1) thành cột 6 (chỉ số 5)
    env = conn.from_moves([1])
    X = np.array([data.to_player_view(env.state, env.turn)])
    y = np.array([1])
    X_flip, y_flip = data.mirror_examples(X, y, "policy")
    assert X_flip[0, 6, 0, 0] == X[0, 0, 0, 0]
    assert X_flip[0, 0, 0, 0] == 0
    assert y_flip[0] == 5


def test_lat_guong_giu_nhan_value():
    env = conn.from_moves([1, 2])
    X = np.array([data.to_player_view(env.state, env.turn)])
    y = np.array([2])
    X_flip, y_flip = data.mirror_examples(X, y, "value")
    assert y_flip[0] == 2


def test_lat_guong_chi_them_vao_tap_train():
    games = []
    for k in range(10):
        env = conn()
        history = []
        moves = [1, 2, 1, 2, 1, 2, 1] if k % 2 == 0 else [7, 6, 7, 6, 7, 6, 7]
        for col in moves:
            history.append((env.state.copy(), col, env.turn))
            env.step(col)
        games.append({"game_id": f"g{k}", "reward": env.reward, "history": history})
    plain = data.make_dataset(games, kind="policy", seed=0)
    flipped = data.make_dataset(games, kind="policy", seed=0, mirror=True)
    assert len(flipped["y_train"]) >= len(plain["y_train"])
    # val và test không có bản lật: chỉ bị bớt đi những thế đã có trong train
    assert len(flipped["y_val"]) <= len(plain["y_val"])
    assert len(flipped["y_test"]) <= len(plain["y_test"])

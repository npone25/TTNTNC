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


def test_gan_nhan_bang_tim_kiem_thay_nuoc_thang_ngay():
    # đỏ có 3 quân dọc ở cột 1, đến lượt đỏ: nhãn phải là cột 1 và điểm là thắng chắc
    label, best = data.label_with_search([1, 2, 1, 2, 1, 3], depth=3)
    assert label == 1
    assert best >= 1.0


def test_the_co_khong_trung_va_tap_nhan_tim_kiem():
    games = []
    for k in range(6):
        env = conn()
        history = []
        for col in [4, 4, 3, 5, 2, 6, 1]:
            history.append((env.state.copy(), col, env.turn))
            env.step(col)
        games.append({"game_id": f"g{k}", "reward": env.reward, "history": history})
    positions = data.game_positions(games)
    # sáu ván giống hệt nhau: chỉ còn 7 thế khác nhau, đều thuộc ván đầu
    assert len(positions) == 7
    labeled = [data._label_job((g, m, 2)) for g, m in positions]
    X, y, ids = data.search_policy_examples(labeled)
    assert X.shape == (7, 7, 6, 1)
    assert all(0 <= v <= 6 for v in y)


def test_policy_gradient_mang_mot_dau_vao():
    import keras
    from utils import nets, policy_gradient
    model = nets.build_policy_net()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    env = conn()
    steps = [{"flat": nets.encode_flat(env)[0], "conv": nets.encode_conv(env)[0],
              "action": 3, "mask": policy_gradient.legal_mask(env)}]
    loss = policy_gradient.reinforce_update(model, optimizer, steps, [1.0])
    assert np.isfinite(loss)

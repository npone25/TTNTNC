"""Test cấp 5 (policy_net.py) và cấp 6 (policy_gradient.py). Phụ trách: Trần Quỳnh Hương.

Dùng mạng chưa huấn luyện: chỉ kiểm tra đi đúng luật, tái lập được và vòng huấn luyện chạy được.
"""

import pytest

from utils.conn_simple_env import conn
from conftest import check_never_illegal, check_reproducible, set_seed


# Cấp 5 và cấp 6: mạng policy (policy_net.py, policy_gradient.py)
# Dùng mạng chưa huấn luyện: chỉ kiểm tra đi đúng luật và vòng huấn luyện chạy được.

def test_cap5_cap6_khong_di_nuoc_sai_va_tai_lap():
    pytest.importorskip("tensorflow")
    from utils import nets, policy_net, policy_gradient
    set_seed(0)
    policy = nets.build_policy_net()
    pg = nets.build_pg_net()
    check_never_illegal(lambda env: policy_net.policy_move(env, policy), n_games=2)
    check_never_illegal(lambda env: policy_gradient.pg_move(env, pg, sample=True), n_games=2)
    check_reproducible(lambda env: policy_net.policy_move(env, policy, sample=True))


def test_cap6_loi_nhuan_chiet_khau():
    pytest.importorskip("tensorflow")
    from utils import policy_gradient
    assert policy_gradient.discounted_returns(3, 1.0, 0.5) == [0.25, 0.5, 1.0]
    assert policy_gradient.discounted_returns(2, -1.0, 1.0) == [-1.0, -1.0]


def test_cap6_huan_luyen_hai_giai_doan_chay_duoc(tmp_path):
    pytest.importorskip("tensorflow")
    import numpy as np
    from utils import nets, policy_gradient
    set_seed(0)
    level5 = nets.build_policy_net()
    model = nets.build_pg_net()
    before = [w.numpy().copy() for w in model.trainable_variables]
    final = str(tmp_path / "pg.keras")
    report = policy_gradient.train_two_stages(model, level5, stage1_batches=2, stage2_batches=2,
                                              games_per_batch=2, checkpoint_dir=str(tmp_path / "ck"),
                                              checkpoint_every=1, final_path=final, verbose=False)
    assert len(report["stage1"]["mean_reward"]) == 2
    assert len(report["stage2"]["checkpoints"]) >= 2
    changed = False
    for old, new in zip(before, model.trainable_variables):
        if not np.allclose(old, new.numpy()):
            changed = True
    assert changed
    loaded = nets.load_model(final)
    assert policy_gradient.pg_move(conn(), loaded) in range(1, 8)

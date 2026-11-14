"""Test bộ thế cờ chiến thuật (utils/tactics.py)."""

from utils import tactics, rules
from utils.conn_simple_env import conn


def test_phan_nhom_dung_dinh_nghia():
    # đỏ có 3 quân ngang ở hàng dưới, cột 4 trống: đỏ đi là thắng ngay
    group, key = tactics.classify(conn.from_moves([1, 7, 2, 7, 3, 6]))
    assert group == "win_now" and key == [4]
    # cùng thế đó nhưng tới lượt vàng: vàng phải chặn cột 4
    group, key = tactics.classify(conn.from_moves([1, 7, 2, 7, 3]))
    assert group == "must_block" and key == [4]


def test_sinh_bo_the_co_nho_va_cham_diem(tmp_path):
    positions = tactics.build_tactics(per_group={"win_now": 2, "must_block": 2, "double_threat": 2},
                                      seed=1, depth=3)
    assert tactics.describe(positions) == {"win_now": 2, "must_block": 2, "double_threat": 2}
    for pos in positions:
        env = conn.from_moves(pos["moves"])
        assert not env.done
        assert env.turn == pos["to_move"]
        assert len(pos["answers"]) <= 2
    path = str(tmp_path / "tactics.json")
    tactics.save_tactics(positions, path)
    assert tactics.load_tactics(path) == positions
    # cấp 2 có luật thắng ngay và chặn nên phải đúng hết hai nhóm này
    acc = tactics.score_agent(rules.conn_think3, positions)
    assert acc["win_now"] == 1.0 and acc["must_block"] == 1.0

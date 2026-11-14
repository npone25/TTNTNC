"""Test giải đấu, log và thống kê. Phụ trách: Nguyễn Trọng Phúc."""

import pytest

from utils import arena
from utils.arena import (wilson_interval, time_stats, play_game,
                         run_match, round_robin, read_log, random_agent)


def always_column_one(env):
    """Tác tử cố ý đi cột 1 mãi, sẽ đi nước sai khi cột 1 đầy."""
    return 1


def test_wilson():
    low, high = wilson_interval(0.5, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    low, high = wilson_interval(1.0, 10)
    assert high == 1.0 and low > 0.6


def test_time_stats():
    mean, p95 = time_stats([0.1, 0.2, 0.3, 0.4])
    assert mean == pytest.approx(0.25)
    assert 0.3 < p95 <= 0.4


def test_play_game_hop_le():
    result = play_game(random_agent, random_agent, seed=5)
    assert result["winner"] in ("red", "yellow", "draw")
    assert len(result["moves"]) == len(result["move_times"])
    assert 7 <= len(result["moves"]) <= 42


def test_play_game_tai_lap():
    first = play_game(random_agent, random_agent, seed=9)
    second = play_game(random_agent, random_agent, seed=9)
    assert first["moves"] == second["moves"]


def test_nuoc_sai_bi_xu_thua():
    result = play_game(always_column_one, always_column_one, seed=1)
    # đỏ đi cột 1 ba lần, vàng ba lần, đến lượt đỏ thứ tư cột 1 đã đầy
    assert result["forfeit"] is True
    assert result["winner"] == "yellow"


def test_run_match_doi_ben_va_ghi_log(tmp_path):
    log = str(tmp_path / "log.jsonl")
    summary = run_match(random_agent, random_agent, "A", "B", n_games=6, seed=3, log_path=log)
    records = read_log(log)
    assert len(records) == 6
    reds = []
    for r in records:
        reds.append(r["red"])
    assert reds == ["A", "B", "A", "B", "A", "B"]
    assert summary["games"] == 6
    assert summary["wins"] + summary["draws"] + summary["losses"] == 6


def test_run_match_chay_tiep_khi_bi_ngat(tmp_path):
    log = str(tmp_path / "log.jsonl")
    run_match(random_agent, random_agent, "A", "B", n_games=4, seed=3, log_path=log)
    full = run_match(random_agent, random_agent, "A", "B", n_games=8, seed=3, log_path=log)
    assert len(read_log(log)) == 8
    # chạy một mạch 8 ván phải ra đúng kết quả như chạy 4 rồi chạy tiếp
    log2 = str(tmp_path / "log2.jsonl")
    once = run_match(random_agent, random_agent, "A", "B", n_games=8, seed=3, log_path=log2)
    assert full["wins"] == once["wins"] and full["losses"] == once["losses"]


def test_round_robin_ma_tran(tmp_path):
    agents = {"A": random_agent, "B": random_agent, "C": random_agent}
    summaries, matrix = round_robin(agents, n_games=4, seed=1, log_path=str(tmp_path / "rr.jsonl"))
    assert len(summaries) == 3
    assert matrix["A"]["B"] + matrix["B"]["A"] == pytest.approx(1.0)


def test_build_agent_tu_config():
    cfg = arena.load_config("config.yaml")
    agent = arena.build_agent(cfg["agents"]["cap4"], overrides={"num_simulations": 20})
    result = play_game(agent, random_agent, seed=2)
    assert result["winner"] in ("red", "yellow", "draw")
    # thông tin số mô phỏng được ghi lại cho mỗi nước của cấp 4
    assert result["move_info"][0]["simulations"] == 20


def test_ghi_de_che_do_thoi_gian_bo_so_mo_phong():
    cfg = arena.load_config("config.yaml")
    agent, info = arena.build_agent(cfg["agents"]["cap4"], overrides={"time_limit": 0.05})
    from utils.conn_simple_env import conn
    agent(conn())
    assert info()["seconds"] >= 0.05

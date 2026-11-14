"""Giải đấu và thống kê.

Kế thừa: ý tưởng hàm one_conn_game trong utils/ch03util.py (chơi một ván).
Sửa: không áp dụng.
Thêm: giải vòng tròn đổi bên đi trước; nhiều seed; hai chế độ ngân sách (quy định và mặc định); log JSONL; thắng/hòa/thua, score, khoảng tin cậy Wilson 95%, thời gian TB và p95, số nút/mô phỏng; ma trận đối đầu; ảnh hưởng bên đi trước.
Phụ trách: Nguyễn Trọng Phúc.

Cách dùng nhanh trong notebook:
    from utils.arena import load_config, build_agent, run_match
    cfg = load_config()
    a = build_agent(cfg["agents"]["cap3"])
    b = build_agent(cfg["agents"]["cap4"], overrides={"time_limit": 0.5})
    summary = run_match(a, b, "cap3", "cap4", n_games=200, seed=1, log_path="results/logs/rq1.jsonl")

Một "tác tử" trong file này là cặp (hàm chọn nước, hàm lấy thông tin):
- hàm chọn nước nhận env và trả về cột 1-7 (quy ước của sách);
- hàm lấy thông tin trả về dict số nút/mô phỏng của nước vừa đi (có thể rỗng).
Quy ước: file tác tử nào muốn ghi số nút/mô phỏng thì có biến toàn cục LAST_INFO (dict).
"""

import json
import math
import os
import random
import time

import numpy as np

from utils.conn_simple_env import conn, InvalidMoveError


# Thống kê

def score_from_counts(wins, draws, losses):
    """Score = (thắng + 0.5 * hòa) / số ván."""
    games = wins + draws + losses
    if games == 0:
        return 0.0
    return (wins + 0.5 * draws) / games


def wilson_interval(score, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ score trên n ván. Trả về (thấp, cao)."""
    if n == 0:
        return (0.0, 1.0)
    denominator = 1 + z * z / n
    center = (score + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def time_stats(times):
    """Thời gian trung bình và phân vị 95 (giây) của một danh sách thời gian mỗi nước."""
    if len(times) == 0:
        return 0.0, 0.0
    values = np.array(times, dtype=float)
    return float(values.mean()), float(np.percentile(values, 95))


# Chơi ván và đấu cặp

def no_info():
    return {}


def as_agent(agent):
    """Chấp nhận cả hàm chọn nước đơn lẻ lẫn cặp (hàm chọn nước, hàm lấy thông tin)."""
    if isinstance(agent, tuple):
        return agent
    return (agent, no_info)


def set_seed(seed):
    """Đặt seed cho random và numpy (mã sách dùng cả hai)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def play_game(red_agent, yellow_agent, seed=None):
    """Chơi một ván. red_agent đi trước.

    Tác tử nhận một BẢN SAO của bàn cờ nên không thể làm hỏng ván đấu.
    Nếu tác tử trả về nước không hợp lệ thì bị xử thua (ghi forfeit=True).
    Trả về dict gồm: moves, winner ("red"/"yellow"/"draw"), forfeit, move_times, move_info.
    """
    if seed is not None:
        set_seed(seed)
    red_fn, red_info = as_agent(red_agent)
    yellow_fn, yellow_info = as_agent(yellow_agent)
    env = conn()
    move_times = []
    move_info = []
    forfeit = False
    winner = None
    while not env.done:
        if env.turn == "red":
            agent_fn, info_fn = red_fn, red_info
        else:
            agent_fn, info_fn = yellow_fn, yellow_info
        start = time.perf_counter()
        move = agent_fn(env.clone())
        move_times.append(time.perf_counter() - start)
        move_info.append(info_fn())
        try:
            env.step(move)
        except InvalidMoveError:
            forfeit = True
            winner = "yellow" if env.turn == "red" else "red"
            break
    if winner is None:
        winner = env.winner()
    moves = []
    for m in env.history:
        moves.append(int(m))
    return {
        "moves": moves,
        "winner": winner,
        "forfeit": forfeit,
        "move_times": move_times,
        "move_info": move_info,
    }


def read_log(path):
    """Đọc file log JSONL, trả về danh sách bản ghi (mỗi dòng một ván)."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_log(path, record):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_records(records, name_a, name_b):
    """Tính thống kê của A khi đấu B từ các bản ghi log của cặp (A, B)."""
    wins = draws = losses = 0
    red_points = 0.0
    red_games = 0
    yellow_points = 0.0
    yellow_games = 0
    a_times = []
    b_times = []
    forfeits = 0
    for r in records:
        a_color = "red" if r["red"] == name_a else "yellow"
        if r["winner"] == "draw":
            draws += 1
            points = 0.5
        elif r["winner"] == a_color:
            wins += 1
            points = 1.0
        else:
            losses += 1
            points = 0.0
        if r.get("forfeit"):
            forfeits += 1
        if a_color == "red":
            red_points += points
            red_games += 1
        else:
            yellow_points += points
            yellow_games += 1
        # nước chẵn (0, 2, 4...) là của bên đỏ
        for i, t in enumerate(r["move_times"]):
            mover = "red" if i % 2 == 0 else "yellow"
            if mover == a_color:
                a_times.append(t)
            else:
                b_times.append(t)
    games = wins + draws + losses
    score = score_from_counts(wins, draws, losses)
    low, high = wilson_interval(score, games)
    a_mean, a_p95 = time_stats(a_times)
    b_mean, b_p95 = time_stats(b_times)
    return {
        "agent": name_a, "opponent": name_b, "games": games,
        "wins": wins, "draws": draws, "losses": losses, "forfeits": forfeits,
        "score": score, "ci_low": low, "ci_high": high,
        "score_as_red": red_points / red_games if red_games else None,
        "score_as_yellow": yellow_points / yellow_games if yellow_games else None,
        "time_mean": a_mean, "time_p95": a_p95,
        "opponent_time_mean": b_mean, "opponent_time_p95": b_p95,
    }


def run_match(agent_a, agent_b, name_a, name_b, n_games, seed=0, log_path=None, resume=True):
    """Đấu n_games ván giữa A và B, đổi bên đi trước: ván chẵn A cầm đỏ, ván lẻ B cầm đỏ.

    Mỗi ván có seed riêng = seed * 100000 + số thứ tự ván, nên chạy lại ra đúng ván cũ
    (với tác tử cố định số mô phỏng/độ sâu).
    Nếu có log_path và resume=True: các ván đã có trong log thì bỏ qua (chạy tiếp khi bị ngắt).
    Trả về dict thống kê theo góc của A.
    """
    done_games = {}
    if log_path is not None and resume:
        for r in read_log(log_path):
            if r["pair"] == [name_a, name_b]:
                done_games[r["game_index"]] = r

    records = []
    for i in range(n_games):
        if i in done_games:
            records.append(done_games[i])
            continue
        game_seed = seed * 100000 + i
        if i % 2 == 0:
            result = play_game(agent_a, agent_b, seed=game_seed)
            red_name, yellow_name = name_a, name_b
        else:
            result = play_game(agent_b, agent_a, seed=game_seed)
            red_name, yellow_name = name_b, name_a
        record = {
            "pair": [name_a, name_b], "game_index": i, "seed": game_seed,
            "red": red_name, "yellow": yellow_name,
        }
        record.update(result)
        if log_path is not None:
            append_log(log_path, record)
        records.append(record)
    return summarize_records(records, name_a, name_b)


def round_robin(agents, n_games, seed=0, log_path=None, resume=True):
    """Giải vòng tròn. agents là dict {tên: tác tử}.

    Trả về (danh sách thống kê từng cặp, ma trận score: matrix[A][B] = score của A khi gặp B).
    """
    names = list(agents.keys())
    summaries = []
    matrix = {}
    for name in names:
        matrix[name] = {}
    pair_index = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            summary = run_match(agents[a], agents[b], a, b, n_games,
                                seed=seed + pair_index, log_path=log_path, resume=resume)
            summaries.append(summary)
            matrix[a][b] = summary["score"]
            matrix[b][a] = 1.0 - summary["score"]
            pair_index += 1
    return summaries, matrix


def first_player_stats(records):
    """Ảnh hưởng bên đi trước: tỷ lệ đỏ thắng, hòa, vàng thắng trên toàn bộ log."""
    red = draw = yellow = 0
    for r in records:
        if r["winner"] == "red":
            red += 1
        elif r["winner"] == "yellow":
            yellow += 1
        else:
            draw += 1
    total = red + draw + yellow
    if total == 0:
        return {"games": 0}
    return {"games": total, "red_win_rate": red / total,
            "draw_rate": draw / total, "yellow_win_rate": yellow / total}


def format_summary(s):
    """Một dòng chữ tóm tắt kết quả một cặp, để in ra notebook."""
    return (f"{s['agent']} vs {s['opponent']}: {s['wins']}-{s['draws']}-{s['losses']} "
            f"(thắng-hòa-thua), score {s['score']:.3f} "
            f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}], "
            f"thời gian TB {s['time_mean']*1000:.1f} ms, p95 {s['time_p95']*1000:.1f} ms")


# Đọc cấu hình và tạo tác tử theo tên

def load_config(path="config.yaml"):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def module_info(module):
    """Hàm lấy LAST_INFO của một module tác tử (rỗng nếu module không có)."""
    def info():
        return dict(getattr(module, "LAST_INFO", {}))
    return info


def random_agent(env):
    """Tác tử ngẫu nhiên đơn giản (giống conn_random của sách), dùng cho test và làm mốc."""
    return env.sample()


def build_agent(spec, overrides=None):
    """Tạo tác tử từ một mục trong config.yaml.

    spec có dạng {"type": ..., "params": {...}, các đường dẫn mô hình nếu cần}.
    overrides: tham số ghi đè, ví dụ {"time_limit": 0.5} cho chế độ cố định thời gian.
    Trả về cặp (hàm chọn nước, hàm lấy thông tin).
    """
    kind = spec["type"]
    params = dict(spec.get("params") or {})
    if overrides:
        # hai chế độ ngân sách loại trừ nhau: ghi đè chế độ này thì bỏ chế độ kia trong config
        if "time_limit" in overrides:
            params.pop("num_simulations", None)
        if "num_simulations" in overrides:
            params.pop("time_limit", None)
        params.update(overrides)

    if kind == "random":
        return (random_agent, no_info)

    if kind == "cap1":
        from utils import rules
        return (rules.conn_random, no_info)

    if kind == "cap2":
        from utils import rules
        return (rules.conn_think3, module_info(rules))

    if kind == "cap3":
        from utils import alphabeta
        def cap3(env):
            return alphabeta.MiniMax_conn(env, **params)
        return (cap3, module_info(alphabeta))

    if kind == "cap4":
        from utils import mcts_uct
        def cap4(env):
            return mcts_uct.mcts_uct(env, **params)
        return (cap4, module_info(mcts_uct))

    if kind == "cap5":
        from utils import nets, policy_net
        model = nets.load_model(spec["model"])
        def cap5(env):
            return policy_net.policy_move(env, model, **params)
        return (cap5, module_info(policy_net))

    if kind == "cap6":
        from utils import nets, policy_gradient
        model = nets.load_model(spec["model"])
        def cap6(env):
            return policy_gradient.pg_move(env, model, **params)
        return (cap6, module_info(policy_gradient))

    if kind == "cap7":
        from utils import nets, alphago_tree
        # mạng nào không khai báo thì để None (dùng cho các cấu hình ablation RQ3)
        prior_model = None
        value_model = None
        fast_model = None
        if spec.get("prior_model"):
            prior_model = nets.load_model(spec["prior_model"])
        if spec.get("value_model"):
            value_model = nets.load_model(spec["value_model"])
        if spec.get("fast_model"):
            fast_model = nets.load_model(spec["fast_model"])
        def cap7(env):
            return alphago_tree.alphago_tree(env, prior_model, value_model, fast_model, **params)
        return (cap7, module_info(alphago_tree))

    raise ValueError(f"Không biết loại tác tử: {kind}")

"""Bộ thế cờ chiến thuật (khoảng 40 thế) và chấm độ chính xác.

Kế thừa: không (tự viết).
Sửa: không áp dụng.
Thêm: ba nhóm thế: thắng ngay, buộc chặn, tạo đôi đe dọa; nhãn bằng Alpha-Beta tìm sâu; in độ chính xác theo nhóm cho tác tử bất kỳ.
Phụ trách: Phạm Thiên Phúc.

Ba nhóm thế cờ (luôn xét theo người SẮP ĐI):
- win_now:      mình có nước thắng ngay.
- must_block:   mình không thắng ngay được, đối thủ có đúng một chỗ thắng ngay ở lượt sau.
- double_threat: chưa ai thắng ngay được, nhưng mình có một nước tạo ra cùng lúc hai chỗ
                 thắng (đòn đôi); đối thủ chỉ chặn được một chỗ.
Thế cờ được lấy từ các ván ngẫu nhiên có seed, rồi lọc theo đúng định nghĩa nhóm.

Nhãn (đáp án) của mỗi thế: các nước có điểm cao nhất khi Alpha-Beta tìm sâu (mặc định 5 nước).
Chỉ giữ thế nào mà đáp án ít (tối đa 2 cột), để câu hỏi rõ ràng; nước then chốt của nhóm
(nước thắng, nước chặn, nước tạo đòn đôi) phải nằm trong đáp án.

Cách dùng:
    positions = build_tactics()              # sinh bộ thế cờ (vài chục giây)
    save_tactics(positions, "files/tactics.json")
    positions = load_tactics("files/tactics.json")
    acc = score_agent(rules.conn_think3, positions)   # {"win_now": 1.0, "must_block": 1.0, ...}
"""

import json
import os
import random

from utils.conn_simple_env import conn
from utils import alphabeta

GROUPS = ["win_now", "must_block", "double_threat"]


def other(color):
    if color == "red":
        return "yellow"
    return "red"


def winning_columns(env, color):
    """Các cột mà bên color thả vào là thắng ngay (thử trên bản sao, đổi lượt nếu cần)."""
    cols = []
    for col in env.validinputs:
        test = env.clone()
        test.turn = color
        test.step(col)
        if test.done and test.reward != 0:
            cols.append(col)
    return cols


def double_threat_moves(env):
    """Các nước tạo đòn đôi: sau nước đó mình có ít nhất hai cột thắng ngay khác nhau,
    còn đối thủ không có nước thắng ngay để thoát."""
    me = env.turn
    moves = []
    for col in env.validinputs:
        after = env.clone()
        after.step(col)
        if after.done:
            continue
        if winning_columns(after, other(me)):
            continue
        if len(winning_columns(after, me)) >= 2:
            moves.append(col)
    return moves


def classify(env):
    """Xếp thế cờ vào một nhóm. Trả về (tên nhóm, các nước then chốt) hoặc (None, [])."""
    me = env.turn
    mine = winning_columns(env, me)
    if mine:
        return "win_now", mine
    theirs = winning_columns(env, other(me))
    if len(theirs) == 1:
        return "must_block", theirs
    if theirs:
        # đối thủ có hai chỗ thắng: mình đã thua, không dùng làm câu hỏi
        return None, []
    combo = double_threat_moves(env)
    if combo:
        return "double_threat", combo
    return None, []


def deep_label(env, depth):
    """Đáp án theo Alpha-Beta tìm sâu: các cột có điểm cao nhất."""
    scores = alphabeta.Search(prune=True).root_scores(env, depth)
    best = max(scores.values())
    answers = []
    for col, score in scores.items():
        if score == best:
            answers.append(col)
    return sorted(answers), best


def build_tactics(per_group=None, seed=2026, depth=5, max_games=20000, max_answers=2):
    """Sinh bộ thế cờ từ các ván ngẫu nhiên có seed.

    per_group: số thế cần cho mỗi nhóm, mặc định 14 / 13 / 13 (tổng 40).
    Mỗi ván ngẫu nhiên chỉ lấy tối đa một thế, để các thế không quá giống nhau.
    """
    if per_group is None:
        per_group = {"win_now": 14, "must_block": 13, "double_threat": 13}
    rng = random.Random(seed)
    positions = []
    counts = {g: 0 for g in GROUPS}
    seen = set()
    for game in range(max_games):
        if all(counts[g] >= per_group[g] for g in GROUPS):
            break
        env = conn()
        moves = []
        # dừng ở một độ dài ngẫu nhiên để có thế cờ ở cả đầu, giữa và cuối ván
        stop_at = rng.randint(6, 30)
        while not env.done and len(moves) < stop_at:
            col = rng.choice(env.validinputs)
            env.step(col)
            moves.append(col)
        if env.done:
            continue
        group, key_moves = classify(env)
        if group is None or counts[group] >= per_group[group]:
            continue
        key = env.state.tobytes() + env.turn.encode()
        if key in seen:
            continue
        answers, best = deep_label(env, depth)
        if len(answers) > max_answers:
            continue
        if not any(m in answers for m in key_moves):
            continue
        seen.add(key)
        counts[group] += 1
        positions.append({
            "id": f"{group}_{counts[group]:02d}",
            "group": group,
            "moves": moves,
            "to_move": env.turn,
            "answers": answers,
            "key_moves": sorted(key_moves),
            "search_depth": depth,
            "best_score": best,
        })
    return positions


def save_tactics(positions, path="files/tactics.json"):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=1)


def load_tactics(path="files/tactics.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_agent(agent, positions, seed=0):
    """Tỷ lệ chọn đúng của tác tử theo từng nhóm, cộng thêm "all" cho toàn bộ.

    Tác tử nhận bản sao của thế cờ, nên không làm hỏng bộ thế cờ.
    Mỗi thế đặt lại seed để tác tử có yếu tố ngẫu nhiên vẫn cho kết quả lặp lại được.
    """
    right = {g: 0 for g in GROUPS}
    total = {g: 0 for g in GROUPS}
    for i, pos in enumerate(positions):
        random.seed(seed + i)
        try:
            import numpy as np
            np.random.seed(seed + i)
        except ImportError:
            pass
        env = conn.from_moves(pos["moves"])
        move = agent(env.clone())
        total[pos["group"]] += 1
        if move in pos["answers"]:
            right[pos["group"]] += 1
    result = {}
    for g in GROUPS:
        result[g] = right[g] / total[g] if total[g] else None
    all_total = sum(total.values())
    result["all"] = sum(right.values()) / all_total if all_total else None
    return result


def describe(positions):
    """Số thế của từng nhóm, để in ra notebook."""
    counts = {g: 0 for g in GROUPS}
    for pos in positions:
        counts[pos["group"]] += 1
    return counts

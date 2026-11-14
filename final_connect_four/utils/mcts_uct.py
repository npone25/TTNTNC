"""Cấp 4: MCTS/UCT dạng cây.

Kế thừa: công thức UCT và mô phỏng ngẫu nhiên trong utils/ch08util.py (select, simulate), sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: dùng lõi cây trong mcts_tree.py thay cho thống kê phẳng ở gốc của sách.
Thêm: hàm tác tử nhận env trả về cột; chế độ số mô phỏng và thời gian; ghi số mô phỏng mỗi nước. Không thêm luật thắng/chặn.
Phụ trách: Nguyễn Trọng Phúc.

Cách dùng:
    from utils.mcts_uct import mcts_uct
    move = mcts_uct(env, num_simulations=400)
    move = mcts_uct(env, time_limit=0.5)
Sau mỗi lần gọi, LAST_INFO chứa số mô phỏng và thời gian của nước vừa chọn.
"""

import math
import random

from utils.mcts_tree import run_search, most_visited_move, root_statistics

# Thông tin của nước gần nhất (giải đấu đọc biến này để ghi log)
LAST_INFO = {}


def uct_select_child(node, temperature=1.4):
    """Chọn con theo UCT: q + temperature * sqrt(ln(N cha) / N con).

    Công thức cân bằng hai mong muốn:
    - q: con đang có điểm trung bình cao thì nên thử tiếp (khai thác);
    - phần căn: con ít được thử so với cha thì được cộng thêm, để không bỏ sót
      nước hay chỉ vì vài lần đầu xui (khám phá). Thử càng nhiều thì phần này càng nhỏ.
    temperature là hằng số khám phá (sách đặt tên temperature, mặc định 1.4): lớn thì
    thử rộng hơn, nhỏ thì dồn vào nước đang tốt.
    Con chưa thăm lần nào được ưu tiên trước (giống sách), chọn ngẫu nhiên giữa chúng.
    """
    unvisited = []
    for child in node.children.values():
        if child.visits == 0:
            unvisited.append(child)
    if len(unvisited) > 0:
        return random.choice(unvisited)

    best_score = None
    best_children = []
    log_parent = math.log(node.visits)
    for child in node.children.values():
        score = child.q() + temperature * math.sqrt(log_parent / child.visits)
        if best_score is None or score > best_score:
            best_score = score
            best_children = [child]
        elif score == best_score:
            best_children.append(child)
    return random.choice(best_children)


def random_rollout_value(env):
    """Chơi ngẫu nhiên đến hết ván từ bản sao của env, để ước lượng thế cờ tốt hay xấu.

    Một ván ngẫu nhiên thì rất hên xui, nhưng cộng dồn qua hàng trăm lần mô phỏng thì
    thế tốt sẽ thắng nhiều hơn thế xấu. Các nước đi trong ván ngẫu nhiên không được ghi vào cây.

    Trả về kết quả theo góc NGƯỜI SẮP ĐI ở env: +1 thắng, -1 thua, 0 hòa.
    """
    player_to_move = env.turn
    sim_env = env.clone()
    while not sim_env.done:
        sim_env.step(random.choice(sim_env.validinputs))
    if sim_env.reward == 0:
        return 0.0
    red_won = sim_env.reward == 1
    if (red_won and player_to_move == "red") or (not red_won and player_to_move == "yellow"):
        return 1.0
    return -1.0


def uniform_rollout_evaluator(env):
    """Đánh giá lá cho cấp 4: tiên nghiệm đều trên các cột hợp lệ, giá trị bằng một ván ngẫu nhiên."""
    priors = {}
    legal = env.validinputs
    for move in legal:
        priors[move] = 1.0 / len(legal)
    value = random_rollout_value(env)
    return priors, value


def mcts_uct(env, num_simulations=None, time_limit=None, temperature=1.4):
    """Tác tử cấp 4. Nhận env, trả về cột 1-7. Không làm thay đổi env.

    Mặc định 400 mô phỏng nếu không chỉ định chế độ nào.
    """
    global LAST_INFO
    if num_simulations is None and time_limit is None:
        num_simulations = 400
    # chỉ còn một nước thì đi luôn, khỏi tìm kiếm (giống sách)
    if len(env.validinputs) == 1:
        LAST_INFO = {"simulations": 0, "seconds": 0.0}
        return env.validinputs[0]

    def select_child(node):
        return uct_select_child(node, temperature)

    root, info = run_search(env, select_child, uniform_rollout_evaluator,
                            num_simulations=num_simulations, time_limit=time_limit)
    move = most_visited_move(root)
    visits, values = root_statistics(root)
    info["visits"] = visits
    info["values"] = values
    LAST_INFO = info
    return move

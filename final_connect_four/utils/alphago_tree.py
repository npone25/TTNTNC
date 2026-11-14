"""Cấp 7: tìm kiếm cây PUCT kết hợp mạng policy, policy nhanh và value (cảm hứng AlphaGo).

Kế thừa: ba thành phần mạng và cách dùng trong utils/ch17util.py và notebook ch18, sách AlphaGo Simplified (Mark Liu), giấy phép MIT: tiên nghiệm từ mạng policy, mô phỏng bằng mạng policy nhanh tới một độ sâu, đánh giá cuối bằng mạng value.
Sửa: thay tìm kiếm phẳng ở gốc bằng PUCT trên cây, dùng lõi trong mcts_tree.py; gọi mạng qua nets.call_model (tf.function) thay model.predict.
Thêm: chế độ cố định số mô phỏng và thời gian; tham số mix trộn giá trị mạng value với giá trị mô phỏng; hai công tắc use_prior, use_value để làm ablation RQ3; ghi LAST_INFO.
Phụ trách: Nguyễn Trọng Phúc.

Thành phần AlphaGo đã hiện thực: mạng policy cho tiên nghiệm (học có giám sát, cấp 5; có thể đổi sang mạng
policy gradient cấp 6), mạng policy nhanh cho mô phỏng, mạng value đánh giá thế cờ, tìm kiếm cây với
công thức PUCT, trộn giá trị value và mô phỏng theo hệ số mix (λ trong AlphaGo).
Chưa hiện thực: mạng rollout tuyến tính riêng siêu nhanh, tìm kiếm song song, mạng value huấn luyện từ ván
tự chơi của chính mạng RL quy mô lớn.

Cách dùng:
    from utils.alphago_tree import alphago_tree
    move = alphago_tree(env, prior_model, value_model, fast_model, num_simulations=200)

Bốn cấu hình ablation cho RQ3 (cùng một hàm, chỉ khác hai công tắc):
- MCTS thuần:     use_prior=False, use_value=False  (tiên nghiệm đều, mô phỏng ngẫu nhiên tới hết ván)
- Thêm prior:     use_prior=True,  use_value=False  (tiên nghiệm từ mạng policy, mô phỏng tới hết ván)
- Thêm value:     use_prior=False, use_value=True   (tiên nghiệm đều, đánh giá lá bằng mạng value)
- Đầy đủ (cấp 7): use_prior=True,  use_value=True
Khi tắt một thành phần thì không cần truyền mạng tương ứng (có thể để None).
"""

import math
import random

from utils import nets
from utils.mcts_tree import run_search, most_visited_move, root_statistics

# Thông tin của nước gần nhất (giải đấu đọc biến này để ghi log)
LAST_INFO = {}


def puct_select_child(node, c_puct=1.5):
    """Chọn con theo PUCT: Q + c_puct * P * sqrt(N cha) / (1 + N con).

    Giống UCT ở cấp 4 nhưng phần khám phá nhân thêm P, xác suất mạng policy gán cho nước đó:
    nước mạng cho là hay sẽ được thử sớm và nhiều hơn, nước mạng cho là dở hiếm khi được thử.
    Khác UCT, con chưa thăm không được ưu tiên tuyệt đối mà có Q = 0 và chỉ dựa vào P.
    Hòa điểm thì chọn ngẫu nhiên.
    """
    sqrt_parent = math.sqrt(node.visits)
    best_score = None
    best_children = []
    for child in node.children.values():
        score = child.q() + c_puct * child.prior * sqrt_parent / (1 + child.visits)
        if best_score is None or score > best_score:
            best_score = score
            best_children = [child]
        elif score == best_score:
            best_children.append(child)
    return random.choice(best_children)


def sample_move(probs):
    """Rút một cột theo phân phối xác suất probs (dict {cột: xác suất})."""
    moves = list(probs.keys())
    weights = list(probs.values())
    return random.choices(moves, weights=weights)[0]


def rollout_value(env, fast_model, value_model, depth):
    """Mô phỏng như sách: đi tối đa depth nước bằng mạng policy nhanh (hoặc ngẫu nhiên nếu
    fast_model là None), rồi đánh giá bằng mạng value nếu ván chưa kết thúc.

    Trả về giá trị theo góc NGƯỜI SẮP ĐI ở env ban đầu, trong [-1, 1].
    """
    player = env.turn
    sim_env = env.clone()
    for _ in range(depth):
        if sim_env.done:
            break
        if fast_model is None:
            move = random.choice(sim_env.validinputs)
        else:
            move = sample_move(nets.move_probabilities(sim_env, fast_model))
        sim_env.step(move)
    if sim_env.done:
        if sim_env.reward == 0:
            return 0.0
        red_won = sim_env.reward == 1
        if (red_won and player == "red") or (not red_won and player == "yellow"):
            return 1.0
        return -1.0
    # chưa hết ván: mạng value cho giá trị theo góc người sắp đi ở sim_env
    value = nets.value_estimate(sim_env, value_model)
    if sim_env.turn == player:
        return value
    return -value


def uniform_priors(env):
    """Tiên nghiệm chia đều cho các cột hợp lệ (dùng khi tắt mạng policy)."""
    priors = {}
    for move in env.validinputs:
        priors[move] = 1.0 / len(env.validinputs)
    return priors


def make_evaluator(prior_model, value_model, fast_model=None, mix=0.5, rollout_depth=10,
                   use_prior=True, use_value=True):
    """Tạo hàm đánh giá lá cho lõi cây.

    Tiên nghiệm: từ mạng policy nếu use_prior, ngược lại chia đều.
    Giá trị lá khi use_value: (1 - mix) * giá trị mạng value + mix * giá trị mô phỏng.
        mix = 0: chỉ dùng mạng value (nhanh nhất). mix = 1: chỉ dùng mô phỏng.
    Giá trị lá khi tắt use_value: mô phỏng tới HẾT ván, không dùng mạng value.
    """
    def evaluate_leaf(env):
        # tiên nghiệm: mạng policy gợi ý cột nào đáng thử, dùng cho bước mở rộng
        if use_prior:
            priors = nets.move_probabilities(env, prior_model)
        else:
            priors = uniform_priors(env)
        if not use_value:
            # ván Connect Four dài tối đa 42 nước nên mô phỏng 42 nước luôn tới hết ván
            return priors, rollout_value(env, fast_model, None, 42)
        # trộn hai nguồn: mạng value nhìn cả bàn một lần, mô phỏng nhanh thì đi thử vài nước
        value = 0.0
        if mix < 1:
            value += (1 - mix) * nets.value_estimate(env, value_model)
        if mix > 0:
            value += mix * rollout_value(env, fast_model, value_model, rollout_depth)
        return priors, value
    return evaluate_leaf


def alphago_tree(env, prior_model, value_model, fast_model=None, num_simulations=None,
                 time_limit=None, c_puct=1.5, mix=0.5, rollout_depth=10,
                 use_prior=True, use_value=True):
    """Tác tử cấp 7. Nhận env, trả về cột 1-7. Không làm thay đổi env.

    Mặc định 200 mô phỏng nếu không chỉ định chế độ nào.
    """
    global LAST_INFO
    if num_simulations is None and time_limit is None:
        num_simulations = 200
    if len(env.validinputs) == 1:
        LAST_INFO = {"simulations": 0, "seconds": 0.0}
        return env.validinputs[0]

    evaluate_leaf = make_evaluator(prior_model, value_model, fast_model, mix, rollout_depth,
                                   use_prior, use_value)

    def select_child(node):
        return puct_select_child(node, c_puct)

    root, info = run_search(env, select_child, evaluate_leaf,
                            num_simulations=num_simulations, time_limit=time_limit)
    move = most_visited_move(root)
    visits, values = root_statistics(root)
    priors = {}
    for m, child in root.children.items():
        priors[m] = round(child.prior, 4)
    info["visits"] = visits
    info["values"] = values
    info["priors"] = priors
    LAST_INFO = info
    return move

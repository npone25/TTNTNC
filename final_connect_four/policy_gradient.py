"""Cấp 6: mạng policy huấn luyện bằng policy gradient (REINFORCE).

Kế thừa: notebook ch15 (playing_red, playing_yellow, discount_rs, vòng huấn luyện bằng tf.GradientTape), sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: chỉ rút nước trong các cột hợp lệ (sách cho rút cả cột đầy rồi phạt -1, tốn nhiều ván vô ích);
     một hàm chơi cho cả hai màu thay cho playing_red/playing_yellow; đối thủ giai đoạn 1 là mô hình cấp 5 do nhóm tự huấn luyện.
Thêm: giai đoạn 2 tự chơi với các checkpoint cũ của chính nó; lưu checkpoint đều đặn; ghi đường cong học
      (phần thưởng trung bình mỗi lô và điểm số khi đấu với đối thủ cố định). Mạng tạo bằng nets.build_pg_net().
Phụ trách: Trần Quỳnh Hương.

Ý tưởng REINFORCE:
- Cho mạng tự chơi, mỗi nước rút theo xác suất của mạng (rút mẫu chính là cách khám phá).
- Hết ván, mỗi nước của mạng nhận một "lợi nhuận": kết quả ván (+1 thắng, -1 thua, 0 hòa)
  nhân với gamma mũ số nước còn lại đến cuối ván. Nước càng gần cuối càng gánh nhiều trách nhiệm.
- Cập nhật mạng để tăng log xác suất của các nước có lợi nhuận dương, giảm với lợi nhuận âm:
  loss = -trung bình(log p(nước đã đi) * lợi nhuận).

Cách dùng (trong nb4):
    model = nets.build_pg_net()
    level5 = nets.load_model("files/policy_conn.keras")
    report = train_two_stages(model, level5, stage1_batches=300, stage2_batches=300)
    move = pg_move(env, model)
"""

import json
import os
import random
import time

import numpy as np
import tensorflow as tf
import keras

from utils import nets
from utils.conn_simple_env import conn
from utils.policy_net import policy_move

# Xác suất các cột ở nước gần nhất (giải đấu đọc biến này để ghi log)
LAST_INFO = {}


def pg_move(env, model, sample=False):
    """Tác tử cấp 6. Nhận env, trả về cột 1-7. Không làm thay đổi env.

    Khi đấu thật thì đi cột xác suất cao nhất; khi huấn luyện thì rút mẫu (sample=True).
    """
    global LAST_INFO
    probs = nets.move_probabilities(env, model)
    LAST_INFO = {"probs": probs}
    if sample:
        moves = list(probs.keys())
        weights = list(probs.values())
        return random.choices(moves, weights=weights)[0]
    best_move = None
    for move, p in probs.items():
        if best_move is None or p > probs[best_move]:
            best_move = move
    return best_move


def legal_mask(env):
    """Mặt nạ 7 ô: 1 ở cột còn đi được, 0 ở cột đầy."""
    mask = np.zeros(7, dtype=np.float32)
    for col in env.validinputs:
        mask[col - 1] = 1.0
    return mask


def play_training_game(model, opponent, pg_color):
    """Chơi một ván để học: mạng cầm màu pg_color, đối thủ là hàm opponent(env) -> cột.

    Chỉ ghi lại các nước của mạng: đầu vào (phẳng và tích chập), cột đã rút, mặt nạ cột hợp lệ.
    Trả về (danh sách nước đã ghi, kết quả theo góc của mạng: +1 / -1 / 0).
    """
    env = conn()
    steps = []
    while not env.done:
        if env.turn == pg_color:
            move = pg_move(env, model, sample=True)
            steps.append({
                "flat": nets.encode_flat(env)[0],
                "conv": nets.encode_conv(env)[0],
                "action": move - 1,
                "mask": legal_mask(env),
            })
            env.step(move)
        else:
            env.step(opponent(env))
    if env.reward == 0:
        outcome = 0.0
    elif (env.reward == 1) == (pg_color == "red"):
        outcome = 1.0
    else:
        outcome = -1.0
    return steps, outcome


def discounted_returns(n_moves, outcome, gamma):
    """Lợi nhuận của từng nước: outcome * gamma^(số nước còn lại sau nó).

    Giống discount_rs của sách khi phần thưởng chỉ có ở cuối ván.
    Ví dụ 3 nước, thắng, gamma 0.95: [0.9025, 0.95, 1.0].
    """
    returns = []
    for t in range(n_moves):
        returns.append(outcome * gamma ** (n_moves - 1 - t))
    return returns


def reinforce_update(model, optimizer, steps, returns):
    """Một bước cập nhật REINFORCE trên cả lô nước đi. Trả về giá trị loss."""
    flats = np.array([s["flat"] for s in steps], dtype=np.float32)
    convs = np.array([s["conv"] for s in steps], dtype=np.float32)
    actions = np.array([s["action"] for s in steps], dtype=np.int32)
    masks = np.array([s["mask"] for s in steps], dtype=np.float32)
    rets = np.array(returns, dtype=np.float32)

    with tf.GradientTape() as tape:
        probs = model([flats, convs], training=True)
        # bỏ cột đầy rồi chuẩn hóa lại, đúng như phân phối đã dùng để rút nước
        masked = probs * masks
        masked = masked / tf.reduce_sum(masked, axis=1, keepdims=True)
        chosen = tf.gather(masked, actions, axis=1, batch_dims=1)
        log_probs = tf.math.log(chosen + 1e-8)
        loss = -tf.reduce_mean(log_probs * rets)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)


def evaluate_against(model, opponent, n_games=20, seed=0):
    """Điểm số của mạng (đi cột xác suất cao nhất) khi đấu đối thủ cố định, đổi màu mỗi ván.

    Điểm = (thắng + 0.5 hòa) / số ván, dùng làm đường cong học.
    """
    from utils.arena import play_game

    def agent(env):
        return pg_move(env, model)

    points = 0.0
    for g in range(n_games):
        if g % 2 == 0:
            result = play_game(agent, opponent, seed=seed + g)
            mine = "red"
        else:
            result = play_game(opponent, agent, seed=seed + g)
            mine = "yellow"
        if result["winner"] == mine:
            points += 1.0
        elif result["winner"] == "draw":
            points += 0.5
    return points / n_games


def train_stage(model, make_opponent, n_batches, games_per_batch=10, gamma=0.95, optimizer=None,
                evaluators=None, eval_every=25, eval_games=20, checkpoint_dir=None,
                checkpoint_every=50, checkpoint_paths=None, stage_name="stage", verbose=True):
    """Huấn luyện một giai đoạn.

    make_opponent(): mỗi ván gọi một lần để lấy hàm đối thủ (giai đoạn 2 rút ngẫu nhiên một checkpoint).
    Mạng đổi màu mỗi ván để học cả cầm đỏ lẫn cầm vàng.
    evaluators: dict {tên: hàm đối thủ} để đo điểm định kỳ, ví dụ {"cap2": rules.conn_think3}.
    Trả về dict đường cong học của giai đoạn.
    """
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=0.00025)
    if checkpoint_paths is None:
        checkpoint_paths = []
    curve = {"batch": [], "mean_reward": [], "loss": [], "eval": []}
    start = time.time()

    for b in range(1, n_batches + 1):
        all_steps = []
        all_returns = []
        rewards = []
        for g in range(games_per_batch):
            pg_color = "red" if g % 2 == 0 else "yellow"
            steps, outcome = play_training_game(model, make_opponent(), pg_color)
            all_steps += steps
            all_returns += discounted_returns(len(steps), outcome, gamma)
            rewards.append(outcome)
        loss = reinforce_update(model, optimizer, all_steps, all_returns)
        curve["batch"].append(b)
        curve["mean_reward"].append(float(np.mean(rewards)))
        curve["loss"].append(loss)

        if evaluators and b % eval_every == 0:
            scores = {}
            for name, opponent in evaluators.items():
                scores[name] = evaluate_against(model, opponent, n_games=eval_games, seed=1000 + b)
            curve["eval"].append({"batch": b, "scores": scores})
            if verbose:
                print(f"{stage_name} lô {b}: thưởng TB 25 lô gần nhất "
                      f"{np.mean(curve['mean_reward'][-25:]):+.3f}, điểm {scores}")

        if checkpoint_dir and b % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"{stage_name}_lo{b:04d}.keras")
            model.save(path)
            checkpoint_paths.append(path)

    curve["seconds"] = round(time.time() - start, 1)
    curve["checkpoints"] = list(checkpoint_paths)
    return curve


def train_two_stages(model, level5_model, stage1_batches=300, stage2_batches=300, games_per_batch=10,
                     gamma=0.95, learning_rate=0.00025, pool_size=5, evaluators=None, eval_every=25,
                     checkpoint_dir="files/pg_checkpoints", checkpoint_every=50,
                     final_path="files/pg_conn.keras", verbose=True):
    """Huấn luyện cấp 6 qua hai giai đoạn rồi lưu mô hình cuối và báo cáo.

    Giai đoạn 1: đối thủ là mô hình cấp 5 (rút mẫu để ván đa dạng), giống sách.
    Giai đoạn 2: tự chơi; mỗi ván rút ngẫu nhiên một trong pool_size checkpoint gần nhất làm đối thủ.
    Đấu với bản cũ của chính mình tránh việc chỉ học cách thắng đúng một đối thủ.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    checkpoint_paths = []

    def level5_opponent():
        return lambda env: policy_move(env, level5_model, sample=True)

    stage1 = train_stage(model, level5_opponent, stage1_batches, games_per_batch, gamma, optimizer,
                         evaluators, eval_every, checkpoint_dir=checkpoint_dir,
                         checkpoint_every=checkpoint_every, checkpoint_paths=checkpoint_paths,
                         stage_name="gd1", verbose=verbose)

    # chưa có checkpoint nào (giai đoạn 1 quá ngắn) thì lưu bản hiện tại làm đối thủ đầu tiên
    if not checkpoint_paths:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "gd1_cuoi.keras")
        model.save(path)
        checkpoint_paths.append(path)

    loaded = {}

    def past_self_opponent():
        path = random.choice(checkpoint_paths[-pool_size:])
        if path not in loaded:
            loaded[path] = nets.load_model(path)
        old = loaded[path]
        return lambda env: pg_move(env, old, sample=True)

    stage2 = train_stage(model, past_self_opponent, stage2_batches, games_per_batch, gamma, optimizer,
                         evaluators, eval_every, checkpoint_dir=checkpoint_dir,
                         checkpoint_every=checkpoint_every, checkpoint_paths=checkpoint_paths,
                         stage_name="gd2", verbose=verbose)

    report = {
        "games_per_batch": games_per_batch, "gamma": gamma, "learning_rate": learning_rate,
        "pool_size": pool_size, "stage1": stage1, "stage2": stage2,
    }
    nets.save_model(model, final_path, report)
    return report


def save_curve(report, path):
    """Ghi riêng đường cong học ra JSON để notebook vẽ hình."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

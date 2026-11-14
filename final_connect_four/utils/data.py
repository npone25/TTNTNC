"""Sinh ván chuyên gia và tạo tập dữ liệu cho cấp 5 (mạng policy) và mạng value.

Kế thừa: cách sinh ván ở notebook ch11 mục 1.2 (50% Minimax độ sâu 3, 50% MCTS), sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: giảm số ván cho vừa thời gian; dùng engine và tác tử của nhóm (MCTS cây cấp 4 thay MCTS phẳng của sách).
Thêm: chia train/val/test theo ván; loại thế cờ trùng và loại thế ở val/test đã có trong train; lưu seed, số ván, thời gian, cấu hình máy; lưu dần để không mất dữ liệu khi bị ngắt.
Phụ trách: Nguyễn Trọng Phúc.

Quy trình (chạy trong notebook nb3):
    from utils import data
    # mỗi máy một seed khác nhau, chạy qua đêm
    data.generate_games(1000, seed=1, out_path="files/games_phuc.p")
    games = data.merge_game_files(["files/games_phuc.p", "files/games_huong.p", "files/games_thienphuc.p"])
    policy_set = data.make_dataset(games, kind="policy", seed=0)
    data.save_dataset(policy_set, "files/policy_data.npz")

Mã hóa bàn cờ: luôn nhìn từ phía NGƯỜI SẮP ĐI (quân của mình = 1, quân đối thủ = -1),
giống sách (sách nhân -1 với bàn cờ khi đến lượt vàng). Hình dạng (7, 6, 1).
"""

import json
import os
import pickle
import platform
import random
import time

import numpy as np

from utils.conn_simple_env import conn


def expert_move(env, minimax_depth=3, mcts_simulations=1000):
    """Nước đi "chuyên gia" giống notebook ch11 của sách.

    Cột 4 còn trống thì đi cột 4; còn một nước thì đi nước đó;
    còn lại 50% dùng Minimax (cấp 3), 50% dùng MCTS (cấp 4).
    """
    from utils.alphabeta import MiniMax_conn
    from utils.mcts_uct import mcts_uct
    if len(env.occupied[3]) == 0:
        return 4
    if len(env.validinputs) == 1:
        return env.validinputs[0]
    if random.random() <= 0.5:
        return MiniMax_conn(env, depth=minimax_depth)
    return mcts_uct(env, num_simulations=mcts_simulations)


def play_expert_game(minimax_depth=3, mcts_simulations=1000):
    """Chơi một ván chuyên gia tự đấu.

    Trả về (history, reward): history là danh sách (state trước khi đi, cột đã đi, người đi).
    """
    env = conn()
    history = []
    while not env.done:
        move = expert_move(env, minimax_depth, mcts_simulations)
        history.append((env.state.copy(), int(move), env.turn))
        env.step(move)
    return history, env.reward


def machine_info():
    """Cấu hình máy, để ghi vào meta (đề yêu cầu khai báo tài nguyên)."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def save_games(games, out_path, meta):
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(games, f)
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def generate_games(n_games, seed, out_path, minimax_depth=3, mcts_simulations=1000,
                   save_every=50, verbose=True):
    """Sinh n_games ván chuyên gia với seed cố định và lưu ra out_path (pickle).

    Cứ save_every ván lại lưu một lần, nên bị ngắt giữa chừng vẫn giữ được phần đã sinh.
    Mỗi ván là dict: game_id, reward, history. Kèm file out_path + ".json" ghi thông tin chạy.
    """
    random.seed(seed)
    np.random.seed(seed)
    games = []
    start = time.time()
    meta = {
        "seed": seed, "n_games_target": n_games, "minimax_depth": minimax_depth,
        "mcts_simulations": mcts_simulations, "machine": machine_info(),
    }
    for i in range(n_games):
        history, reward = play_expert_game(minimax_depth, mcts_simulations)
        games.append({"game_id": f"s{seed}_g{i}", "reward": reward, "history": history})
        if (i + 1) % save_every == 0 or i + 1 == n_games:
            meta["n_games_done"] = len(games)
            meta["seconds"] = round(time.time() - start, 1)
            save_games(games, out_path, meta)
            if verbose:
                print(f"đã sinh {len(games)}/{n_games} ván, {meta['seconds']} giây")
    return games


def load_games(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def merge_game_files(paths):
    """Gộp các file ván từ nhiều máy (game_id có seed nên không trùng)."""
    games = []
    for path in paths:
        games.extend(load_games(path))
    return games


def to_player_view(state, turn):
    """Bàn cờ nhìn từ phía người sắp đi: đỏ giữ nguyên, vàng nhân -1 (giống sách)."""
    if turn == "red":
        board = state
    else:
        board = -state
    return np.array(board, dtype=np.float32).reshape(7, 6, 1)


def policy_examples(games):
    """Mẫu cho mạng policy, giống sách: chỉ lấy nước đi của bên THẮNG.

    Trả về (X, y, game_ids): X (N, 7, 6, 1), y là chỉ số cột 0-6.
    """
    X = []
    y = []
    ids = []
    for game in games:
        reward = game["reward"]
        if reward == 0:
            continue
        winner = "red" if reward == 1 else "yellow"
        for state, move, turn in game["history"]:
            if turn == winner:
                X.append(to_player_view(state, turn))
                y.append(move - 1)
                ids.append(game["game_id"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), ids


def value_examples(games):
    """Mẫu cho mạng value: mọi thế cờ, nhãn theo góc NGƯỜI SẮP ĐI ở thế đó.

    Nhãn: 0 = hòa, 1 = người sắp đi thắng, 2 = người sắp đi thua.
    """
    X = []
    y = []
    ids = []
    for game in games:
        reward = game["reward"]
        for state, move, turn in game["history"]:
            if reward == 0:
                label = 0
            elif (reward == 1 and turn == "red") or (reward == -1 and turn == "yellow"):
                label = 1
            else:
                label = 2
            X.append(to_player_view(state, turn))
            y.append(label)
            ids.append(game["game_id"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), ids


def split_by_game(game_ids, ratios=(0.8, 0.1, 0.1), seed=0):
    """Chia theo VÁN (không theo thế cờ) để các thế cùng một ván không rơi vào hai tập.

    Trả về dict {"train": chỉ số, "val": chỉ số, "test": chỉ số}.
    """
    unique_ids = sorted(set(game_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n_train = int(len(unique_ids) * ratios[0])
    n_val = int(len(unique_ids) * ratios[1])
    group = {}
    for k, game_id in enumerate(unique_ids):
        if k < n_train:
            group[game_id] = "train"
        elif k < n_train + n_val:
            group[game_id] = "val"
        else:
            group[game_id] = "test"
    split = {"train": [], "val": [], "test": []}
    for index, game_id in enumerate(game_ids):
        split[group[game_id]].append(index)
    for name in split:
        split[name] = np.array(split[name], dtype=np.int64)
    return split


def remove_duplicates(X, y):
    """Bỏ các cặp (thế cờ, nhãn) trùng nhau, giữ lần xuất hiện đầu tiên."""
    seen = set()
    keep = []
    for i in range(len(X)):
        key = (X[i].tobytes(), int(y[i]))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return X[keep], y[keep]


def remove_overlap(X_train, X_other, y_other):
    """Bỏ khỏi val/test những thế cờ đã có trong train (tránh rò rỉ dữ liệu)."""
    train_keys = set()
    for board in X_train:
        train_keys.add(board.tobytes())
    keep = []
    for i in range(len(X_other)):
        if X_other[i].tobytes() not in train_keys:
            keep.append(i)
    return X_other[keep], y_other[keep]


def make_dataset(games, kind="policy", seed=0, ratios=(0.8, 0.1, 0.1)):
    """Tạo tập train/val/test đã chia theo ván và đã loại trùng.

    kind = "policy" (cấp 5) hoặc "value" (mạng value cho cấp 7).
    """
    if kind == "policy":
        X, y, ids = policy_examples(games)
    elif kind == "value":
        X, y, ids = value_examples(games)
    else:
        raise ValueError("kind phải là 'policy' hoặc 'value'")
    split = split_by_game(ids, ratios, seed)
    X_train, y_train = remove_duplicates(X[split["train"]], y[split["train"]])
    X_val, y_val = remove_duplicates(X[split["val"]], y[split["val"]])
    X_test, y_test = remove_duplicates(X[split["test"]], y[split["test"]])
    X_val, y_val = remove_overlap(X_train, X_val, y_val)
    X_test, y_test = remove_overlap(X_train, X_test, y_test)
    return {
        "kind": kind, "seed": seed, "n_games": len(set(ids)),
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def describe_dataset(dataset):
    """Một dòng chữ mô tả số mẫu từng tập."""
    return (f"{dataset['kind']}: {dataset['n_games']} ván | train {len(dataset['y_train'])}, "
            f"val {len(dataset['y_val'])}, test {len(dataset['y_test'])} mẫu")


def save_dataset(dataset, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in dataset.items() if isinstance(v, np.ndarray)},
                        kind=dataset["kind"], seed=dataset["seed"], n_games=dataset["n_games"])


def load_dataset(path):
    loaded = np.load(path)
    dataset = {}
    for key in loaded.files:
        dataset[key] = loaded[key]
    dataset["kind"] = str(dataset["kind"])
    dataset["seed"] = int(dataset["seed"])
    dataset["n_games"] = int(dataset["n_games"])
    return dataset

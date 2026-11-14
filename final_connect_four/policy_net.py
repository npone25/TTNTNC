"""Cấp 5: tác tử chơi bằng mạng policy học có giám sát.

Kế thừa: cách dùng mạng policy để chọn nước ở notebook ch11 và ch15 (hàm opponent), sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: chỉ chọn trong các cột hợp lệ; mặc định đi cột có xác suất lớn nhất (sách luôn rút mẫu); có seed.
Thêm: hàm tác tử nhận env trả về cột; hàm huấn luyện gọn cho nb3, dùng dữ liệu của data.py và hàm mạng trong nets.py.
Phụ trách: Trần Quỳnh Hương.

Cách dùng:
    model = nets.load_model("files/policy_conn.keras")
    move = policy_move(env, model)                 # đi cột xác suất cao nhất
    move = policy_move(env, model, sample=True)    # rút mẫu theo xác suất (đa dạng hơn, dùng làm đối thủ khi huấn luyện cấp 6)
"""

import random

from utils import nets
from utils import data

# Xác suất các cột ở nước gần nhất (giải đấu đọc biến này để ghi log)
LAST_INFO = {}


def policy_move(env, model, sample=False):
    """Tác tử cấp 5. Nhận env, trả về cột 1-7. Không làm thay đổi env.

    Mạng cho ra 7 xác suất; nets.move_probabilities đã bỏ các cột đầy và chuẩn hóa lại
    nên cột trả về luôn hợp lệ.
    """
    global LAST_INFO
    probs = nets.move_probabilities(env, model)
    LAST_INFO = {"probs": probs}
    if sample:
        moves = list(probs.keys())
        weights = list(probs.values())
        return random.choices(moves, weights=weights)[0]
    # nhiều cột cùng xác suất lớn nhất (hiếm) thì lấy cột đứng trước
    best_move = None
    for move, p in probs.items():
        if best_move is None or p > probs[best_move]:
            best_move = move
    return best_move


def train_policy(games, path="files/policy_conn.keras", epochs=30, batch_size=128,
                 learning_rate=0.001, seed=0, verbose=1):
    """Huấn luyện mạng cấp 5 từ danh sách ván chuyên gia (kết quả của data.generate_games).

    Các bước: tạo mẫu (thế cờ, nước của bên thắng), chia train/val/test theo ván,
    huấn luyện, đánh giá trên val và test, lưu mô hình kèm báo cáo.
    Trả về (model, report).
    """
    dataset = data.make_dataset(games, kind="policy", seed=seed)
    model = nets.build_policy_net(learning_rate=learning_rate)
    report = nets.train_model(model, dataset, epochs=epochs, batch_size=batch_size, verbose=verbose)
    report["learning_rate"] = learning_rate
    nets.save_model(model, path, report)
    return model, report

"""Lõi tìm kiếm cây dùng chung cho cấp 4 (mcts_uct.py) và cấp 7 (alphago_tree.py).

Kế thừa: tên và vai trò bốn bước select, expand, simulate, backpropagate theo utils/ch08util.py (sách bản gốc là MCTS phẳng, chỉ thống kê ở gốc).
Sửa: không áp dụng.
Thêm: tự viết cây thật: nút cây, vòng lặp bốn bước, chế độ cố định số mô phỏng và cố định thời gian, seed, ghi số mô phỏng; cách chọn nhánh và cách đánh giá lá được truyền vào từ cấp 4 và cấp 7.
Phụ trách: Nguyễn Trọng Phúc.

Một lần mô phỏng gồm bốn bước, lặp lại nhiều lần trong run_search():
  1. select:        đi theo cây từ gốc xuống tới một nút lá (nút chưa có con).
  2. simulate:      chấm điểm thế cờ ở lá (cấp 4 chơi ngẫu nhiên tới hết ván, cấp 7 dùng mạng).
  3. expand:        tạo nút con cho lá, để lần sau đi sâu thêm được một tầng.
  4. backpropagate: cộng điểm vừa có vào mọi nút trên đường từ lá về gốc.
Hết số lần mô phỏng thì chọn cột ở gốc được thăm nhiều nhất.

Quy ước góc nhìn (đọc kỹ trước khi sửa file này):
- Mỗi nút ứng với thế cờ SAU KHI đi nước node.move, và là "nước của" người đã đi nước đó.
- value_sum của nút ghi điểm theo góc nhìn của chính người đó: dương là tốt cho người
  đã đi nước này. Người này cũng là người đứng ở nút cha và chọn nút này, nên ở nút cha
  chỉ cần chọn con có q() lớn nhất, không phải nghĩ điểm này là của bên nào.
- Hai tầng liền nhau thuộc về hai người đối nghịch, nên khi lan truyền ngược thì
  cứ lên một tầng là đổi dấu điểm một lần.

Hai hàm được truyền vào:
- select_child(node): nhận một nút đã mở rộng, trả về nút con được chọn.
- evaluate_leaf(env): nhận thế cờ CHƯA kết thúc, trả về (priors, value):
  priors là dict {cột: xác suất tiên nghiệm} cho các cột hợp lệ;
  value là giá trị trong [-1, 1] theo góc nhìn NGƯỜI SẮP ĐI ở thế đó.
"""

import random
import time


class Node:
    """Một nút của cây tìm kiếm."""

    def __init__(self, parent, move, prior):
        self.parent = parent      # nút cha (None nếu là gốc)
        self.move = move          # cột đã đi để tới nút này (None nếu là gốc)
        self.prior = prior        # mức "đáng thử" do evaluate_leaf đưa ra; cấp 4 chia đều, cấp 7 lấy từ mạng policy
        self.children = {}        # {cột: nút con}; rỗng nghĩa là nút lá
        self.visits = 0           # số lần mô phỏng đã đi qua nút này
        self.value_sum = 0.0      # tổng điểm qua các lần đó, theo góc người đã đi node.move

    def q(self):
        """Điểm trung bình theo góc người đã đi node.move, trong [-1, 1] (0 nếu chưa thăm)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_leaf(self):
        return len(self.children) == 0


def select(root, env, select_child):
    """Bước 1 - Chọn: đi từ gốc xuống theo các nút đã có, tới khi gặp lá hoặc ván kết thúc.

    Hàm này chỉ đi, không tạo nút mới. Ở mỗi tầng, select_child quyết định đi nhánh nào
    (cấp 4 dùng UCT, cấp 7 dùng PUCT). env là bản nháp của thế cờ ở gốc; nước nào đi trên
    cây thì cũng đi trên env để env luôn khớp với nút đang đứng.
    Trả về (danh sách nút trên đường đi, env ở cuối đường đi).
    """
    node = root
    path = [node]
    # còn con thì còn đi xuống được; ván đã kết thúc thì dừng dù nút có con hay không
    while not node.is_leaf() and not env.done:
        node = select_child(node)
        env.step(node.move)
        path.append(node)
    return path, env

def expand(node, priors):
    """Bước 2 - Mở rộng: tạo nút con cho nút lá, mỗi cột còn đi được một con.

    priors là dict {cột: xác suất tiên nghiệm} do simulate trả về (lấy từ evaluate_leaf).
    Cấp 4 cho mọi cột như nhau (1 / số cột), cấp 7 lấy từ mạng policy.
    Con mới tạo chưa được thăm lần nào; các lần mô phỏng sau sẽ chọn tới chúng.
    """
    for move, prior in priors.items():
        node.children[move] = Node(node, move, prior)


def simulate(env, evaluate_leaf):
    """Bước 3 - Chấm điểm thế cờ ở nút lá: tốt hay xấu cho người vừa đi nước dẫn tới lá.

    Trả về (priors, điểm). priors dùng cho bước mở rộng; None nếu ván đã kết thúc.
    - Ván đã kết thúc ở lá: không cần đoán. Chỉ người vừa đi mới có thể vừa thắng,
      nên thắng là +1 cho người đó, hòa là 0.
    - Ván chưa kết thúc: hỏi evaluate_leaf. Hàm này chấm theo góc người SẮP ĐI (đối thủ
      của người vừa đi), nên đổi dấu để ra điểm theo góc người vừa đi.
    """
    if env.done:
        if env.reward != 0:
            return None, 1.0
        return None, 0.0
    priors, value_for_player_to_move = evaluate_leaf(env)
    return priors, -value_for_player_to_move


def backpropagate(path, value):
    """Bước 4 - Lan truyền ngược: ghi kết quả vào mọi nút trên đường đi.

    Đi từ lá lên gốc (reversed chỉ duyệt ngược, không sửa path). value lúc đầu đúng
    theo góc người đi nút lá; lên tầng trên là nước của đối thủ nên đổi dấu trước khi ghi.
    """
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


def run_search(env, select_child, evaluate_leaf, num_simulations=None, time_limit=None):
    """Chạy tìm kiếm cây từ thế cờ env (không làm thay đổi env).

    Chọn đúng một trong hai chế độ:
    - num_simulations: số lần mô phỏng cố định (kết quả tái lập được với cùng seed);
    - time_limit: số giây suy nghĩ cho nước này (luôn chạy ít nhất 1 lần).
    Trả về (root, info) với info = {"simulations": ..., "seconds": ...}.
    """
    if (num_simulations is None) == (time_limit is None):
        raise ValueError("Phải chọn đúng một trong num_simulations hoặc time_limit.")
    if env.done:
        raise ValueError("Ván đã kết thúc, không thể tìm kiếm.")

    root = Node(None, None, 1.0)
    start = time.perf_counter()
    count = 0
    while True:
        if num_simulations is not None and count >= num_simulations:
            break
        # count > 0: dù hết giờ ngay vẫn chạy đủ một lần để gốc có con mà chọn
        if time_limit is not None and count > 0 and time.perf_counter() - start >= time_limit:
            break
        # mỗi lần mô phỏng làm trên một bản nháp mới, ván thật không bị đụng tới
        sim_env = env.clone()
        path, sim_env = select(root, sim_env, select_child)
        leaf = path[-1]
        priors, value = simulate(sim_env, evaluate_leaf)
        if priors is not None:
            expand(leaf, priors)
        backpropagate(path, value)
        count += 1

    info = {"simulations": count, "seconds": time.perf_counter() - start}
    return root, info


def most_visited_move(root):
    """Nước đi thật: cột ở gốc được thăm nhiều nhất; bằng nhau thì chọn ngẫu nhiên.

    Dùng số lượt thăm chứ không dùng điểm trung bình: một cột mới thử 2 lần mà may thắng
    cả 2 có điểm 1.0 nhưng chưa đáng tin; cột được thử nhiều nhất là cột đã kiểm chứng kỹ nhất.
    """
    best_visits = -1
    best_moves = []
    for move, child in root.children.items():
        if child.visits > best_visits:
            best_visits = child.visits
            best_moves = [move]
        elif child.visits == best_visits:
            best_moves.append(move)
    return random.choice(best_moves)


def root_statistics(root):
    """Số lượt thăm và giá trị trung bình của từng cột ở gốc (để ghi log, phân tích)."""
    visits = {}
    values = {}
    for move, child in root.children.items():
        visits[move] = child.visits
        values[move] = round(child.q(), 4)
    return visits, values

"""Cấp 3: Minimax với cắt tỉa Alpha-Beta, giới hạn độ sâu.

Kế thừa: max_payoff_conn và MiniMax_conn trong utils/ch06util.py, sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: viết lại theo dạng Negamax (một hàm cho cả hai bên); dùng step/undo trên một bản sao thay deepcopy ở mỗi nút.
Thêm: hàm lượng giá khi chạm độ sâu (sách trả về 0); cờ prune bật/tắt cắt tỉa; bộ đếm nút; sắp xếp nước đi từ cột giữa; chế độ giới hạn thời gian (đào sâu dần) cho RQ1.
Phụ trách: Phạm Thiên Phúc.

Cách dùng:
    MiniMax_conn(env, depth=3)              # nhìn nước của mình + 3 nước sau đó (giống sách)
    MiniMax_conn(env, prune=False)          # tắt cắt tỉa, cho ablation
    MiniMax_conn(env, time_limit=0.5)       # đào sâu dần trong 0,5 giây (RQ1)
Sau mỗi lần gọi, LAST_INFO ghi số nút đã duyệt, độ sâu đạt được và thời gian.

Ý tưởng Negamax: điểm của một thế cờ luôn tính theo người SẮP ĐI ở thế đó.
Điểm của mình = -(điểm tốt nhất của đối thủ sau nước của mình), nên chỉ cần một hàm
cho cả hai bên, không phải tách nhánh max và min như sách.
"""

import random
import time

# Thứ tự xét nước: từ giữa ra hai biên. Nước giữa thường tốt hơn, xét trước thì
# alpha tăng sớm và cắt tỉa được nhiều nhánh hơn.
MOVE_ORDER = [4, 3, 5, 2, 6, 1, 7]

# Điểm của thế đã phân thắng thua. Hàm lượng giá luôn nằm trong (-1, 1)
# nên một thế thắng thật luôn được ưu tiên hơn một thế "trông có vẻ tốt".
WIN_SCORE = 1.0

# Trọng số cho mỗi cửa sổ 4 ô (chỉ tính cửa sổ chưa có quân đối thủ)
WEIGHT_THREE = 5      # 3 quân + 1 ô trống: sắp thành 4
WEIGHT_TWO = 2        # 2 quân + 2 ô trống
WEIGHT_CENTER = 3     # mỗi quân ở cột 4 (cột giữa nằm trong nhiều cửa sổ nhất)
SCORE_SCALE = 100.0   # chia điểm thô cho số này rồi kẹp vào [-0.9, 0.9]

# Thông tin của nước gần nhất (giải đấu đọc biến này để ghi log)
LAST_INFO = {}


def all_windows():
    """Liệt kê 69 cửa sổ 4 ô liên tiếp trên bàn 7x6, mỗi cửa sổ là 4 cặp (cột, hàng)."""
    windows = []
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]   # ngang, dọc, chéo /, chéo \
    for col in range(7):
        for row in range(6):
            for d_col, d_row in directions:
                end_col = col + 3 * d_col
                end_row = row + 3 * d_row
                if 0 <= end_col <= 6 and 0 <= end_row <= 5:
                    cells = []
                    for k in range(4):
                        cells.append((col + k * d_col, row + k * d_row))
                    windows.append(cells)
    return windows


# Tính sẵn một lần, hàm lượng giá dùng lại ở mọi nút lá
WINDOWS = all_windows()


def evaluate(env):
    """Hàm lượng giá khi chạm độ sâu: thế cờ này tốt hay xấu cho người SẮP ĐI.

    Cộng điểm cho các cửa sổ còn "mở" của mình (chưa có quân đối thủ), trừ điểm
    cho cửa sổ mở của đối thủ, cộng thêm quân ở cột giữa. Kết quả nằm trong [-0.9, 0.9].
    """
    me = 1 if env.turn == "red" else -1
    state = env.state
    score = 0
    for cells in WINDOWS:
        mine = 0
        theirs = 0
        for col, row in cells:
            value = state[col][row]
            if value == me:
                mine += 1
            elif value == -me:
                theirs += 1
        # cửa sổ có quân của cả hai bên thì không ai dùng được nữa
        if mine > 0 and theirs > 0:
            continue
        if mine == 3:
            score += WEIGHT_THREE
        elif mine == 2:
            score += WEIGHT_TWO
        elif theirs == 3:
            score -= WEIGHT_THREE
        elif theirs == 2:
            score -= WEIGHT_TWO
    for row in range(6):
        if state[3][row] == me:
            score += WEIGHT_CENTER
        elif state[3][row] == -me:
            score -= WEIGHT_CENTER
    value = score / SCORE_SCALE
    return max(-0.9, min(0.9, value))


def ordered_moves(env):
    moves = []
    for col in MOVE_ORDER:
        if col in env.validinputs:
            moves.append(col)
    return moves


class SearchTimeout(Exception):
    """Hết giờ giữa chừng khi đang đào sâu dần."""


class Search:
    """Một lượt tìm kiếm: giữ bộ đếm nút và hạn giờ."""

    def __init__(self, prune=True, deadline=None):
        self.prune = prune
        self.deadline = deadline
        self.nodes = 0

    def negamax(self, env, depth, alpha, beta):
        """Điểm của thế env theo góc người sắp đi, nhìn trước depth nước.

        alpha: điểm tốt nhất mình đã chắc chắn có được ở nhánh khác.
        beta:  nếu điểm ở đây vượt beta thì đối thủ sẽ không bao giờ để mình tới thế này.
        """
        self.nodes += 1
        if self.deadline is not None and self.nodes % 256 == 0 and time.perf_counter() > self.deadline:
            raise SearchTimeout()

        # ván đã kết thúc sau nước của đối thủ: đối thủ vừa thắng (xấu cho mình) hoặc hòa
        if env.done:
            if env.reward != 0:
                return -WIN_SCORE
            return 0.0
        if depth == 0:
            return evaluate(env)

        best = -2.0
        for col in ordered_moves(env):
            env.step(col)
            # điểm của mình là điểm của đối thủ đổi dấu; cửa sổ (alpha, beta) cũng đổi vai
            score = -self.negamax(env, depth - 1, -beta, -alpha)
            env.undo()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            # đối thủ đã có lựa chọn khác tốt hơn cho họ, không cần xét các nước còn lại
            if self.prune and alpha >= beta:
                break
        return best

    def root_scores(self, env, depth):
        """Điểm chính xác của từng nước ở gốc.

        Ở gốc không thu hẹp cửa sổ để điểm các nước so sánh được với nhau
        (thu hẹp thì có nước chỉ nhận được cận trên, dễ hòa điểm giả).
        """
        board = env.clone()
        scores = {}
        for col in ordered_moves(board):
            board.step(col)
            scores[col] = -self.negamax(board, depth, -2.0, 2.0)
            board.undo()
        return scores


def pick_best(scores):
    """Chọn cột điểm cao nhất; nhiều cột bằng điểm thì chọn ngẫu nhiên (có seed) như sách."""
    best = max(scores.values())
    candidates = []
    for col, score in scores.items():
        if score == best:
            candidates.append(col)
    return random.choice(candidates)


def immediate_win(env):
    """Cột thắng ngay nếu có (giống sách: thấy thắng là đi luôn, khỏi tìm kiếm)."""
    for col in ordered_moves(env):
        test = env.clone()
        test.step(col)
        if test.done and test.reward != 0:
            return col
    return None


def MiniMax_conn(env, depth=3, prune=True, time_limit=None, max_depth=20):
    """Tác tử cấp 3. Nhận env, trả về cột 1-7. Không làm thay đổi env.

    depth: số nước nhìn trước SAU nước của mình (depth=3 nghĩa là xét 4 nước, giống sách).
    time_limit: nếu có thì bỏ qua depth, đào sâu dần 0, 1, 2, ... đến khi hết giờ,
    dùng kết quả của độ sâu cuối cùng chạy xong trọn vẹn.
    """
    global LAST_INFO
    start = time.perf_counter()

    win = immediate_win(env)
    if win is not None:
        LAST_INFO = {"nodes": 0, "depth": 0, "seconds": time.perf_counter() - start}
        return win

    if time_limit is None:
        search = Search(prune=prune)
        scores = search.root_scores(env, depth)
        LAST_INFO = {"nodes": search.nodes, "depth": depth, "seconds": time.perf_counter() - start}
        return pick_best(scores)

    # chế độ thời gian: độ sâu 0 luôn chạy xong để chắc chắn có nước trả về
    deadline = start + time_limit
    total_nodes = 0
    scores = Search(prune=prune).root_scores(env, 0)
    reached = 0
    for d in range(1, max_depth + 1):
        search = Search(prune=prune, deadline=deadline)
        try:
            scores = search.root_scores(env, d)
            reached = d
        except SearchTimeout:
            total_nodes += search.nodes
            break
        total_nodes += search.nodes
        # đã thấy thắng chắc thì không cần đào sâu thêm
        if max(scores.values()) >= WIN_SCORE:
            break
        if time.perf_counter() >= deadline:
            break
    LAST_INFO = {"nodes": total_nodes, "depth": reached, "seconds": time.perf_counter() - start}
    return pick_best(scores)

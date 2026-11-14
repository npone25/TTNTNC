"""Cấp 1 (ngẫu nhiên) và cấp 2 (luật heuristic).

Kế thừa: conn_random, conn_think3, to_avoid trong utils/ch03util.py, sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: to_avoid của sách chỉ đúng khi bên đỏ đi (kiểm tra r == -1), nay đúng cho cả hai màu;
     vòng nhìn trước của sách vẫn đi tiếp khi ván đã kết thúc, nay dừng đúng lúc;
     luật "chiếm cột giữa" của sách đứng trước luật thắng ngay nên có lúc bỏ lỡ nước thắng, nay đưa xuống sau luật chặn;
     dùng env.clone() thay deepcopy.
Thêm: ghi lại luật đã dùng ở mỗi nước (LAST_INFO["rule"]) để phân tích; ngẫu nhiên đi qua module random nên có seed.
Phụ trách: Phạm Thiên Phúc.

Thứ tự ưu tiên của cấp 2 (luật đứng trước thắng luật đứng sau, nên hai luật mâu thuẫn
thì luật nào đứng trước sẽ quyết định):
  1. Chỉ còn một cột thì đi cột đó.
  2. Có nước thắng ngay thì đi.
  3. Đối thủ sắp thắng ngay ở một cột thì chặn cột đó.
  4. Cột giữa (cột 4) còn trống thì đi cột 4.
  5. Bỏ các cột "bệ đỡ": đi vào đó thì đối thủ thắng ngay ở ô ngay trên.
  6. Nhìn trước ba nước (mình - đối thủ - mình), chọn nước đầu dẫn tới thắng nhiều nhất.
  7. Không luật nào dùng được thì đi ngẫu nhiên trong các cột an toàn.
"""

import random

# Luật vừa được dùng ở nước gần nhất (giải đấu đọc biến này để ghi log)
LAST_INFO = {}


def conn_random(env):
    """Cấp 1: chọn đều một cột trong các cột còn đi được."""
    return random.choice(env.validinputs)


def other(color):
    if color == "red":
        return "yellow"
    return "red"


def wins_with(env, col, color):
    """Nếu bên color thả quân vào cột col thì có thắng ngay không.

    Thử trên bản sao nên không đụng tới env. Đổi lượt trên bản sao để hỏi được
    cả câu "nếu đối thủ đi cột này thì sao", dù lúc này chưa tới lượt đối thủ.
    """
    if col not in env.validinputs:
        return False
    test = env.clone()
    test.turn = color
    test.step(col)
    return test.done and test.reward != 0


def winning_moves(env, color):
    """Các cột mà bên color thả vào là thắng ngay."""
    moves = []
    for col in env.validinputs:
        if wins_with(env, col, color):
            moves.append(col)
    return moves


def to_avoid(env):
    """Các cột "bệ đỡ": mình đi vào thì đối thủ thắng ngay bằng ô ngay phía trên.

    Sách chỉ kiểm tra trường hợp vàng thắng (r == -1) nên luật này chỉ đúng khi đỏ đi.
    Ở đây hỏi thẳng "người vừa đi có thắng không", nên đúng cho cả hai màu.
    """
    me = env.turn
    avoid = []
    for col in env.validinputs:
        after = env.clone()
        after.step(col)
        if after.done:
            continue
        # cột đầy sau nước của mình thì đối thủ không đi vào đó được nữa
        if wins_with(after, col, other(me)):
            avoid.append(col)
    return avoid


def count_three_step_wins(env, candidates):
    """Nhìn trước ba nước: mình đi m1, đối thủ đi m2, mình đi m3.

    Đếm số chuỗi (m1, m2, m3) mà mình thắng ở m3, gom theo m1.
    Chuỗi nào mà ván đã kết thúc giữa chừng thì dừng ở đó (sách vẫn đi tiếp, gây lỗi).
    """
    me = env.turn
    counts = {}
    for m1 in candidates:
        counts[m1] = 0
        after1 = env.clone()
        after1.step(m1)
        if after1.done:
            continue
        for m2 in after1.legal_actions():
            after2 = after1.clone()
            after2.step(m2)
            if after2.done:
                continue
            for m3 in after2.legal_actions():
                if wins_with(after2, m3, me):
                    counts[m1] += 1
    return counts


def conn_think3(env):
    """Cấp 2: áp dụng lần lượt các luật trong docstring đầu file, luật nào dùng được trước thì quyết định."""
    global LAST_INFO
    me = env.turn
    legal = env.legal_actions()

    if len(legal) == 1:
        LAST_INFO = {"rule": "one_move"}
        return legal[0]

    wins = winning_moves(env, me)
    if wins:
        LAST_INFO = {"rule": "win"}
        return wins[0]

    threats = winning_moves(env, other(me))
    if threats:
        # đối thủ có hai chỗ thắng thì chặn được một chỗ, vẫn chặn còn hơn không
        LAST_INFO = {"rule": "block"}
        return threats[0]

    if len(env.occupied[3]) == 0:
        LAST_INFO = {"rule": "center"}
        return 4

    avoid = to_avoid(env)
    safe = []
    for col in legal:
        if col not in avoid:
            safe.append(col)
    # cột nào cũng là bệ đỡ thì đành đi một cột trong số đó
    if not safe:
        safe = legal

    counts = count_three_step_wins(env, safe)
    best = max(counts.values())
    if best > 0:
        # chọn cột có nhiều chuỗi thắng nhất; bằng nhau thì lấy cột đứng trước
        for col in safe:
            if counts[col] == best:
                LAST_INFO = {"rule": "look_ahead"}
                return col

    LAST_INFO = {"rule": "random_safe"}
    return random.choice(safe)

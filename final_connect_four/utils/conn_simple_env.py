"""Engine Connect Four 7x6 và Game API thống nhất.

Kế thừa: utils/conn_simple_env.py, sách AlphaGo Simplified (Mark Liu), giấy phép MIT. Giữ quy ước: cột 1-7, state (7, 6), lượt 'red'/'yellow', reward theo góc bên đỏ.
Sửa: bỏ import turtle không dùng; step() ném lỗi khi nước đi không hợp lệ (cột đầy, ngoài 1-7, ván đã kết thúc).
Thêm: legal_actions(), is_terminal(), result(), render() dạng chữ, clone() nhẹ thay deepcopy, undo().
Phụ trách: Nguyễn Trọng Phúc.

Quy ước (giữ đúng như sách):
- Cột đánh số 1..7. state là mảng (7, 6): state[cột-1][hàng], hàng 0 là hàng dưới cùng.
- Ô của đỏ = 1, ô của vàng = -1, ô trống = 0. Đỏ luôn đi trước.
- reward = 1 nếu đỏ thắng, -1 nếu vàng thắng, 0 nếu hòa hoặc ván chưa kết thúc.
- Khi ván kết thúc, turn KHÔNG đổi (vẫn là người vừa đi), giống sách.
"""

from random import choice
from numbers import Integral
import numpy as np


class InvalidMoveError(ValueError):
    """Nước đi không hợp lệ: cột ngoài 1-7, cột đã đầy, hoặc ván đã kết thúc."""
    pass


# Hai lớp phụ giữ lại từ sách cho giống giao diện môi trường kiểu Gym:
# action_space.n là số hành động (7 cột), observation_space.shape là kích thước bàn (7, 6).
class action_space:
    def __init__(self, n):
        self.n = n


class observation_space:
    def __init__(self, row, col):
        self.shape = (row, col)


class conn():
    def __init__(self):
        self.action_space = action_space(7)
        self.observation_space = observation_space(7, 6)
        self.info = ""
        # tọa độ tâm 7 cột và 6 hàng trên màn hình turtle, giữ theo sách cho phần vẽ
        self.xs = [-300, -200, -100, 0, 100, 200, 300]
        self.ys = [-250, -150, -50, 50, 150, 250]
        # quân vừa đặt: [cột, hàng, màu], phần vẽ dùng để biết thả quân vào ô nào
        self.game_piece = None
        self.reset()

    # một cột ngẫu nhiên trong các cột còn trống (dùng module random nên đặt seed được)
    def sample(self):
        return choice(self.validinputs)

    def reset(self):
        self.turn = "red"
        # các cột chưa đầy, tức các nước còn đi được
        self.validinputs = [1, 2, 3, 4, 5, 6, 7]
        # occupied[c] là danh sách màu các quân trong cột c, từ dưới lên;
        # độ dài danh sách chính là chiều cao cột, phần kiểm tra thắng đọc màu từ đây
        self.occupied = [list(), list(), list(), list(), list(), list(), list()]
        # state là bàn cờ dạng số (1 đỏ, -1 vàng, 0 trống), mạng nơ-ron và render() đọc từ đây
        self.state = np.zeros((7, 6), dtype=int)
        self.done = False
        self.reward = 0
        # các cột đã đi theo thứ tự, undo() lấy nước cuối từ đây
        self.history = []
        self.game_piece = None
        return self.state

    def _check_move(self, inp):
        if self.done:
            raise InvalidMoveError("Ván đã kết thúc, không thể đi thêm.")
        # True/False cũng được Python coi là số nguyên (True == 1),
        # nên phải chặn riêng, không thì step(True) sẽ thành đi cột 1
        if isinstance(inp, bool) or not isinstance(inp, Integral):
            raise InvalidMoveError(f"Nước đi phải là số nguyên 1-7, nhận được {inp!r}.")
        if inp < 1 or inp > 7:
            raise InvalidMoveError(f"Cột {inp} nằm ngoài 1-7.")
        if len(self.occupied[inp - 1]) >= 6:
            raise InvalidMoveError(f"Cột {inp} đã đầy.")

    # Đi một nước: kiểm tra hợp lệ, đặt quân, rồi xét thắng / hòa / đổi lượt
    def step(self, inp):
        self._check_move(inp)
        # đổi về int thường (tác tử dùng mạng hay trả về kiểu số của numpy)
        inp = int(inp)
        # quân rơi xuống ô trống thấp nhất của cột, tức hàng = chiều cao hiện tại của cột
        self.game_piece = [inp - 1, len(self.occupied[inp - 1]), self.turn]
        # (self.turn == "red") là True/False, nhân 2 trừ 1 ra 1 cho đỏ, -1 cho vàng
        self.state[inp - 1][len(self.occupied[inp - 1])] = 2 * (self.turn == "red") - 1
        self.occupied[inp - 1].append(self.turn)
        self.history.append(inp)

        # cột vừa đủ 6 quân thì không đi vào được nữa
        if len(self.occupied[inp - 1]) == 6 and inp in self.validinputs:
            self.validinputs.remove(inp)
        # chỉ người vừa đi mới có thể vừa thắng, và chỉ cần xét quanh quân vừa đặt
        if self.win_game(inp) == True:
            self.done = True
            # reward luôn theo góc bên đỏ: đỏ thắng 1, vàng thắng -1
            self.reward = 2 * (self.turn == "red") - 1
            self.validinputs = []
        # không ai thắng mà bàn đã kín 42 ô thì hòa
        elif len(self.validinputs) == 0:
            self.done = True
            self.reward = 0
        else:
            # ván chưa xong thì đổi lượt; ván xong thì giữ nguyên lượt như sách
            if self.turn == "red":
                self.turn = "yellow"
            else:
                self.turn = "red"
        return self.state, self.reward, self.done, self.info

    # Bốn hàm dưới xét đường 4 quân đi qua ô (x, y) vừa đặt, mỗi hàm một hướng.
    # Mỗi hàm trượt một khung 4 ô qua các vị trí có chứa ô (x, y) (dif = -3..0).
    # occupied lưu quân theo từng cột nên ô trống (cột chưa cao tới) hay ô ngoài bàn
    # đều gây IndexError, coi như không thắng. Riêng chỉ số âm thì Python không báo lỗi
    # mà lấy ngược từ cuối list, nên phải kiểm tra thêm chỉ số >= 0.

    # hàng ngang
    def horizontal4(self, x, y):
        win = False
        for dif in (-3, -2, -1, 0):
            try:
                if self.occupied[x+dif][y] == self.turn\
                and self.occupied[x+dif+1][y] == self.turn\
                and self.occupied[x+dif+2][y] == self.turn\
                and self.occupied[x+dif+3][y] == self.turn\
                and  x+dif >= 0:
                    win = True
            except IndexError:
                pass
        return win

    # hàng dọc: chỉ cần nhìn xuống dưới, vì quân vừa đặt luôn nằm trên cùng của cột
    def vertical4(self, x, y):
        win = False
        try:
            if self.occupied[x][y] == self.turn\
            and self.occupied[x][y-1] == self.turn\
            and self.occupied[x][y-2] == self.turn\
            and self.occupied[x][y-3] == self.turn\
            and y-3 >= 0:
                win = True
        except IndexError:
            pass
        return win

    # chéo / (lên sang phải)
    def forward4(self, x, y):
        win = False
        for dif in (-3, -2, -1, 0):
            try:
                if self.occupied[x+dif][y+dif] == self.turn\
                and self.occupied[x+dif+1][y+dif+1] == self.turn\
                and self.occupied[x+dif+2][y+dif+2] == self.turn\
                and self.occupied[x+dif+3][y+dif+3] == self.turn\
                and x+dif >=  0 and y+dif >= 0:
                    win = True
            except IndexError:
                pass
        return win

    # chéo \ (xuống sang phải)
    def back4(self, x, y):
        win = False
        for dif in (-3, -2, -1, 0):
            try:
                if self.occupied[x+dif][y-dif] == self.turn\
                and self.occupied[x+dif+1][y-dif-1] == self.turn\
                and self.occupied[x+dif+2][y-dif-2] == self.turn\
                and self.occupied[x+dif+3][y-dif-3] == self.turn\
                and x+dif >=  0 and y-dif-3 >= 0:
                    win = True
            except IndexError:
                pass
        return win

    # Người vừa đi (self.turn) có thắng nhờ quân vừa đặt ở cột inp không
    def win_game(self, inp):
        win = False
        x = inp - 1
        # y là hàng của quân vừa đặt, tức quân trên cùng của cột
        y = len(self.occupied[inp - 1]) - 1
        if self.vertical4(x, y) == True:
            win = True
        if self.horizontal4(x, y) == True:
            win = True
        if self.forward4(x, y) == True:
            win = True
        if self.back4(x, y) == True:
            win = True
        return win

    def legal_actions(self):
        return list(self.validinputs)

    def is_terminal(self):
        return self.done

    def result(self):
        """Kết quả ván theo góc bên đỏ: 1 đỏ thắng, -1 vàng thắng, 0 hòa."""
        if not self.done:
            return None
        return self.reward

    def winner(self):
        """Tên người thắng: "red", "yellow", "draw", hoặc None nếu chưa kết thúc."""
        if not self.done:
            return None
        if self.reward == 1:
            return "red"
        if self.reward == -1:
            return "yellow"
        return "draw"

    def render(self):
        """Trả về chuỗi vẽ bàn cờ: X là đỏ, O là vàng, . là ô trống.

        Hàng trên cùng in trước, dòng cuối là số cột 1-7.
        """
        lines = []
        for row in range(5, -1, -1):
            cells = []
            for col in range(7):
                value = self.state[col][row]
                if value == 1:
                    cells.append("X")
                elif value == -1:
                    cells.append("O")
                else:
                    cells.append(".")
            lines.append(" ".join(cells))
        lines.append("1 2 3 4 5 6 7")
        return "\n".join(lines)

    def clone(self):
        """Bản sao để tác tử đi thử nước mà không đụng vào ván thật.

        Chỉ chép những thứ thay đổi trong ván (lượt, cột trống, quân, bàn số, lịch sử).
        Nhanh hơn deepcopy của sách nhiều lần, quan trọng vì MCTS gọi hàm này hàng nghìn lần mỗi nước.
        """
        # __new__ tạo đối tượng trống, không chạy __init__ (khỏi dựng bàn mới rồi lại ghi đè)
        new_env = conn.__new__(conn)
        new_env.action_space = self.action_space
        new_env.observation_space = self.observation_space
        new_env.info = self.info
        new_env.xs = self.xs
        new_env.ys = self.ys
        new_env.game_piece = None
        new_env.turn = self.turn
        new_env.validinputs = list(self.validinputs)
        # mỗi cột là một list riêng, phải chép từng cột; chép cả list ngoài thôi thì
        # bản sao và bản gốc vẫn dùng chung các cột bên trong
        new_env.occupied = []
        for column in self.occupied:
            new_env.occupied.append(list(column))
        new_env.state = self.state.copy()
        new_env.done = self.done
        new_env.reward = self.reward
        new_env.history = list(self.history)
        return new_env

    def undo(self):
        """Hoàn tác nước đi cuối cùng. Ném InvalidMoveError nếu chưa có nước nào."""
        if len(self.history) == 0:
            raise InvalidMoveError("Chưa có nước nào để hoàn tác.")
        inp = self.history.pop()
        col = inp - 1
        # lấy quân cuối cùng của cột ra; màu quân đó chính là người vừa đi
        mover = self.occupied[col].pop()
        self.state[col][len(self.occupied[col])] = 0
        # tính lại danh sách cột hợp lệ từ đầu cho chắc chắn
        self.validinputs = []
        for c in range(7):
            if len(self.occupied[c]) < 6:
                self.validinputs.append(c + 1)
        self.done = False
        self.reward = 0
        self.turn = mover
        self.game_piece = None

    @classmethod
    def from_moves(cls, moves):
        """Tạo ván mới rồi đi lần lượt các cột trong danh sách moves."""
        env = cls()
        for move in moves:
            env.step(move)
        return env

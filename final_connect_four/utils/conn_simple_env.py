"""Engine Connect Four 7x6 và Game API thống nhất.

Kế thừa: utils/conn_simple_env.py, sách AlphaGo Simplified (Mark Liu), giấy phép MIT. Giữ quy ước: cột 1-7, state (7, 6), lượt 'red'/'yellow', reward theo góc bên đỏ.
Sửa: bỏ import turtle không dùng; step() ném lỗi khi nước đi không hợp lệ (cột đầy, ngoài 1-7, ván đã kết thúc).
Thêm: legal_actions(), is_terminal(), result(), render() dạng chữ, clone()thay deepcopy, undo().
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


# Số cột có thể thả quân
class action_space:
    def __init__(self, n):
        self.n = n


# Không gian quan sát của bàn cờ 
class observation_space:
    def __init__(self, row, col):
        self.shape = (row, col)


class conn():
    def __init__(self):
        self.action_space = action_space(7)
        self.observation_space = observation_space(7, 6)
        self.info = ""
        self.xs = [-300, -200, -100, 0, 100, 200, 300]
        self.ys = [-250, -150, -50, 50, 150, 250]
        # Thông tin quân cờ được đánh gần nhất
        self.game_piece = None
        self.reset()

    # Trả về 1 nước đi ngẫu nhiên
    def sample(self):
        return choice(self.validinputs)

    def reset(self):
        self.turn = "red"
        # Các cột chưa đầy - có thể đi
        self.validinputs = [1, 2, 3, 4, 5, 6, 7]
        self.occupied = [list(), list(), list(), list(), list(), list(), list()]
        self.state = np.zeros((7, 6), dtype=int)
        self.done = False
        self.reward = 0
        # Thêm: lịch sử các cột đã đi, dùng cho undo()
        self.history = []
        self.game_piece = None
        return self.state

    def _check_move(self, inp):
        if self.done:
            raise InvalidMoveError("Ván đã kết thúc, không thể đi thêm.")
        # Nhập vào True trả về 1 nên phải đưa vào
        if isinstance(inp, bool) or not isinstance(inp, Integral):
            raise InvalidMoveError(f"Nước đi phải là số nguyên 1-7, nhận được {inp!r}.")
        if inp < 1 or inp > 7:
            raise InvalidMoveError(f"Cột {inp} nằm ngoài 1-7.")
        if len(self.occupied[inp - 1]) >= 6:
            raise InvalidMoveError(f"Cột {inp} đã đầy.")

    # step() function: place piece on board and update state
    def step(self, inp):
        self._check_move(inp)
        inp = int(inp)
        self.game_piece = [inp - 1, len(self.occupied[inp - 1]), self.turn] #phục vụ đồ họa
        self.state[inp - 1][len(self.occupied[inp - 1])] = 2 * (self.turn == "red") - 1 #phục vụ tốc độ Engine 
        self.occupied[inp - 1].append(self.turn) #phục vụ mạng nơ ron
        self.history.append(inp)

        # Cập nhật danh sách nước đi hợp lệ
        if len(self.occupied[inp - 1]) == 6 and inp in self.validinputs:
            self.validinputs.remove(inp)
        # Kiểm tra có ai thắng không?
        if self.win_game(inp) == True:
            self.done = True
            # red thắng -> +1, yellow thắng -> -1
            self.reward = 2 * (self.turn == "red") - 1
            self.validinputs = []
        # Không còn nước đi hợp lệ -> hòa 
        elif len(self.validinputs) == 0:
            self.done = True
            self.reward = 0
        else:
            # Đổi lượt chơi
            if self.turn == "red":
                self.turn = "yellow"
            else:
                self.turn = "red"
        return self.state, self.reward, self.done, self.info

    # Thắng theo các hướng 
    # Thắng theo hàng ngang, 4 ô liên tiếp
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

    # Thắng theo hàng dọc, 4 ô liên tiếp
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

    # Thắng theo chéo xuôi /, 4 ô liên tiếp
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

    # Thắn theo chéo ngược \, 4 ô liên tiếp 
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

    # Define a win_game() function to check if someone wins the game
    def win_game(self, inp):
        win = False
        x = inp - 1
        y = len(self.occupied[inp - 1]) - 1
        # Check all winning possibilities
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
        new_env = conn.__new__(conn)
        new_env.action_space = self.action_space
        new_env.observation_space = self.observation_space
        new_env.info = self.info
        new_env.xs = self.xs
        new_env.ys = self.ys
        new_env.game_piece = None
        new_env.turn = self.turn
        new_env.validinputs = list(self.validinputs)
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

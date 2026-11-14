"""Demo: người–máy và máy–máy, chọn cấp độ AI cho từng bên.

Kế thừa: cách vẽ bàn cờ và hiệu ứng quân rơi trong conn_env.py (display_board, render), conn_manual trong ch03util.py, repo markhliu/AlphaGoSimplified của sách AlphaGo Simplified (Mark Liu), giấy phép MIT; bản sao ở tham_khao_sach/utils/.
Sửa: vẽ bàn cờ trên các turtle riêng để chơi được nhiều ván trong cùng một cửa sổ (bản của sách đóng cửa sổ sau mỗi ván).
Thêm: bấm chuột vào cột để đi; chọn cấp 1-7 cho mỗi bên bằng hộp thoại; dòng trạng thái; đánh dấu nước vừa đi; chế độ máy–máy; chế độ chỉ in chữ (--text).
Phụ trách: Nguyễn Trọng Phúc.

Chạy (đứng trong thư mục final_connect_four):
    python play.py            # cửa sổ bàn cờ, bấm chuột để đi
    python play.py --text     # chỉ in bàn cờ dạng chữ, gõ số cột
Tham số của từng cấp lấy từ config.yaml (mục agents). Cấp 5-7 cần file mô hình đã huấn luyện trong files/.
"""

import sys
import time

from utils.conn_simple_env import conn
from utils.arena import load_config, build_agent

LEVEL_NAMES = {
    1: "cấp 1 – ngẫu nhiên",
    2: "cấp 2 – luật heuristic",
    3: "cấp 3 – Alpha-Beta",
    4: "cấp 4 – MCTS/UCT",
    5: "cấp 5 – mạng policy",
    6: "cấp 6 – policy gradient",
    7: "cấp 7 – PUCT + mạng (AlphaGo)",
}

# Tọa độ tâm 7 cột và 6 hàng, giống bàn cờ turtle của sách
COLUMN_X = [-300, -200, -100, 0, 100, 200, 300]
ROW_Y = [-250, -150, -50, 50, 150, 250]
COLOR_NAME = {"red": "Đỏ", "yellow": "Vàng"}


def column_from_x(x):
    """Đổi tọa độ x của cú bấm chuột thành số cột 1-7; bấm ra ngoài bàn thì trả về None."""
    if x < -350 or x > 350:
        return None
    col = int(round((x + 300) / 100)) + 1
    if col < 1 or col > 7:
        return None
    return col


def make_player(level, cfg):
    """level 0 là người chơi, 1-7 là cấp AI. Trả về (tên hiển thị, hàm chọn nước hoặc None nếu là người)."""
    if level == 0:
        return ("Người chơi", None)
    agent_fn, info_fn = build_agent(cfg["agents"][f"cap{level}"])
    return (LEVEL_NAMES[level], agent_fn)


class GameController:
    """Phần logic của một ván demo, không phụ thuộc giao diện (để test được)."""

    def __init__(self, red_player, yellow_player):
        self.players = {"red": red_player, "yellow": yellow_player}
        self.env = conn()

    def player_name(self, color):
        return self.players[color][0]

    def is_human_turn(self):
        return (not self.env.done) and self.players[self.env.turn][1] is None

    def human_move(self, col):
        """Người chơi đi cột col. Trả về (hàng, cột) của quân vừa đặt, hoặc None nếu không hợp lệ."""
        if not self.is_human_turn():
            return None
        if col is None or col not in self.env.validinputs:
            return None
        return self._apply(col)

    def ai_move(self):
        """Máy chọn và đi một nước. Trả về (hàng, cột) của quân vừa đặt."""
        agent_fn = self.players[self.env.turn][1]
        col = agent_fn(self.env.clone())
        return self._apply(col)

    def _apply(self, col):
        color = self.env.turn
        row = len(self.env.occupied[col - 1])
        self.env.step(col)
        return (row, col, color)

    def status_text(self):
        """Dòng trạng thái hiện trên cửa sổ."""
        if self.env.done:
            winner = self.env.winner()
            if winner == "draw":
                return "Hòa!"
            return f"{COLOR_NAME[winner]} ({self.player_name(winner)}) thắng!"
        color = self.env.turn
        if self.is_human_turn():
            return f"Lượt {COLOR_NAME[color]}: bấm vào một cột để đi"
        return f"Lượt {COLOR_NAME[color]}: {self.player_name(color)} đang suy nghĩ..."


class TurtleBoard:
    """Cửa sổ bàn cờ bằng turtle: vẽ bàn, thả quân có hiệu ứng rơi, dòng trạng thái."""

    def __init__(self):
        import turtle
        self.turtle = turtle
        self.screen = turtle.Screen()
        self.screen.setup(760, 760)
        self.screen.title("Connect Four 7x6 – Game AI đa cấp độ")
        self.screen.tracer(0)
        # dời khung nhìn lên một chút để dòng trạng thái và bàn cờ nằm giữa cửa sổ
        self.screen.setworldcoordinates(-380, -340, 380, 400)
        self.board_pen = self._new_pen()
        self.disc_pen = self._new_pen()
        self.fall_pen = self._new_pen()
        self.mark_pen = self._new_pen()
        self.status_pen = self._new_pen()

    def _new_pen(self):
        pen = self.turtle.Turtle()
        pen.hideturtle()
        pen.up()
        pen.speed(0)
        return pen

    def draw_empty_board(self):
        """Khung xanh, 42 lỗ trắng và số cột (giống display_board của sách)."""
        for pen in (self.board_pen, self.disc_pen, self.fall_pen, self.mark_pen):
            pen.clear()
        pen = self.board_pen
        pen.goto(-350, -300)
        pen.pensize(5)
        pen.color("black", "blue")
        pen.down()
        pen.begin_fill()
        for side in (700, 600, 700, 600):
            pen.forward(side)
            pen.left(90)
        pen.end_fill()
        pen.up()
        for i in range(7):
            pen.goto(COLUMN_X[i] - 7, 305)
            pen.write(i + 1, font=("Arial", 20, "normal"))
        for x in COLUMN_X:
            for y in ROW_Y:
                pen.goto(x, y)
                pen.dot(80, "white")
        self.screen.update()

    def drop_disc(self, row, col, color):
        """Hiệu ứng quân rơi từ trên xuống rồi nằm ở ô (hàng, cột), giống render của sách."""
        x = COLUMN_X[col - 1]
        for r in range(5, row, -1):
            self.fall_pen.goto(x, ROW_Y[r])
            self.fall_pen.dot(80, color)
            self.screen.update()
            time.sleep(0.04)
            self.fall_pen.clear()
        self.disc_pen.goto(x, ROW_Y[row])
        self.disc_pen.dot(80, color)
        # đánh dấu nước vừa đi bằng một chấm đen nhỏ ở giữa quân
        self.mark_pen.clear()
        self.mark_pen.goto(x, ROW_Y[row])
        self.mark_pen.dot(16, "black")
        self.screen.update()

    def show_status(self, text):
        self.status_pen.clear()
        self.status_pen.goto(0, 345)
        self.status_pen.write(text, align="center", font=("Arial", 16, "bold"))
        self.screen.update()


class TurtleGame:
    """Điều khiển các ván chơi trên cửa sổ turtle theo kiểu chờ sự kiện (bấm chuột, hẹn giờ)."""

    def __init__(self, cfg, pause_ms=300):
        self.cfg = cfg
        self.pause_ms = pause_ms
        self.board = TurtleBoard()
        self.controller = None
        self.waiting_for_click = False

    def ask_player(self, color):
        """Hộp thoại chọn người hoặc cấp AI. Trả về None nếu người dùng bấm Hủy."""
        prompt = f"Bên {COLOR_NAME[color]}: 0 = người chơi, 1-7 = cấp AI"
        while True:
            level = self.board.screen.numinput("Chọn người chơi", prompt, default=0, minval=0, maxval=7)
            if level is None:
                return None
            try:
                return make_player(int(level), self.cfg)
            except Exception as error:
                # ví dụ chưa có file mô hình của cấp 5-7, hoặc cấp đó chưa viết xong
                prompt = f"Chưa dùng được cấp {int(level)} ({error}). Chọn lại – bên {COLOR_NAME[color]}:"

    def start(self):
        red = self.ask_player("red")
        if red is None:
            self.board.turtle.bye()
            return
        yellow = self.ask_player("yellow")
        if yellow is None:
            self.board.turtle.bye()
            return
        self.controller = GameController(red, yellow)
        self.board.draw_empty_board()
        self.board.show_status(self.controller.status_text())
        self.board.screen.onclick(self.on_click)
        self.board.screen.ontimer(self.next_turn, self.pause_ms)

    def next_turn(self):
        controller = self.controller
        if controller.env.done:
            self.board.show_status(controller.status_text() + "  –  bấm chuột để chơi ván mới")
            self.waiting_for_click = True
            return
        self.board.show_status(controller.status_text())
        if controller.is_human_turn():
            self.waiting_for_click = True
            return
        row, col, color = controller.ai_move()
        self.board.drop_disc(row, col, color)
        self.board.show_status(controller.status_text())
        self.board.screen.ontimer(self.next_turn, self.pause_ms)

    def on_click(self, x, y):
        if not self.waiting_for_click:
            return
        controller = self.controller
        if controller.env.done:
            self.waiting_for_click = False
            self.start()
            return
        move = controller.human_move(column_from_x(x))
        if move is None:
            self.board.show_status("Không đi được vào đó – hãy bấm vào một cột còn trống")
            return
        self.waiting_for_click = False
        row, col, color = move
        self.board.drop_disc(row, col, color)
        self.board.show_status(controller.status_text())
        self.board.screen.ontimer(self.next_turn, 100)

    def run(self):
        self.start()
        self.board.turtle.mainloop()


# Chế độ chữ (khi máy không mở được cửa sổ turtle)

def ask_number(prompt, allowed):
    """Hỏi đến khi người dùng nhập một số nằm trong allowed."""
    while True:
        text = input(prompt).strip()
        try:
            number = int(text)
        except ValueError:
            print("Hãy nhập một số.")
            continue
        if number in allowed:
            return number
        print(f"Chỉ nhận các giá trị: {sorted(allowed)}")


def choose_player_text(color, cfg):
    print(f"\nChọn cho bên {COLOR_NAME[color]}:")
    print("  0. Người chơi")
    for level, name in LEVEL_NAMES.items():
        print(f"  {level}. {name}")
    while True:
        level = ask_number("Lựa chọn: ", [0, 1, 2, 3, 4, 5, 6, 7])
        try:
            return make_player(level, cfg)
        except Exception as error:
            print(f"Chưa dùng được {LEVEL_NAMES[level]}: {error}")
            print("Hãy chọn cấp khác.")


def play_text_game(cfg):
    controller = GameController(choose_player_text("red", cfg), choose_player_text("yellow", cfg))
    print("\n" + controller.env.render())
    while not controller.env.done:
        color = controller.env.turn
        if controller.is_human_turn():
            while True:
                col = ask_number(f"Lượt {COLOR_NAME[color]}, nhập cột (1-7): ", [1, 2, 3, 4, 5, 6, 7])
                if controller.human_move(col) is not None:
                    break
                print("Cột này đã đầy, chọn cột khác.")
        else:
            row, col, color = controller.ai_move()
            print(f"\n{COLOR_NAME[color]} ({controller.player_name(color)}) đi cột {col}")
        print(controller.env.render())
    print("\n" + controller.status_text())


def main():
    cfg = load_config("config.yaml")
    if "--text" in sys.argv:
        while True:
            play_text_game(cfg)
            if input("\nChơi ván nữa? (c/k): ").strip().lower() != "c":
                break
        return
    try:
        game = TurtleGame(cfg)
    except Exception as error:
        print(f"Không mở được cửa sổ bàn cờ ({error}). Hãy chạy: python play.py --text")
        return
    game.run()


if __name__ == "__main__":
    main()

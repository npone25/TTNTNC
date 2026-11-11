"""Cấp 4: MCTS/UCT dạng cây.

Kế thừa: công thức UCT và mô phỏng ngẫu nhiên trong utils/ch08util.py (select, simulate), sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: dùng lõi cây trong mcts_tree.py thay cho thống kê phẳng ở gốc của sách.
Thêm: hàm tác tử nhận env trả về cột; chế độ số mô phỏng và thời gian; ghi số mô phỏng mỗi nước. Không thêm luật thắng/chặn.
Phụ trách: Nguyễn Trọng Phúc.
"""

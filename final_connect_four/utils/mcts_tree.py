"""Lõi tìm kiếm cây dùng chung cho cấp 4 (mcts_uct.py) và cấp 7 (alphago_tree.py).

Kế thừa: tên và vai trò bốn bước select, expand, simulate, backpropagate theo utils/ch08util.py (sách bản gốc là MCTS phẳng, chỉ thống kê ở gốc).
Sửa: không áp dụng.
Thêm: tự viết cây thật: nút cây, vòng lặp bốn bước, chế độ cố định số mô phỏng và cố định thời gian, seed, ghi số mô phỏng; cách chọn nhánh và cách đánh giá lá được truyền vào từ cấp 4 và cấp 7.
Phụ trách: Nguyễn Trọng Phúc.
"""

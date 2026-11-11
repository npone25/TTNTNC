"""Cấp 7: tìm kiếm cây PUCT kết hợp mạng policy, policy nhanh và value (cảm hứng AlphaGo).

Kế thừa: ba thành phần mạng và cách dùng trong utils/ch17util.py và notebook ch18, sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: thay tìm kiếm phẳng ở gốc bằng PUCT trên cây, dùng lõi trong mcts_tree.py.
Thêm: chế độ cố định số mô phỏng và thời gian; danh sách thành phần AlphaGo đã/chưa hiện thực.
Phụ trách: Nguyễn Trọng Phúc.
"""

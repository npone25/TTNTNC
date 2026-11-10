"""Cấp 6: REINFORCE (policy gradient) và huấn luyện mạng value.

Kế thừa: notebook ch15 (playing_red, playing_yellow, discount_rs, create_batch, huấn luyện value), sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: đối thủ là mô hình cấp 5 do nhóm tự huấn luyện.
Thêm: giai đoạn 2 tự chơi với các checkpoint cũ của chính nó; lưu checkpoint đều đặn; đường cong học; theo dõi tiến bộ qua đối thủ cố định (cấp 2, 3, 5). Dùng hàm mạng trong nets.py.
Phụ trách: Trần Quỳnh Hương.
"""

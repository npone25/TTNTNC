"""Hàm phục vụ cho mạng: mã hóa trạng thái, tạo mạng policy, policy gradient, value; huấn luyện; lưu và nạp.

Kế thừa: kiến trúc ở notebook ch11 mục 2.1 và ch15 mục 1.1, 2.3, sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: gom về một file thành các hàm tạo mô hình.
Thêm: đánh giá độ chính xác trên tập validation và test (sách không có); ghi đường cong học, thời gian huấn luyện và cấu hình máy.
Phụ trách: Nguyễn Trọng Phúc.
"""

# Connect Four 7×6 – Game AI đa cấp độ

Đồ án cuối kỳ môn Trí tuệ nhân tạo nâng cao. Bảy cấp tác tử: ngẫu nhiên, luật heuristic,
Alpha-Beta, MCTS/UCT, mạng policy học có giám sát, policy gradient, MCTS kết hợp mạng
(cảm hứng AlphaGo). Không thực hiện cấp 8.

Nhóm: Nguyễn Trọng Phúc, Trần Quỳnh Hương, Phạm Thiên Phúc.
Link GitHub: (điền)

**Thành viên nhóm: đọc docs/HUONG_DAN_NHOM.md trước khi bắt đầu.**

## 1. Cài đặt

Python 3.13 (3.10–3.12 cũng được), TensorFlow 2.21 với Keras 3, chạy bằng CPU.

    pip install -r requirements.txt

Cấu hình máy dùng cho thực nghiệm: (điền CPU, RAM, hệ điều hành)

## 2. Kiểm thử

    pytest            # test nhanh
    pytest -m slow    # test chậm

## 3. Demo

    python play.py

Chọn cấp độ AI (1-7) cho từng bên; chơi người–máy hoặc máy–máy.

## 4. Huấn luyện

Chạy theo thứ tự (mỗi bước lưu kết quả vào files/):

1. Sinh dữ liệu cấp 5: notebook nb3_cap5_policy.ipynb, phần 1 (lệnh: điền)
2. Huấn luyện cấp 5: nb3_cap5_policy.ipynb, phần 2 → files/(tên mô hình).keras
3. Huấn luyện cấp 6: nb4_cap6_rq2.ipynb → files/(tên checkpoint).keras
4. Mạng value cho cấp 7: (điền)

Ghi lại seed, số ván, số episode, thời gian huấn luyện cho từng bước.

## 5. Đánh giá

Toàn bộ tham số nằm trong config.yaml. Log thô ghi vào results/logs/, hình vào results/figures/.

- RQ1 và hai ablation: nb2_cap3_4_rq1.ipynb
- RQ2: nb4_cap6_rq2.ipynb
- RQ3: nb5_cap7_rq3.ipynb
- Giải vòng tròn cấp 1-7: nb6_giai_dau_tong_hop.ipynb

## 6. Cấu trúc thư mục

- utils/: toàn bộ mã nguồn (engine, tác tử, mạng, giải đấu)
- nb1 … nb6: notebook chạy thí nghiệm, tạo bảng và hình
- tests/: kiểm thử tự động
- files/: mô hình .keras (đẩy lên git) và dữ liệu thô (không đẩy, tạo lại được)
- results/: log thô và hình
- docs/: hướng dẫn nhóm, báo cáo, slide, poster
- tham_khao_sach/: bản gốc mã sách để đối chiếu

## 7. Phần kế thừa

Nền tảng là mã nguồn sách AlphaGo Simplified của Mark Liu (repo markhliu/AlphaGoSimplified,
giấy phép MIT, xem LICENSE_MarkLiu_MIT.txt). Mỗi file trong utils/ ghi rõ ở đầu file:
Kế thừa / Sửa / Thêm. Các trọng số .h5 có sẵn của sách không được dùng làm kết quả;
nhóm tự huấn luyện lại.

## 8. Đóng góp

Xem docs/HUONG_DAN_NHOM.md (bản đồ thư mục, phân công, lịch, tỷ lệ đóng góp).

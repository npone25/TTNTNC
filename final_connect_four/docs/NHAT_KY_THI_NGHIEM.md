# Nhật ký thí nghiệm

Ghi lại theo thứ tự thời gian mọi lần chạy, kết quả chính và lý do của từng quyết định. Số liệu chi tiết nằm trong notebook, bảng `results/tables/`, log thô `results/logs/` và file `.json` cạnh mỗi mô hình. Mã commit ghi trong ngoặc để đối chiếu với lịch sử GitHub.

## Môi trường

- Máy: Windows 11, Intel Core i5-12500H (12 nhân, 16 luồng), RAM 16 GB, SSD NVMe.
- Python 3.13.5 (Anaconda), TensorFlow 2.21 với Keras 3, chạy bằng CPU. TensorFlow từ bản 2.11 không còn hỗ trợ GPU trên Windows thường, nên card RTX của máy không được dùng.
- Các notebook chạy bằng `jupyter nbconvert --execute`, nhiều notebook song song trên cùng máy (mỗi notebook một tiến trình).

## Phiên bản 1 (tag `v1-ket-qua`, commit `91e7cb1`)

Kết quả và mô hình của phiên bản này được giữ ở `results/v1/` và `files/v1/` (commit `0e918a4`).

### Chuẩn bị

- Hàm lượng giá cấp 3 đổi sang "quân × đường mở", không còn trọng số chọn tay (`c4492a0`).
- Thêm khai cuộc 2 nước ngẫu nhiên có seed cho các cặp có tác tử tất định; không có nó thì 200 ván giữa cấp 5 và cấp 6 chỉ là 2 ván lặp lại (`3f9e26c`).
- Mỗi thí nghiệm một seed riêng trong `config.yaml` (`60cd1a9`); viết đủ 6 notebook (`0abc841`, `a6627b6`, `a6b2d86`).

### Dữ liệu chuyên gia

- 3 phần × 1000 ván, seed 1, 2, 3, chạy song song 3 tiến trình: 4330 s, 3991 s, 3783 s. Mỗi nước chọn ngẫu nhiên 50% Alpha-Beta độ sâu 3, 50% MCTS 1000 mô phỏng.
- 3000 ván: đỏ thắng 47,9%, vàng thắng 43,0%, hòa 9,1%; trung bình 33,3 nước mỗi ván.

### nb1: engine, cấp 1 và 2 (`8c10805`)

- Perft độ sâu 1 đến 6 khớp 7^d. Kiểm thử: 79 test lần đầu, 82 test sau khi thêm test dữ liệu.
- Cấp 2 thắng cấp 1 với score 0,995; cấp 1 đấu ngẫu nhiên 0,47. Luật cấp 2 dùng nhiều nhất là đi ngẫu nhiên an toàn (42%), nhìn trước (22%), thắng ngay (17%).

### nb2: cấp 3, cấp 4, RQ1

- Bộ 40 thế chiến thuật sinh trong 30,9 s; cấp 2, 3, 4 đúng 100%, cấp 1 đúng 25%.
- Cắt tỉa: điểm gốc giống hệt không cắt tỉa ở 50/50 thế; độ sâu 5 giảm từ 87.739 xuống 7.226 nút, xét nước từ giữa tiết kiệm thêm khoảng 2 lần so với 1 đến 7.
- Số mô phỏng MCTS khi đấu cấp 2: 25 mô phỏng 0,17; 400 mô phỏng 0,91; 1600 mô phỏng 0,98.
- Hàm lượng giá: hàm hiện tại thắng cấp 2 100/100; bỏ hẳn lượng giá (như sách chương 5–6) chỉ được 0,04 khi đấu hàm hiện tại; hai hàm thay thế không vượt hàm hiện tại.
- RQ1, cấp 3 đấu cấp 4 cùng thời gian: 0,2 s được 0,527; 0,5 s được 0,460; 1,0 s được 0,407. Thời gian càng nhiều thì MCTS càng lấn lên.
- Lưu ý: RQ1 chạy cùng lúc với notebook khác trên cùng máy. Hai bên trong mỗi ván chịu cùng điều kiện, nhưng độ sâu và số mô phỏng đạt được thấp hơn khi máy rảnh.

### Cấp 5 (nb3)

- Lần 1, 30 epoch: val_loss thấp nhất ở epoch 6 rồi tăng đến 1,91, tức mạng học thuộc; test acc 0,456; đấu cấp 2 được 0,27. Hình `nb3_duong_cong_policy_lan1_khong_dung_som.png`.
- Quyết định, chỉ dựa vào tập val: dừng sớm khi val_loss không giảm 5 epoch liền, giữ trọng số tốt nhất; thêm bản lật gương vào tập train (`88d89df`).
- Lần 2: dừng ở epoch 10, giữ epoch 5; test acc 0,505 (mốc luôn đoán một cột 0,19). Đấu cấp 1 được 0,87, cấp 2 được 0,10, cấp 3 được 0,14. Chiến thuật đúng 30%, riêng thấy nước thắng ngay chỉ 14% (`21ec1dc`).
- Không quay lại lần 1 dù lần 1 đấu cấp 2 tốt hơn, vì chọn mô hình theo ván đấu là dùng ván đánh giá để chọn siêu tham số.

### Cấp 6 (nb4)

- Lần 1, 300 + 300 lô (222 s): đấu cấp 5 được 0,15; phần thưởng giai đoạn 1 vẫn đang tăng ở lô cuối; đường đánh giá định kỳ chỉ ra 0 hoặc 0,5 vì cả hai bên tất định (`e66be44`).
- Sửa: đánh giá định kỳ có khai cuộc ngẫu nhiên; tăng lên 1500 + 1500 lô (`ae526c0`).
- Lần 2: hỏng ở giai đoạn 2, khoảng lô 700, trọng số thành NaN. Nguyên nhân: mạng dồn xác suất vào cột đầy nên tổng xác suất cột hợp lệ gần 0, phép chuẩn hóa chia cho 0. Sửa bằng cộng 1e-8 vào mẫu số và giới hạn gradient `clipnorm=1.0` (`e9aa5d5`).
- Lần 3: chạy ổn định, 951 s; đấu cấp 5 được 0,15, thua cấp 2, 3, 4 cả 200/200 ván (`8c6d8c7`). RQ2 phiên bản 1: cấp 6 không vượt cấp 5.

### Cấp 7 (nb5)

- Mạng value: 107.629 mẫu train; dừng ở epoch 6, giữ epoch 1; test acc 0,675 (mốc 0,44).
- Chọn mix bằng 40 ván riêng (seed 900): mix 0,0 đấu mix 0,5 được 0,375, khoảng tin cậy [0,24; 0,53] chứa 0,5; thời gian 391 ms so với 2017 ms mỗi nước, nên chọn 0,0 (`16ba287`).
- Ablation đấu cấp 3: MCTS thuần 0,16; thêm prior 0,40; thêm value 0,32; đầy đủ 0,32.
- RQ3 cùng số mô phỏng, đấu cấp 4: 50 mô phỏng 0,61; 200 mô phỏng 0,22; 800 mô phỏng 0,35. Cùng thời gian: 0,2 s 0,33; 0,5 s 0,18; 1,0 s 0,24. Cấp 7 đấu cấp 5: 0,877.
- Chiến thuật cấp 7 đúng 78% (chặn 85%, đòn đôi 46%), thấp hơn MCTS thuần vì prior từ cấp 5 yếu chiến thuật.

### Giải vòng tròn (nb6)

- 21 cặp × 200 ván. Xếp hạng theo score trung bình: cấp 3 0,849; cấp 4 0,789; cấp 7 0,708; cấp 2 0,564; cấp 5 0,355; cấp 6 0,171; cấp 1 0,064.
- Bên đi trước có lợi ở mọi cấp, rõ nhất ở cấp 7 (0,76 khi cầm đỏ, 0,66 khi cầm vàng).

### Ảnh hưởng của seed (nb7)

- Seed huấn luyện cấp 5: test acc 0,505; 0,504; 0,499, gần như không đổi.
- Seed ván đấu, cấp 4 đấu cấp 3, 100 ván mỗi seed: 0,385; 0,36; 0,25. Dao động lớn, nên cần nhiều ván và khoảng tin cậy.

## Quyết định làm phiên bản 2

Ba cấp học được đều yếu vì cùng một gốc: nhãn của cấp 5 là nước của bên thắng trong ván do hai chuyên gia khác nhau chơi, nên nhiễu, chỉ phủ một nửa số thế và thiếu thế chiến thuật. Cấp 6 khởi tạo ngẫu nhiên nên phải học lại từ đầu; cấp 7 dùng cấp 5 làm prior nên yếu chiến thuật theo.

Phiên bản 2 (`a3726a5`) giữ nguyên 3000 ván và đổi ba điểm:

- Cấp 5: mọi thế cờ (66.314 thế sau khi bỏ trùng) được gán nhãn bằng Alpha-Beta tìm sâu 5 (nhìn trước 6 nước), chạy song song 14 tiến trình.
- Cấp 6: khởi tạo từ mạng cấp 5 phiên bản 2 như AlphaGo, learning rate nhỏ, chọn checkpoint tốt nhất bằng 100 ván riêng (seed 5000).
- Cấp 7: mạng value lấy nhãn thắng/thua đã được tìm kiếm chứng minh khi có; dò `c_puct` trong {1,0; 1,5; 3,0} bằng 60 ván riêng (seed 900) với cấp 4.

## Phiên bản 2

(Điền khi chạy xong.)

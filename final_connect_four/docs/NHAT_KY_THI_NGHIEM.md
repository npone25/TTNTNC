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

### Cấp 5 (nb3, `88fd114`)

- Gán nhãn 66.314 thế bằng Alpha-Beta độ sâu 5, 14 tiến trình, 2169 s. Trong đó 11,7% thế người sắp đi thắng chắc, 10,4% thua chắc trong tầm nhìn.
- Huấn luyện: dừng ở epoch 12, giữ epoch 7; test acc 0,588 (lần 2 của phiên bản 1: 0,505). Đấu 100 ván: cấp 1 được 0,91, cấp 2 được 0,16, cấp 3 được 0,17 (30 ván hòa).
- Chiến thuật đúng 30%, bằng đúng phiên bản 1 nhưng là do trùng hợp: hai mô hình trả lời khác nhau ở 14/40 thế. Bộ 40 thế chiến thuật sinh từ ván ngẫu nhiên, khác xa thế trong ván chuyên gia mà mạng được học, nên đây là bài kiểm tra ngoài phân phối.

### Cấp 6 (nb4)

- Khởi tạo bằng bản sao mạng cấp 5, learning rate 0,00005, 1500 + 1500 lô (732 s + 869 s), không còn lỗi NaN.
- Chọn checkpoint bằng 100 ván riêng với cấp 5 (seed 5000): score tăng từ 0,54 (giai đoạn 1, lô 250) lên 0,80 (giai đoạn 2, lô 1000); mô hình cuối được 0,79. Giữ `gd2_lo1000`.
- RQ2, 200 ván mỗi cặp: cấp 6 thắng cấp 5 với score 0,74 [0,68; 0,80]. Trước đối thủ chung, cấp 6 không hơn cấp 5: cấp 1 được 0,83 so với 0,85; cấp 2 được 0,09 so với 0,13; cấp 3 được 0,035 so với 0,20; cấp 4 được 0,02 so với 0,04. Chiến thuật đúng 32,5% (cấp 5: 30%).
- Nhận xét: học tăng cường bằng cách đấu cấp 5 và checkpoint của chính nó làm cấp 6 giỏi khai thác điểm yếu của cấp 5, nhưng không giỏi hơn trước đối thủ khác.

### Cấp 7, lần 1 (dừng giữa chừng; kết quả ở `results/v2_lan1/`, mạng value ở `files/v2_lan1/`)

- Mạng value 3 lớp học kết quả ván, có thay nhãn bằng kết quả tìm kiếm chứng minh (chỉ 196 thế đổi nhãn). Dừng ở epoch 7, giữ epoch 2; test acc 0,690. Từ epoch 2, val_loss tăng trong khi train acc lên 0,84: mạng học thuộc ván.
- Dò `c_puct` bằng 60 ván với cấp 4: 1,0 được 0,425; 1,5 được 0,308; 3,0 được 0,275. Giá trị tốt nhất nằm ở mép dải, và ở mọi giá trị cấp 7 đều thua cấp 4.
- Dừng nb5 khi ablation đã chạy 341/800 ván, vì có thể sửa được nguyên nhân.

### Chẩn đoán trước lần 2

Kiểm tra nhanh trên máy ảo (2 nhân, Linux), cấp 7 đấu cấp 4, 30 ván cùng khai cuộc (seed 900). Các ván này chỉ để định hướng, không dùng làm kết quả:

- Mạng value hồi quy điểm Alpha-Beta (chỉ thế chuyên gia): test MSE 0,159, mốc luôn đoán trung bình 0,282. Score 0,45.
- Soi 15 ván thua bằng Alpha-Beta độ sâu 5: 8 ván cấp 7 đi một nước dẫn tới thua chắc trong 6 nước, dù vẫn còn nước không thua. Ví dụ: vàng có hai quân liền nhau ở hàng đáy với hai đầu trống, cấp 7 không chặn, vàng tạo hàng ba mở hai đầu.
- Thêm kiểm tra thắng ngay ở lá (`win_check`): score 0,533; còn 5 ván có nước thua chắc như trên.
- Thêm vào tập học mạng value 18.899 thế mới từ 1000 ván tự chơi của cấp 5, gán nhãn cùng cách. Trên cùng tập test chỉ gồm thế chuyên gia, MSE giảm từ 0,147 xuống 0,135. Score cùng 30 ván: 0,750 [0,57; 0,87].

### Cấp 7, lần 2: thay đổi

- Mạng value học điểm Alpha-Beta tìm sâu 5 (hồi quy, đầu ra tanh, loss MSE) thay cho kết quả ván. Thế thắng chắc hoặc thua chắc lấy +1 hoặc −1. Thế còn lại lấy tanh(k × điểm lượng giá), với k chọn trên tập train sao cho gần kết quả ván nhất. Trên máy ảo, k được chọn là 10.
- Thêm thế cờ từ 3000 ván tự chơi của mạng cấp 5: rút nước theo xác suất của mạng, mỗi nước có 10% khả năng đi ngẫu nhiên, seed 7. Các thế mới được gán nhãn cùng cách. Lý do: ván chuyên gia hiếm khi có thế sau một nước sai, trong khi cây tìm kiếm phải chấm cả những thế đó. AlphaGo lấy dữ liệu value theo cách tương tự.
- `win_check` bật cho cấp 7 và cho cả bốn cấu hình ablation, để ablation vẫn chỉ khác ở prior và value. Ngoài ra, nb5 chơi lại các ván thử với `win_check` tắt để đo riêng tác dụng của nó.
- Dải `c_puct` mở rộng thành {0,5; 0,7; 1,0; 1,5}.

### Cấp 7, lần 2: kết quả (nb5)

- Sinh 3000 ván tự chơi mất 57 s, trung bình 27 nước mỗi ván. Gán nhãn 53.845 thế mới bằng 14 tiến trình mất 2174 s. Mạng value học tổng cộng 120.159 thế.
- Chọn k = 10: sai số với kết quả ván 0,836, so với 0,839 ở k = 5 và 0,881 ở k = 20.
- Mạng value: dừng ở epoch 10, giữ epoch 5, huấn luyện 188,5 s. Test MSE 0,161 và MAE 0,267, mốc luôn đoán trung bình là MSE 0,336 và MAE 0,437. Trong 3404 thế thắng chắc hoặc thua chắc ở tập test, mạng đoán đúng dấu 88,1%.
- Dò `c_puct` bằng 60 ván với cấp 4: 0,5 được 0,75; 0,7 được 0,617; 1,0 được 0,55; 1,5 được 0,60. Chọn 0,5. Giá trị này lại nằm ở mép dưới của dải, nhưng không còn thời gian dò thêm; ghi là hạn chế.
- Tác dụng của `win_check`, cùng 60 ván với c = 0,5: bật được 0,75, tắt được 0,633.
- Ablation, 200 ván với cấp 3:
  - MCTS thuần: 0,075;
  - chỉ thêm prior: 0,282;
  - chỉ thêm value: 0,618;
  - đầy đủ: 0,652.
  Ở phiên bản 1, thêm value làm cấp 7 yếu đi; lần này mạng value là thành phần đóng góp nhiều nhất.
- RQ3, cùng số mô phỏng với cấp 4: 50 mô phỏng được 0,887; 200 được 0,848; 800 được 0,698. Cùng thời gian: 0,2 s được 0,422; 0,5 s được 0,422; 1,0 s được 0,287.
  - Cùng thời gian, cấp 4 chạy được nhiều mô phỏng hơn khoảng 2–2,5 lần (2592 so với 1027 ở 0,2 s), vì mỗi mô phỏng của cấp 7 phải gọi mạng.
  - Mỗi mô phỏng của cấp 7 có giá trị hơn mô phỏng của cấp 4, nhưng chưa đủ để bù số lượng. Lợi thế cũng thu hẹp khi tăng ngân sách.
- Cấp 7 đấu cấp 5: 0,935 [0,89; 0,96].
- Chiến thuật: cấp 7 đúng 80%, chặn 100%, đòn đôi 54%. Cấu hình "chỉ thêm value" đúng 97,5%.
- `agents.cap7.params.c_puct` trong `config.yaml` sửa thành 0,5 trước khi chạy giải vòng tròn.

### Giải vòng tròn (nb6, phiên bản 2)

- 21 cặp × 200 ván, mở đầu 2 nước ngẫu nhiên có seed. Notebook chạy một mình trên máy, nên số đo thời gian không bị notebook khác làm chậm. Các cặp có cấp 7 chiếm khoảng 3 trong khoảng 3,5 tiếng chạy.
- Xếp hạng theo score trung bình:
  1. cấp 7: 0,865
  2. cấp 3: 0,800
  3. cấp 4: 0,741
  4. cấp 2: 0,488
  5. cấp 6: 0,293
  6. cấp 5: 0,254
  7. cấp 1: 0,058
- Cấp 7 thắng mọi cấp còn lại: cấp 3 được 0,625; cấp 4 được 0,705; cấp 5 được 0,917; cấp 6 được 0,988; cấp 2 được 0,958; cấp 1 được 1,0.
- So với phiên bản 1: cấp 7 từ hạng 3 lên hạng 1. Cấp 6 vượt cấp 5 (đối đầu 0,802), nhưng cả hai vẫn thua cấp 2, 3, 4. Cấp 5 và 6 còn thua cấp 1 khoảng 15–20% số ván: hai mạng này đi tất định và yếu chiến thuật, nên nước ngẫu nhiên đôi khi tạo được hàng bốn mà chúng không chặn.
- Bên đi trước: mọi cấp đều có lợi khi cầm đỏ. Chênh lệch nhỏ nhất ở cấp 7 (0,872 so với 0,859) và cấp 4 (0,749 so với 0,732).
- Chi phí trung bình mỗi nước:
  - cấp 7: 686 ms (196 mô phỏng);
  - cấp 4: 87 ms (397 mô phỏng);
  - cấp 3: 59 ms (462 nút);
  - cấp 5, cấp 6: khoảng 2,4 ms.
- Mô hình: cấp 5 và 6 mỗi mạng 4,24 MB; cấp 7 dùng hai mạng, tổng 8,48 MB. Thời gian huấn luyện: cấp 5 113 s, cấp 6 1601 s, mạng value của cấp 7 302 s (chưa tính sinh ván tự chơi và gán nhãn).
- Chiến thuật: cấp 2, 3, 4 đúng 100%; cấp 7 đúng 80%; cấp 6 đúng 32,5%; cấp 5 đúng 30%; cấp 1 đúng 25%.

### Ảnh hưởng của seed (nb7, phiên bản 2)

- Seed huấn luyện cấp 5 (nhãn Alpha-Beta): test acc 0,583; 0,565; 0,576. Trung bình 0,575, độ lệch chuẩn khoảng 0,009. Cả ba seed đều giữ epoch 6.
- Seed ván đấu, cấp 4 đấu cấp 3: 0,385; 0,36; 0,25, trùng khớp từng ván với phiên bản 1. Lý do là cấp 3, cấp 4 và các seed không đổi; điều này cho thấy kết quả chạy lại được.

### Kiểm thử (nb1)

- Chạy lại sau mọi thay đổi của phiên bản 2: 88 test đều qua.

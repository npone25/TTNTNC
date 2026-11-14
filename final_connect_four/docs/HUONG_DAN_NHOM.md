
# HƯỚNG DẪN NHÓM – ĐỌC FILE NÀY ĐẦU TIÊN

Đồ án Connect Four 7×6, môn Trí tuệ nhân tạo nâng cao. File này cho biết: mỗi file chứa gì, ai làm, lấy đầu vào từ đâu, xuất kết quả ra đâu, hạn khi nào. Có thay đổi về phân công hay hạn thì sửa ngay trong file này.

Viết tắt: **P** = Nguyễn Trọng Phúc, **H** = Trần Quỳnh Hương, **T** = Phạm Thiên Phúc.

**Nguyên tắc chia việc:** P viết toàn bộ hạ tầng và các hàm phục vụ (engine, lõi tìm kiếm cây, giải đấu, thống kê, dữ liệu, hàm mạng, demo), cùng cấp 4 và cấp 7. H và T viết **code tác tử** của cấp mình và **chạy thí nghiệm cho RQ của mình**, bằng các hàm P cung cấp; riêng T làm thêm bộ thế cờ chiến thuật vì nhãn lấy từ Alpha-Beta của T. Thiếu hàm phục vụ nào thì báo P, không tự viết thêm vào file của P.

**Mốc cuối: xong toàn bộ ngày 20/11. Cấp 1–4 phải xong trước báo cáo tiến độ ngày 14/11.**

## 1. Bắt đầu

1. Dùng Python 3.13 (3.10–3.12 cũng được), rồi chạy `pip install -r requirements.txt` (TensorFlow 2.21 với Keras 3, chạy bằng CPU). Kiểm tra: `python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"` phải ra `2.21.x 3.x`.
2. Clone repo, tạo nhánh của mình: `git checkout -b <tên>/<việc>`, ví dụ `thienphuc/alphabeta`.
3. Mở file mình phụ trách (xem mục 4). Đọc docstring ở đầu file: Kế thừa / Sửa / Thêm là việc phải làm.
4. Muốn xem code gốc của sách thì mở `tham_khao_sach/`. Không sửa, không import từ thư mục đó.
5. Làm xong một phần: chạy `pytest`, commit, đẩy nhánh lên, tạo pull request vào `main`. Không đẩy thẳng lên `main`. Mỗi người chỉ sửa file của mình.

## 2. Quy ước chung (cả nhóm phải theo)

- Giữ quy ước của sách: cột đánh số **1–7**; `env.state` là mảng **(7, 6)** theo cột; lượt đi là `"red"` (đi trước) và `"yellow"`; `reward` = +1 nếu đỏ thắng, −1 nếu vàng thắng, 0 nếu hòa.
- Mọi tác tử là **một hàm nhận `env` và trả về một cột hợp lệ (1–7)**, giống sách (ví dụ `conn_think3(env)`, `MiniMax_conn(env, depth)`). Tham số thêm (độ sâu, số mô phỏng, mạng) đặt sau `env`. Tác tử không được làm thay đổi `env`; muốn thử nước thì dùng `env.clone()`.
- Tên hàm tác tử mà giải đấu (`utils/arena.py`, hàm `build_agent`) đang gọi: cấp 1 `rules.conn_random(env)`, cấp 2 `rules.conn_think3(env)`, cấp 3 `alphabeta.MiniMax_conn(env, depth=...)`, cấp 4 `mcts_uct.mcts_uct(env, ...)`, cấp 5 `policy_net.policy_move(env, model)`, cấp 6 `policy_gradient.pg_move(env, model)`, cấp 7 `alphago_tree.alphago_tree(env, ...)`. Muốn đổi tên hàm hay thêm tham số thì báo P để sửa `build_agent` và `config.yaml`.
- File tác tử nào muốn giải đấu ghi lại số nút/số mô phỏng thì khai báo biến toàn cục `LAST_INFO = {}` và gán dict mới sau mỗi nước, ví dụ `LAST_INFO = {"nodes": 1234, "depth": 4}` (xem `mcts_uct.py`).
- Engine báo lỗi `InvalidMoveError` khi đi nước sai, kể cả đi tiếp sau khi ván đã kết thúc. Giải đấu đưa cho tác tử một bản sao của bàn cờ và xử thua tác tử nào trả về nước sai.
- Ngẫu nhiên phải có seed để chạy lại ra cùng kết quả.
- Mạng dùng Keras 3 (đi kèm TensorFlow 2.21). Mã sách viết cho Keras 2 nên khi chép sang phải sửa: `tf.concat` khi dựng mạng (ch15) đổi thành `layers.Concatenate`; lớp đầu dùng `keras.Input(shape=(7, 6, 1))` thay cho tham số `input_shape=`; lưu mô hình đuôi `.keras`; import kiểu `import keras`, `from keras import layers`. Mọi hàm dựng, lưu, nạp mạng nằm trong `utils/nets.py`, không tự dựng mạng ở file khác.
- Đầu mỗi file kế thừa ghi Kế thừa / Sửa / Thêm. Mỗi lỗi của sách đã sửa phải có một test chứng minh.
- **Không** dùng các file `.h5` có sẵn của sách làm kết quả. Nhóm tự huấn luyện lại.
- Code viết đơn giản, từng bước, dễ giải thích khi vấn đáp.

## 3. Bản đồ thư mục: cái gì nằm ở đâu

- `utils/`: toàn bộ mã nguồn (engine, tác tử, mạng, giải đấu). Chỉ chứa hàm, không chạy thí nghiệm.
- `nb1` … `nb6` (ở thư mục gốc): notebook gọi hàm trong `utils/`, chạy huấn luyện, chạy thí nghiệm, vẽ bảng và hình. Đặt ở gốc để `from utils... import ...` chạy được ngay.
- `tests/`: kiểm thử tự động, chạy bằng `pytest`.
- `play.py`: demo chơi, chọn cấp độ cho từng bên.
- `config.yaml`: mọi tham số thí nghiệm (độ sâu, số mô phỏng, thời gian mỗi nước, số ván, seed). Không ghi cứng tham số trong notebook.
- `files/`: mô hình đã huấn luyện `.keras` (đẩy lên git) và dữ liệu ván cờ thô `.p`, `.npz` (không đẩy, tạo lại được).
- `results/logs/`: log thô từng ván đấu (đẩy lên git, đề bắt buộc nộp). `results/figures/`: hình cho báo cáo.
- `docs/`: file hướng dẫn này, `bao_cao/`, `slide/`, `poster/`.
- `tham_khao_sach/`: bản gốc mã sách để đối chiếu.
- `README.md`: hướng dẫn cho người chấm (cài đặt, huấn luyện, đánh giá, demo).

## 4. Từng file: chứa gì, ai làm, hạn

### Hạ tầng và hàm phục vụ (P)

- `utils/conn_simple_env.py` (**P**, 11/11, đã xong): engine. Đã có mã sách. Thêm `legal_actions()`, `is_terminal()`, `result()`, `render()` dạng chữ, `clone()`, `undo()`, báo lỗi khi đi nước sai. **Mọi file khác dùng file này.**
- `tests/test_env.py` (**P**, 11/11, đã xong): test thắng ngang/dọc/chéo, hòa, nước sai, undo, clone, perft.
- `tests/conftest.py` (**P**, đã xong): bộ kiểm tra thắng độc lập và **các hàm kiểm tra tác tử dùng chung**: `check_takes_win`, `check_blocks`, `check_never_illegal`, `check_reproducible`. H và T chỉ cần gọi các hàm này cho tác tử của mình trong `tests/test_agents.py` (xem phần cấp 4 làm mẫu).
- `tests/test_data_nets.py` (**P**, đã xong): test sinh dữ liệu, chia theo ván, mã hóa, mạng, lưu nạp.
- `utils/mcts_tree.py` (**P**, 12/11, đã xong): lõi tìm kiếm cây dùng chung (nút cây, vòng lặp select/expand/simulate/backpropagate, chế độ số mô phỏng và chế độ thời gian). Cách chọn nhánh và cách đánh giá lá được truyền vào, dùng cho cấp 4 và cấp 7.
- `utils/arena.py`, `config.yaml`, `tests/test_arena.py` (**P**, 13/11, đã xong): chơi một ván, giải vòng tròn đổi bên, nhiều seed, hai chế độ ngân sách, log JSONL, thống kê (thắng/hòa/thua, score, Wilson 95%, thời gian TB và p95, số nút/mô phỏng, ma trận đối đầu, ảnh hưởng bên đi trước). **T và H dùng file này để chạy RQ.**
- `utils/data.py` (**P**, 14/11, đã xong code): sinh ván chuyên gia bằng cấp 3 và 4, chia train/val/test theo ván, loại thế trùng, lưu vào `files/`. Mỗi ván mất khoảng 1 giây, nên 3.000 ván chỉ mất chừng 1 giờ trên một máy. Chạy lại sau khi T sửa xong cấp 3 (13/11).
- `utils/nets.py` (**P**, 15/11, đã xong code): mã hóa trạng thái cho mạng; hàm tạo các mạng của sách (policy, policy gradient, value); hàm huấn luyện có đánh giá validation và test; lưu, nạp mô hình; ghi đường cong học, thời gian huấn luyện và cấu hình máy. Hàm H dùng nhiều nhất: `nets.move_probabilities(env, model)` trả về xác suất các cột hợp lệ (tự nhận mạng một hay hai đầu vào), `nets.build_policy_net()`, `nets.build_pg_net()`, `nets.train_model(...)`, `nets.save_model(...)`, `nets.load_model(...)`.
- **Mạng value cho cấp 7** (**P**, 16/11): P tự huấn luyện trên ván của `data.py` (nhãn là kết quả ván), lưu `files/value_conn.keras`, nên cấp 7 không phải chờ cấp 6.
- `play.py` (**P**, 17/11, đã xong code): demo chọn cấp 1–7 cho mỗi bên, chơi người–máy hoặc máy–máy, vẽ bàn cờ bằng turtle theo cách của sách, luật chơi dùng `conn_simple_env`.
- `README.md` (**P**): điền lệnh cài đặt, huấn luyện, đánh giá, demo khi có code.

### Tác tử và việc của từng người

- `utils/rules.py` (**T**, 12/11): cấp 1 `conn_random`, cấp 2 `conn_think3`. Tự viết lại dựa trên mã sách ở `tham_khao_sach/utils/ch03util.py`, dùng engine `utils.conn_simple_env`. Sửa `to_avoid` (chỉ đúng cho bên đỏ), thêm seed, ghi thứ tự ưu tiên luật và cách xử lý xung đột luật trong docstring. Lưu ý: vòng lặp nhìn trước 2 nước trong `conn_think2/3` của sách đi tiếp `m2` cả khi `m1` đã làm ván kết thúc; engine mới sẽ báo lỗi ở đó, cần kiểm tra `done` trước khi đi `m2`.
- `utils/alphabeta.py` (**T**, 13/11): cấp 3. Đã có mã sách. Thêm hàm lượng giá khi chạm độ sâu, cờ `prune` bật/tắt cắt tỉa, đếm số nút, chế độ giới hạn thời gian. Đã thử: bản gốc của sách ở độ sâu 2 vẫn thua tác tử ngẫu nhiên 2/6 ván, vì chạm độ sâu là trả 0 (coi như hòa), nên hàm lượng giá là việc quan trọng nhất.
- `utils/tactics.py` (**T**, 16/11): khoảng 40 thế cờ chiến thuật (thắng ngay, buộc chặn, tạo đôi đe dọa), nhãn bằng Alpha-Beta tìm sâu, lưu `files/tactics.json`, hàm chấm độ chính xác theo nhóm cho tác tử bất kỳ.
- `utils/mcts_uct.py` (**P**, 13/11, đã xong): cấp 4. Cắm công thức UCT và mô phỏng ngẫu nhiên vào lõi `mcts_tree.py`. Không thêm luật thắng/chặn.
- `utils/policy_net.py` (**H**, 16/11): cấp 5. Hàm tác tử `policy_move(env, model)` chọn nước bằng mạng policy (dùng `nets.move_probabilities`); huấn luyện chạy trong `nb3` bằng hàm của `nets.py` và dữ liệu của `data.py`. Trước khi có dữ liệu thật (15/11) thì đọc ch11, ch15 và thử trên ván của `conn_random`.
- `utils/policy_gradient.py` (**H**, 16/11 đến 18/11): cấp 6, theo REINFORCE của sách (ch15), mạng tạo bằng `nets.build_pg_net()`, tác tử `pg_move(env, model)`. Giai đoạn 1 đấu với mô hình cấp 5, giai đoạn 2 tự chơi với các checkpoint cũ. Lưu ít nhất 5 checkpoint và đường cong học.
- `utils/alphago_tree.py` (**P**, 17/11, đã xong code, chờ mô hình thật): cấp 7, PUCT trên lõi cây, dùng mạng policy (cấp 5 của H) làm tiên nghiệm và mô phỏng nhanh, mạng value của P. Khi cấp 6 xong thì thử đổi tiên nghiệm sang mạng cấp 6.
- `tests/test_agents.py` (**người viết cấp nào thêm test cấp đó**): thắng ngay, chặn, không đi nước sai, chạy lại cùng seed ra cùng kết quả.

## 5. Notebook: ai chạy cái nào

- `nb1_engine_cap1_2` (**T**): minh họa engine, API, cấp 1 và 2.
- `nb2_cap3_4_rq1` (**T**): RQ1 và hai ablation bắt buộc (bật/tắt cắt tỉa Alpha-Beta; số mô phỏng MCTS).
- `nb3_cap5_policy` (**H**): huấn luyện cấp 5 (dữ liệu từ `data.py`).
- `nb4_cap6_rq2` (**H**): huấn luyện cấp 6, đường cong học, RQ2.
- `nb5_cap7_rq3` (**P**): cấp 7, RQ3.
- `nb6_giai_dau_tong_hop` (**P**): giải vòng tròn cấp 1–7, ma trận đối đầu, khoảng tin cậy, ảnh hưởng bên đi trước.

Mỗi hình và bảng trong báo cáo phải chạy ra từ một notebook. Hình lưu vào `results/figures/`, tên hình và bảng theo quy định của thầy.

## 6. Ai chờ ai (thứ tự phụ thuộc)

- Engine của P (11/11) → mọi người.
- Lõi cây của P (12/11) → cấp 4 (P, 13/11) và cấp 7 (P, 17/11).
- `arena.py` của P (13/11) → T và H chạy RQ.
- Cấp 3 của T và cấp 4 của P (13/11) → `data.py` của P (14/11) → cấp 5 của H (16/11) → cấp 6 của H (18/11) và cấp 7 của P (17/11).
- Cấp 3 của T (13/11) → bộ thế cờ của T (16/11) → mọi người đo độ chính xác chiến thuật.
- Trong lúc chờ: H đọc ch11, ch15 và thử mạng trên ván ngẫu nhiên; T làm cấp 3 trên mã sách, chỉ cần engine.

## 7. Câu hỏi nghiên cứu và người chủ

- **RQ1** (T): cấp 3 với cấp 4 ở cùng ngân sách 0,2 / 0,5 / 1,0 giây mỗi nước.
- **RQ2** (H): cấp 6 (policy gradient, đấu mô hình bắt chước rồi tự chơi) có vượt cấp 5 không.
- **RQ3** (P): cấp 7 (PUCT cây + mạng) so với cấp 4 và cấp 5, cả chế độ cố định số mô phỏng và cố định thời gian.

Người chủ RQ chạy thí nghiệm bằng `arena.py` và `tactics.py`, phân tích số liệu và viết phần kết quả của RQ đó (Chương 4) cùng câu trả lời (Chương 5).

## 8. Điểm kiểm soát rủi ro

- 14/11: nếu cấp 4 chưa chạy đúng thì bỏ cấp 7.
- 16/11: nếu cấp 5 chưa dùng được thì dừng cấp 6, dồn sức cho cấp 5 và 7.
- P giữ đường găng từ 11/11 đến 15/11. Nếu P trễ quá 1 ngày, chuyển demo (`play.py`) cho T.

## 9. Sản phẩm nộp ngoài code

- `docs/bao_cao/`: báo cáo, bản Word và PDF cuối. Mỗi người viết phần RQ của mình; người tổng hợp: (chốt sau).
- `docs/slide/`: slide thuyết trình. `docs/poster/`: poster tóm tắt kết quả. Làm ngày 19/11.
- Video demo: để link trong README.
- File nộp: `NhomXX_ConnectFour_TTNTNC_HoTen_MaSV.zip`, kèm link GitHub, lịch sử commit, phân công và tỷ lệ đóng góp.

## 10. Lịch tổng

- 11/11: engine, test (P); bắt đầu cấp 1–2 và cấp 3 trên mã sách (T); H đọc ch11, ch15 và chuẩn bị mạng.
- 12/11: lõi cây (P); cấp 1–2 xong (T).
- 13/11: cấp 4 xong (P); giải đấu (P); cấp 3 xong (T).
- 14/11: P sinh dữ liệu bằng data.py (khoảng 1 giờ cho 3.000 ván); báo cáo tiến độ (cấp 1–4 xong).
- 15/11: hàm mạng (P); huấn luyện cấp 5 (H); T chạy RQ1 + 2 ablation và làm bộ thế cờ.
- 16/11: cấp 5 xong (H); bộ thế cờ xong (T); bắt đầu cấp 6 (H) và cấp 7 (P).
- 17/11: cấp 7 xong (P); demo xong (P); RQ1 xong (T).
- 18/11: cấp 6 xong (H); RQ2 (H), RQ3 (P), giải vòng tròn cấp 1–7 (P).
- 19/11: biểu đồ, viết Chương 4–5, slide, poster.
- 20/11: chạy lại từ bản clone mới, quay video, đóng gói – XONG HẾT.

## 11. Tỷ lệ đóng góp

Điền cuối dự án, theo lịch sử commit và phần việc thực tế (đề yêu cầu nộp).

- Nguyễn Trọng Phúc: …%
- Trần Quỳnh Hương: …%
- Phạm Thiên Phúc: …%

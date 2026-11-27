# Connect Four 7×6 – Game AI đa cấp độ

Đồ án cuối kỳ môn Trí tuệ nhân tạo nâng cao. Bảy cấp tác tử: ngẫu nhiên, luật heuristic,
Alpha-Beta, MCTS/UCT, mạng policy học có giám sát, policy gradient, MCTS kết hợp mạng
(cảm hứng AlphaGo). Không thực hiện cấp 8.

Repository: https://github.com/npone25/TTNTNC (thư mục `final_connect_four`)

## Sản phẩm nộp và vị trí

| Yêu cầu | Vị trí |
|---|---|
| Báo cáo, slide, poster | `docs/bao_cao/`, `docs/slide/`, `docs/poster/` |
| Mã nguồn | `utils/` (engine, 7 tác tử, mạng, giải đấu), `play.py` (demo), `tests/` (kiểm thử) |
| File cấu hình | `config.yaml`: tham số từng cấp, số ván, seed, ngân sách tính toán |
| Hướng dẫn cài đặt, chạy | mục 1 đến 4 của file này |
| Dataset | `files/games_*.p` (3000 ván chuyên gia), `files/nhan_alphabeta.p` (66.314 thế gán nhãn bằng Alpha-Beta tìm sâu), `files/tu_choi_cap5.p` (3000 ván tự chơi của cấp 5) và `files/nhan_alphabeta_tu_choi.p` (53.845 thế mới từ các ván đó, gán nhãn cùng cách); thông tin sinh dữ liệu trong các file `.json` đi kèm |
| Notebook | `nb1` … `nb7` ở thư mục gốc, đã chạy, giữ nguyên output |
| Mô hình đã huấn luyện | `files/policy_conn.keras` (cấp 5), `files/pg_conn.keras` (cấp 6), `files/value_conn.keras` (mạng value cấp 7), mỗi mô hình kèm `.json` ghi siêu tham số, đường cong học, thời gian huấn luyện, cấu hình máy |
| Kết quả thực nghiệm | `results/logs/` (log thô từng ván), `results/tables/` (bảng CSV) |
| Hình ảnh | `results/figures/` |
| Nhật ký thí nghiệm | `docs/NHAT_KY_THI_NGHIEM.md`: mọi lần chạy, kết quả, lý do từng quyết định |
| Video demo | (cập nhật link) |
| Tài liệu tham khảo | mục 8 |
| Danh sách thành viên, tỷ lệ đóng góp | mục 9 |

### Hai phiên bản

- **Phiên bản 1** (tag git `v1-ket-qua`; kết quả trong `results/v1/`, mô hình trong `files/v1/`): cấp 5 học nước đi của bên thắng trong ván chuyên gia, cấp 6 học tăng cường từ trọng số ngẫu nhiên.
- **Phiên bản 2** (hiện tại): cấp 5 học nhãn Alpha-Beta tìm sâu ở mọi thế cờ; cấp 6 khởi tạo từ cấp 5 rồi học tăng cường; cấp 7 dùng mạng value hồi quy điểm Alpha-Beta tìm sâu trên thế chuyên gia và thế tự chơi, kiểm tra thắng ngay ở lá, `c_puct` chọn bằng ván thử riêng. Lần đầu của cấp 7 trong phiên bản này (mạng value học kết quả ván) được giữ ở `results/v2_lan1/` và `files/v2_lan1/`.

Lý do và số liệu của từng bước nằm trong `docs/NHAT_KY_THI_NGHIEM.md`.

## 1. Cài đặt

Python 3.13 (3.10–3.12 cũng được), TensorFlow 2.21 với Keras 3, chạy bằng CPU.

    pip install -r requirements.txt

Cấu hình máy dùng cho thực nghiệm: Windows 11, Intel Core i5-12500H (12 nhân, 16 luồng), RAM 16 GB.
TensorFlow từ bản 2.11 không hỗ trợ GPU trên Windows thường, nên mọi huấn luyện và đánh giá chạy bằng CPU.

## 2. Kiểm thử

    python -m pytest                                   # test nhanh
    python -m pytest --override-ini=addopts=           # toàn bộ, kể cả test chậm

## 3. Demo

    python play.py            # cửa sổ bàn cờ, bấm chuột để đi
    python play.py --text     # chỉ in bàn cờ dạng chữ, gõ số cột

Chọn cấp độ AI (1-7) cho từng bên; chơi người–máy hoặc máy–máy.

## 4. Chạy thí nghiệm và huấn luyện

Mọi thí nghiệm nằm trong 7 notebook ở thư mục gốc, chạy từ thư mục `final_connect_four`, ví dụ:

    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 nb2_cap3_4_rq1.ipynb

Ô đầu mỗi notebook có biến `QUICK`:

- `QUICK = True`: chạy thử rất nhanh (vài ván, vài epoch) để kiểm tra code; kết quả ghi vào các
  thư mục con `thu_nhanh/` và không dùng cho báo cáo.
- `QUICK = False`: chạy thật theo `config.yaml`.

Log thô ghi dần vào `results/logs/`; bị ngắt thì chạy lại notebook, các ván đã xong được bỏ qua.

Thứ tự chạy (notebook sau dùng kết quả của notebook trước):

1. `nb1_engine_cap1_2`: Game API, perft, toàn bộ kiểm thử, cấp 1 và 2, thống kê luật của cấp 2.
2. `nb2_cap3_4_rq1`: bộ thế cờ chiến thuật `files/tactics.json`; ablation bật/tắt cắt tỉa;
   ablation số mô phỏng MCTS; độ nhạy hàm lượng giá cấp 3; RQ1 (cấp 3 và cấp 4 cùng thời gian).
3. `nb3_cap5_policy`: sinh 3000 ván chuyên gia (3 phần, seed 1-3), gán nhãn mọi thế cờ bằng Alpha-Beta
   tìm sâu (chạy song song), huấn luyện cấp 5, lưu `files/policy_conn.keras`.
4. `nb4_cap6_rq2`: cấp 6 khởi tạo từ cấp 5, học tăng cường hai giai đoạn (đấu cấp 5, rồi tự chơi với
   checkpoint cũ), chọn checkpoint tốt nhất, lưu `files/pg_conn.keras`; RQ2.
5. `nb5_cap7_rq3`: sinh ván tự chơi của cấp 5 và gán nhãn thế mới; mạng value `files/value_conn.keras`;
   chọn `c_puct` bằng ván thử riêng, đo tác dụng của kiểm tra thắng ngay; bốn cấu hình ablation của cấp 7; RQ3.
6. `nb6_giai_dau_tong_hop`: giải vòng tròn cấp 1-7, ma trận đối đầu, ảnh hưởng bên đi trước,
   chi phí tính toán, độ chính xác chiến thuật, ván minh họa.
7. `nb7_anh_huong_seed`: ảnh hưởng của seed huấn luyện (cấp 5) và seed ván đấu (cấp 4 đấu cấp 3).

Mỗi ván của các cặp có tác tử tất định mở đầu bằng 2 nước ngẫu nhiên có seed (`opening_random_moves`),
để 200 ván không lặp lại cùng một ván.

## 5. Các tác tử

| Cấp | Tác tử | File | Hàm |
|---|---|---|---|
| 1 | Ngẫu nhiên | utils/rules.py | `conn_random(env)` |
| 2 | Luật heuristic | utils/rules.py | `conn_think3(env)` |
| 3 | Minimax Alpha-Beta | utils/alphabeta.py | `MiniMax_conn(env, depth=3)` |
| 4 | MCTS/UCT dạng cây | utils/mcts_uct.py | `mcts_uct(env, num_simulations=400)` |
| 5 | Mạng policy học có giám sát | utils/policy_net.py | `policy_move(env, model)` |
| 6 | Policy gradient (REINFORCE) | utils/policy_gradient.py | `pg_move(env, model)` |
| 7 | PUCT + mạng policy, value | utils/alphago_tree.py | `alphago_tree(env, prior, value, fast)` |

Mọi tác tử nhận `env` và trả về một cột 1-7, không làm thay đổi `env`.

## 6. Cấu trúc thư mục

- utils/: toàn bộ mã nguồn (engine, tác tử, mạng, giải đấu)
- nb1 … nb7: notebook chạy thí nghiệm, tạo bảng và hình
- tests/: kiểm thử tự động
- files/: dataset, mô hình `.keras`; `files/v1/`: mô hình phiên bản 1
- results/: log thô (logs/), bảng (tables/), hình (figures/); `results/v1/`: kết quả phiên bản 1
- docs/: hướng dẫn nhóm, nhật ký thí nghiệm, hướng phát triển, báo cáo, slide, poster
- tham_khao_sach/: bản gốc mã sách để đối chiếu

Checkpoint trung gian của cấp 6 (`files/pg_checkpoints*/`) quá nặng để đưa lên git, có trong file ZIP nộp.

## 7. Phần kế thừa

Nền tảng là mã nguồn sách AlphaGo Simplified của Mark Liu (repo markhliu/AlphaGoSimplified,
giấy phép MIT, xem LICENSE_MarkLiu_MIT.txt). Mỗi file trong utils/ ghi rõ ở đầu file:
Kế thừa / Sửa / Thêm. Các trọng số .h5 có sẵn của sách không được dùng làm kết quả;
nhóm tự huấn luyện lại.

## 8. Tài liệu tham khảo

1. Liu, M. (2024). *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*. CRC Press. Mã nguồn: github.com/markhliu/AlphaGoSimplified.
2. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
3. Knuth, D. E., & Moore, R. W. (1975). An analysis of alpha-beta pruning. *Artificial Intelligence*, 6(4), 293–326.
4. Kocsis, L., & Szepesvári, C. (2006). Bandit based Monte-Carlo planning. *ECML 2006*, 282–293.
5. Rosin, C. D. (2011). Multi-armed bandits with episode context. *Annals of Mathematics and Artificial Intelligence*, 61(3), 203–230.
6. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.
7. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529, 484–489.
8. Allis, L. V. (1988). *A Knowledge-based Approach of Connect-Four*. Master's thesis, Vrije Universiteit Amsterdam.
9. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.

## 9. Thành viên và tỷ lệ đóng góp

| Họ tên | MSSV | Nhiệm vụ, công việc thực hiện | Tỷ lệ |
|---|---|---|---|
| Nguyễn Trọng Phúc | 3123410281 | Engine và kiểm thử, lõi tìm kiếm cây, cấp 4, cấp 7, giải đấu và thống kê, sinh dữ liệu và hàm mạng, demo, notebook, chạy thí nghiệm, RQ3 | (điền) % |
| Trần Quỳnh Hương | (điền) | Cấp 5, cấp 6, RQ2 | (điền) % |
| Phạm Thiên Phúc | (điền) | Cấp 1, 2, 3, bộ thế cờ chiến thuật, RQ1 | (điền) % |
| **Tổng** | | | **100%** |

Phân công chi tiết và lịch sử thực hiện: `docs/HUONG_DAN_NHOM.md` và lịch sử commit trên GitHub.

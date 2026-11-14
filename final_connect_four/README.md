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

## 4. Chạy thí nghiệm và huấn luyện

Mọi thí nghiệm nằm trong 6 notebook ở thư mục gốc, chạy bằng Jupyter từ thư mục `final_connect_four`.
Ô đầu mỗi notebook có biến `QUICK`:

- `QUICK = True`: chạy thử rất nhanh (vài ván, vài epoch) để kiểm tra code; kết quả ghi vào các
  thư mục con `thu_nhanh/` và không dùng cho báo cáo.
- `QUICK = False`: chạy thật theo `config.yaml`.

Log thô ghi dần vào `results/logs/`; bị ngắt thì chạy lại ô đó, các ván đã xong được bỏ qua.
Bảng ghi vào `results/tables/` (CSV), hình vào `results/figures/`.

Thứ tự chạy (notebook sau dùng kết quả của notebook trước):

1. `nb1_engine_cap1_2`: Game API, chạy toàn bộ kiểm thử, cấp 2 đấu cấp 1, thống kê luật của cấp 2.
2. `nb2_cap3_4_rq1`: tạo bộ thế cờ chiến thuật `files/tactics.json`; RQ1 (cấp 3 và cấp 4 cùng thời gian);
   ablation bật/tắt cắt tỉa; ablation số mô phỏng MCTS.
3. `nb3_cap5_policy`: mỗi máy sinh ván chuyên gia với seed riêng (`MACHINE` = phuc / huong / thienphuc,
   khoảng 1 giây mỗi ván), gộp lại, huấn luyện cấp 5, lưu `files/policy_conn.keras`.
4. `nb4_cap6_rq2`: huấn luyện cấp 6 hai giai đoạn (đấu cấp 5, rồi tự chơi với checkpoint cũ),
   lưu `files/pg_conn.keras`; RQ2.
5. `nb5_cap7_rq3`: huấn luyện mạng value `files/value_conn.keras`; bốn cấu hình ablation của cấp 7; RQ3.
6. `nb6_giai_dau_tong_hop`: giải vòng tròn cấp 1-7, ma trận đối đầu, ảnh hưởng bên đi trước, thời gian mỗi nước.

Dùng trực tiếp trong Python:

    from utils.arena import load_config, build_agent, run_match, format_summary
    cfg = load_config()
    a = build_agent(cfg["agents"]["cap3"], overrides={"time_limit": 0.5})
    b = build_agent(cfg["agents"]["cap4"], overrides={"time_limit": 0.5})
    s = run_match(a, b, "cap3", "cap4", n_games=200, seed=1, log_path="results/logs/rq1.jsonl")
    print(format_summary(s))

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
- nb1 … nb6: notebook chạy thí nghiệm, tạo bảng và hình
- tests/: kiểm thử tự động
- files/: mô hình .keras (đẩy lên git) và dữ liệu thô (không đẩy, tạo lại được)
- results/: log thô (logs/), bảng (tables/) và hình (figures/)
- docs/: hướng dẫn nhóm, báo cáo, slide, poster
- tham_khao_sach/: bản gốc mã sách để đối chiếu

## 7. Phần kế thừa

Nền tảng là mã nguồn sách AlphaGo Simplified của Mark Liu (repo markhliu/AlphaGoSimplified,
giấy phép MIT, xem LICENSE_MarkLiu_MIT.txt). Mỗi file trong utils/ ghi rõ ở đầu file:
Kế thừa / Sửa / Thêm. Các trọng số .h5 có sẵn của sách không được dùng làm kết quả;
nhóm tự huấn luyện lại.

## 8. Đóng góp

Xem docs/HUONG_DAN_NHOM.md (bản đồ thư mục, phân công, lịch, tỷ lệ đóng góp).

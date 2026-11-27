# Hướng phát triển

Ghi lại các ý tưởng cải tiến chưa làm trong đồ án, để khi cần thì quay lại xem.

## 1. Tái sử dụng cây giữa các nước đi (tree reuse) cho cấp 4 và cấp 7

### Hiện tại code đang làm gì

Mỗi lần tới lượt, `mcts_uct(env)` (cấp 4) hoặc `alphago_tree(env, ...)` (cấp 7) gọi `run_search` trong `utils/mcts_tree.py`. Hàm này tạo một gốc mới tinh:

```python
root = Node(None, None, 1.0)
```

Sau đó nó chạy đủ số mô phỏng, chọn cột ở gốc được thăm nhiều nhất, rồi bỏ cả cây. Tới lượt sau, cây lại được xây từ đầu.

```
Lượt Đỏ : gốc = bàn hiện tại -> xây cây -> đi cột 4 -> bỏ cây
Lượt Vàng: đi cột 3
Lượt Đỏ : gốc = bàn mới -> xây cây mới từ đầu
```

### Tree reuse là gì

Sau khi mình đi cột 4 và đối thủ đi cột 3, thế cờ mới chính là nút "cột 4 -> cột 3" trong cây cũ. Nút đó đã có sẵn số lượt thăm và value_sum từ lượt trước. Thay vì bỏ đi, ta lấy nút đó làm gốc mới và chạy tiếp mô phỏng trên nó.

```
Cây cũ:        gốc
              / | \
            c3  c4  c7        <- mình đi c4
               / | \
             c3  c4  c7       <- đối thủ đi c3
             |
        (nhánh con đã có sẵn n, Σ)  -> dùng làm gốc mới
```

### Lợi ích

- Với cùng số mô phỏng mỗi nước, cây thật sự có nhiều lượt thăm hơn, vì được cộng thêm phần đã tìm ở lượt trước. Thường sẽ chọn nước chắc hơn.
- Ở chế độ cố định thời gian, phần đã tìm không bị lãng phí.

### Cách làm (phác thảo)

Hàm tác tử hiện nhận `env` và không nhớ gì giữa các lần gọi, nên phải thêm một chỗ lưu cây cũ, ví dụ một biến toàn cục trong `mcts_uct.py`:

```python
# cây của lượt trước và lịch sử nước đi lúc đó
SAVED = {"root": None, "history": None}


def find_reused_root(env):
    """Tìm trong cây cũ nút ứng với thế cờ hiện tại. Không tìm được thì trả None."""
    old_root = SAVED["root"]
    old_history = SAVED["history"]
    if old_root is None:
        return None
    # thế cờ hiện tại phải là thế cờ cũ đi thêm vài nước
    if env.history[:len(old_history)] != old_history:
        return None
    node = old_root
    # đi xuống cây theo các nước đã đi từ lúc đó (nước của mình và nước của đối thủ)
    for move in env.history[len(old_history):]:
        if move not in node.children:
            return None
        node = node.children[move]
    node.parent = None   # cắt khỏi cây cũ, nút này thành gốc mới
    return node
```

Các bước còn lại:

1. Sửa `run_search` nhận thêm tham số `root=None`. Nếu được truyền một nút thì dùng nút đó, không thì tạo `Node(None, None, 1.0)` như cũ.
2. Trong `mcts_uct`, gọi `find_reused_root(env)` rồi truyền vào `run_search`. Tìm xong thì lưu lại `SAVED["root"] = root` và `SAVED["history"] = list(env.history)`.
3. Khi bắt đầu ván mới (`env.history` rỗng hoặc không khớp) thì xóa `SAVED`.

Quy ước góc nhìn vẫn giữ nguyên. Nút được chọn làm gốc mới thuộc về đối thủ (người vừa đi), đúng như gốc bình thường, nên `select`, `backpropagate` và việc đổi dấu không phải sửa gì.

### Những điểm cần cẩn thận

- **Công bằng khi so sánh.** Tác tử có tree reuse thực chất được nhiều mô phỏng hơn con số ghi trong config. Nếu bật cho cấp 4 thì phải bật cho cả cấp 7 (hoặc tắt cả hai), và ghi rõ trong báo cáo, nếu không kết quả RQ1 và RQ3 sẽ không công bằng.
- **Một tác tử chơi nhiều ván song song hoặc hai bên cùng là MCTS.** Biến toàn cục `SAVED` dùng chung sẽ bị lẫn cây. Cần lưu theo từng tác tử (ví dụ dùng class thay cho hàm, mỗi tác tử một đối tượng).
- **Tái lập theo seed.** Kết quả của một nước giờ phụ thuộc cả cây của lượt trước, nên chạy lại một nước riêng lẻ sẽ khác. Chạy lại cả ván với cùng seed thì vẫn giống.
- **Kiểm thử.** Cần thêm test: gốc tái sử dụng đúng là thế cờ hiện tại (đi các nước trong `history` từ thế cũ ra đúng bàn), và tắt tree reuse thì kết quả giống hệt code hiện tại.

## 2. Ghi điểm cho cả các nước trong ván chơi thử (AMAF / RAVE)

### Ý tưởng

Hiện tại, khi chơi thử ngẫu nhiên từ một lá, chỉ kết quả cuối ván được ghi lên các nút trên đường đi. Các nước trong ván chơi thử (kể cả nước đầu tiên của Đỏ) bị bỏ, dù chúng cũng góp phần vào kết quả.

Biến thể AMAF ("all moves as first") và RAVE dùng thêm các nước đó: nếu trong ván chơi thử Đỏ có đi cột 5 và Đỏ thắng, thì nút "Đỏ đi cột 5" ở các tầng phía trên cũng được cộng một ít điểm phụ. Điểm phụ này được trộn với q thật, ảnh hưởng lớn khi nút còn ít lượt thăm và nhỏ dần khi lượt thăm tăng.

### Lợi ích và rủi ro

- Lợi: mỗi ván chơi thử cho nhiều thông tin hơn, cây ước lượng nhanh hơn ở ngân sách nhỏ (rất hiệu quả trong cờ vây).
- Rủi ro: ở Connect Four, giá trị của một cột phụ thuộc mạnh vào thời điểm đi (do trọng lực), nên "cột 5 tốt ở nước 1" chưa chắc đúng ở nước 9. Cần thí nghiệm mới biết có lợi không.
- Cần thêm vào Node hai trường (số lượt và tổng điểm AMAF), sửa backpropagate và công thức chọn nhánh.

## 3. Chơi thử tất cả các con ngay lúc mở rộng

### Ý tưởng

Khi mở rộng một lá, thay vì chỉ chơi thử một ván cho chính lá, chơi thử một ván cho từng con vừa tạo, để con nào cũng có điểm ngay.

### Nhận xét

- Mỗi lần mô phỏng tốn khoảng 7 lần hơn, nên với cùng ngân sách (400 lần hoặc 0,5 giây) cây sẽ nông hơn nhiều.
- Tốn công cho cả những nước rõ ràng là dở, trong khi UCT vốn đã tự dồn ván thử vào nhánh tốt.
- Làm "số mô phỏng" khó so sánh giữa các cấp (một lần mô phỏng chứa nhiều ván thử).
- Có thể thử như một ablation: cùng thời gian mỗi nước, so bản hiện tại với bản chơi thử mọi con.

Ghi chú liên quan: lõi cây hiện đánh giá lá trước rồi mới tạo hết con (kiểu AlphaZero) để dùng chung với cấp 7. Kiểu sách giáo khoa là tạo một con rồi chơi thử từ con đó. Có thể so hai kiểu này trong cùng một thí nghiệm.

## 4. Gom thế cờ thành lô khi gọi mạng, sau đó mới tính tới GPU

### Hiện trạng

Cấp 7 gọi mạng trên từng thế cờ một, khoảng 1.500 lần mỗi nước khi mix = 0,5 (khoảng 330 lần khi mix = 0). Mỗi lần gọi khoảng 0,76 ms, phần lớn là chi phí cố định của Python và TensorFlow chứ không phải tính toán. Vì vậy GPU hiện không giúp gì, và trên Windows thì TensorFlow từ bản 2.11 cũng không dùng được GPU (phải qua WSL2).

### Hướng làm

1. Gom lô trên CPU: chạy nhiều ván song song hoặc dùng virtual loss để mỗi lần lấy nhiều lá, gom vài chục đến vài trăm thế cờ rồi gọi mạng một lần.
2. Bỏ bớt chi phí TensorFlow khi suy luận: chép trọng số ra NumPy và tự tính, vì mạng rất nhỏ.
3. Khi lô đã lớn và mạng đã to thì mới chuyển sang GPU (WSL2 hoặc Colab có GPU).

Phải sửa mcts_tree.py (lấy nhiều lá cùng lúc, thêm virtual loss), sửa cấp 7 và viết lại test. Ước tính 1–2 ngày.

## 5. Tối ưu tốc độ cấp 1–4 (không cần GPU)

- Biểu diễn bàn cờ bằng bitboard (hai số nguyên 64 bit) thay cho mảng NumPy, kiểm tra thắng bằng phép dịch bit.
- Bảng chuyển vị (transposition table) cho Alpha-Beta: nhớ giá trị các thế cờ đã tính để không tính lại.
- Viết phần lõi bằng Numba hoặc C++.

Các cách này có thể nhanh hơn hàng chục lần, cho phép Alpha-Beta tìm sâu hơn và MCTS chạy nhiều mô phỏng hơn trong cùng thời gian. Đây là việc của CPU; GPU không hợp với tìm kiếm rẽ nhánh.

## 6. Tăng chất lượng các mô hình (cấp 5, 6, 7) khi có nhiều thời gian chạy máy

CPU hay GPU không làm mô hình tốt hơn, chỉ nhanh hơn. Chất lượng đến từ dữ liệu và lượng tính toán. Theo thứ tự đáng làm:

1. Dữ liệu chuyên gia: tăng từ 3.000 lên 20.000–50.000 ván; làm chuyên gia mạnh hơn (Alpha-Beta độ sâu 5–6, MCTS 3.000–5.000 mô phỏng). Nâng cùng lúc cấp 5, cấp 6 (giai đoạn 1) và cấp 7 (prior và mạng value).
2. Cấp 6: tăng số lô mỗi giai đoạn (300 lên 1.000–2.000), giữ nhiều checkpoint hơn cho giai đoạn tự chơi.
3. Huấn luyện mạng: chọn epoch theo tập validation (dừng sớm khi loss validation không giảm) thay vì cố định 30 epoch; thử mạng to hơn khi có nhiều dữ liệu.
4. Cấp 7: chọn mix và số mô phỏng theo kết quả đấu.

Phần lớn chỉ cần đổi config.yaml và số lô trong nb4, không phải sửa code tác tử. Riêng dừng sớm theo validation cần thêm vài dòng vào train_model trong utils/nets.py.

Lưu ý: ngân sách lúc chơi trong RQ1, RQ3 phải giữ cố định để so sánh công bằng; chỉ phần huấn luyện mới tăng.

## 7. Cấp 8: AlphaZero-inspired

### Gồm những gì

- Tác tử: giống cấp 7 nhưng dùng một mạng hai đầu (policy và value), đánh giá lá chỉ bằng đầu value, không mô phỏng. Dùng lại được run_search và puct_select_child.
- Vòng lặp tự chơi: tự chơi bằng PUCT, ghi phân phối lượt thăm ở gốc và kết quả ván; huấn luyện đầu policy theo phân phối lượt thăm, đầu value theo kết quả; cho mạng mới đấu mạng cũ để giữ checkpoint; lặp nhiều vòng. Không dùng dữ liệu chuyên gia.

### Việc cần làm

- utils/nets.py: mạng hai đầu và hàm huấn luyện hai hàm mất mát.
- utils/alphazero.py: hàm tác tử, hàm tự chơi một ván, vòng lặp huấn luyện và đấu checkpoint.
- config.yaml: mục cap8; notebook nb7; test.

### Ước tính chi phí (đo trên máy nhóm, CPU)

- Một nước tự chơi (200 mô phỏng, 1 lần gọi mạng mỗi lá): khoảng 0,16 s; một ván khoảng 5 s.
- Một vòng: 200 ván tự chơi (~17 phút) + huấn luyện (~2 phút) + đấu checkpoint 40 ván (~3 phút).
- 20 vòng (4.000 ván): ~7–8 giờ trên một tiến trình, ~2–3 giờ nếu chạy 4 tiến trình song song.
- Viết code và test: khoảng 1,5–2 ngày.

Với vài nghìn ván tự chơi, cấp 8 nhiều khả năng chỉ vượt được cấp 1–2. Muốn mạnh cần lượng tự chơi lớn hơn nhiều, khi đó nên làm cùng mục 4 (gom lô, GPU).

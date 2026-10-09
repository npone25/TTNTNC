# Báo Cáo Lab 2 - Nhóm X (Thiên Phúc)

## Bài 2. Mô hình hóa bài toán mê cung
**5 thành phần của bài toán tìm kiếm:**
1. **Trạng thái đầu (Initial state):** Tọa độ xuất phát $S(x_S, y_S)$ trong mê cung.
2. **Tập hành động (Actions):** 4 hướng di chuyển: `North` (Lên), `South` (Xuống), `West` (Trái), `East` (Phải), miễn là ô đó không phải là tường (tức là hợp lệ).
3. **Mô hình chuyển trạng thái (Transition model):** Thực hiện hành động $a$ tại trạng thái $(x, y)$ sẽ dẫn đến trạng thái mới $(x', y')$ tương ứng với hướng di chuyển.
4. **Trạng thái đích (Goal state):** Tọa độ của đích $G(x_G, y_G)$. (Nếu nhiều đích, tập đích là các ô chứa mục tiêu cần gom).
5. **Chi phí đường đi (Path cost):** Mỗi bước di chuyển (action) từ ô này sang ô kề cạnh đều có chi phí là 1. $c(s, a, s') = 1$.

**Ước lượng các đại lượng n, d, m, b (đối với mê cung nói chung / small_maze):**
- **$n$ (kích thước không gian trạng thái):** Tổng số ô có thể đi được (không phải tường) trong mê cung. Đối với small maze, $n$ khá nhỏ (khoảng 140 ô).
- **$d$ (độ sâu của lời giải tối ưu):** Khoảng cách đường đi ngắn nhất từ $S$ đến $G$. Ví dụ trong small_maze, đường đi tối ưu là 16 bước $\Rightarrow d = 16$.
- **$m$ (độ sâu tối đa):** Độ dài của con đường dài nhất không chứa vòng lặp trong mê cung. Thường $m \le n$.
- **$b$ (hệ số phân nhánh tối đa):** Từ một ô, tác tử có tối đa 4 hướng để đi, do đó $b \le 4$ (trung bình thực tế trong mê cung có tường thì $b$ khoảng 1 đến 3).

---

## Bài 4. Depth-Limited DFS và Iterative Deepening Search (IDS)
- **Số lần tăng giới hạn độ sâu (IDS):** Để tìm được lời giải trong `small_maze` có độ sâu thực tế là 16, thuật toán IDS bắt đầu từ limit = 0 và tăng dần mỗi bước 1 đơn vị. Do đó, nó sẽ phải tăng giới hạn độ sâu **16 lần** (từ 0 lên 1, ..., lên 16) trước khi tìm thấy đích ở limit = 16.
- **Giải thích đặc tính của IDS:**
  - IDS tìm kiếm theo độ sâu tăng dần (lặp lại DFS với giới hạn $l=0, 1, 2...$). Vì ở mỗi vòng lặp nó dùng thuật toán DFS, nó kế thừa **khả năng tiết kiệm bộ nhớ cực tốt của DFS** (bộ nhớ $O(b \times d)$ thay vì $O(b^d)$).
  - Tuy nhiên, vì nó duyệt quét sạch các nút ở độ sâu $l$ rồi mới tăng lên $l+1$, hành vi duyệt này giống hệt kiểu lan tỏa theo tầng của BFS. Nhờ đó, nó cũng kết hợp được đặc tính **luôn tìm thấy lời giải tối ưu (đường đi ngắn nhất)** của BFS.

---

## Bài 7. Kiểm tra chu trình trong DFS (Cycle Checking)
So sánh 3 phương án cycle checking:
1. **Kiểm tra trạng thái trên đường đi hiện tại (Path checking):**
   - *Đặc điểm:* Chỉ chống được việc đi lùi hoặc lặp lại trên đúng 1 nhánh đang đi.
   - *Số lần lặp / Hiện tượng lặp:* Rất dễ bị lặp lại ở những ô ngã tư khi đi theo một nhánh khác tới. Thuật toán có thể tốn rất nhiều thời gian duyệt đi duyệt lại một khu vực (số lần mở nút rất cao).
   - *Kích thước frontier:* Lớn hơn vì phải nạp nhiều trạng thái trùng lặp từ các nhánh khác nhau.

2. **Kiểm tra đường đi và không thêm lại trạng thái đang có trong frontier (Memoization/Explored Set):**
   - *Đặc điểm:* Sử dụng bộ nhớ bổ sung để ghi nhớ tất cả các ô đã từng đưa vào frontier hoặc đã đi qua.
   - *Số lần lặp:* Không bao giờ mở lại một ô đã đi qua, do đó triệt tiêu hoàn toàn sự lặp trạng thái.
   - *Kích thước frontier / Bộ nhớ:* Tốn nhiều bộ nhớ nhất do phải lưu trữ Explored Set khổng lồ (kích thước $O(b^d)$).

3. **Điều chỉnh frontier bằng cách đưa trạng thái tìm thấy lên đầu stack (Frontier Modification):**
   - *Đặc điểm:* Khắc phục việc phải sinh lại các node bằng cách nếu node đó đã có trong frontier, ta cập nhật lại và đẩy lên đầu stack để ưu tiên duyệt.
   - *Hiện tượng lặp:* Giảm tối đa hiện tượng lặp không cần thiết mà vẫn đảm bảo tính chất duyệt sâu (DFS).

---

## Câu 12. Cycle checking làm tăng chi phí quản lý dữ liệu
Cycle checking giúp loại bỏ hoàn toàn các vòng lặp (giảm lặp trạng thái), nhưng lại buộc chúng ta phải **lưu trữ toàn bộ các trạng thái đã từng mở (Explored Set) hoặc đang nằm trong Frontier**. 
- Trong môi trường rộng hoặc số lượng ô lớn, kích thước của tập hợp này tăng theo cấp số nhân $O(b^d)$.
- Mỗi lần thuật toán xét 1 node mới, nó phải thực hiện phép **tra cứu (lookup)** xem node này đã tồn tại trong Explored Set hay Frontier chưa. Nếu cấu trúc dữ liệu không được tối ưu (ví dụ dùng list thay vì hash set), chi phí thời gian cho mỗi lần tra cứu sẽ rất cao.
- **Hậu quả:** Làm mất đi ưu điểm tiết kiệm bộ nhớ vốn có của DFS và tăng gánh nặng xử lý dữ liệu.

---

## Bảng kết quả thí nghiệm (Chờ gom số liệu)
*Thiên Phúc nhớ kết hợp số liệu từ các bạn (Trọng Phúc và Quỳnh Hương) để điền vào bảng dưới đây:*

| Maze | Thuật toán | Path cost | Nút mở rộng | Max depth | Max frontier | Thời gian |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| small | BFS | 16 | | | | |
| small | DFS | | | | | |
| small | GBFS | | | | | |
| small | A* | | | | | |
| medium | BFS | | | | | |
| medium | A* | | | | | |
| large | BFS/A* | | | | | |
| open | DFS+Cycle | | | | | |
*(Cần thu thập thêm từ output thực tế khi chạy full notebook)*

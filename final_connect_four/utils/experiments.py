"""Hàm dùng chung cho 6 notebook thí nghiệm: đường dẫn, tạo tác tử theo tên, bảng và hình.

Kế thừa: không (tự viết).
Sửa: không áp dụng.
Thêm: chế độ chạy thử nhanh (QUICK) ghi ra thư mục riêng để không lẫn với kết quả thật;
      đổi đường dẫn mô hình theo chế độ; bảng kết quả dạng pandas; vẽ hình score kèm khoảng tin cậy và ma trận đối đầu.
Phụ trách: Nguyễn Trọng Phúc.

Mỗi notebook mở đầu bằng:
    QUICK = False
    from utils.experiments import setup
    ex = setup(QUICK)
    a = ex.agent("cap3", overrides={"time_limit": 0.5})
Chế độ QUICK ghi log, hình, mô hình vào các thư mục con "thu_nhanh", nên chạy thử bao nhiêu
lần cũng không đè lên kết quả thật.
"""

import copy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.arena import load_config, build_agent, read_log

# Tên hiển thị của từng cấp trong bảng và hình
LEVEL_LABELS = {
    "cap1": "Cấp 1", "cap2": "Cấp 2", "cap3": "Cấp 3", "cap4": "Cấp 4",
    "cap5": "Cấp 5", "cap6": "Cấp 6", "cap7": "Cấp 7", "random": "Ngẫu nhiên",
}


class Experiment:
    """Giữ cấu hình và các đường dẫn của một lần chạy notebook."""

    def __init__(self, quick=False, config_path="config.yaml"):
        self.quick = quick
        self.cfg = load_config(config_path)
        suffix = "thu_nhanh" if quick else ""
        self.log_dir = os.path.join(self.cfg["log_dir"], suffix)
        self.fig_dir = os.path.join("results", "figures", suffix)
        self.table_dir = os.path.join("results", "tables", suffix)
        self.model_dir = os.path.join("files", suffix)
        for folder in (self.log_dir, self.fig_dir, self.table_dir, self.model_dir):
            os.makedirs(folder, exist_ok=True)

    def n(self, full, quick):
        """Số lượng dùng cho lần chạy thật (full) hoặc chạy thử (quick)."""
        return quick if self.quick else full

    def log(self, name):
        return os.path.join(self.log_dir, name)

    def fig(self, name):
        return os.path.join(self.fig_dir, name)

    def table(self, name):
        return os.path.join(self.table_dir, name)

    def model(self, name):
        return os.path.join(self.model_dir, name)

    def spec(self, name, section="agents"):
        """Mục cấu hình của một tác tử, đã đổi đường dẫn mô hình sang thư mục của chế độ đang chạy.

        section: "agents" cho cấp 1-7, "rq3_ablation" cho bốn cấu hình cấp 7.
        """
        spec = copy.deepcopy(self.cfg[section][name])
        for key, value in spec.items():
            if key.endswith("model") and isinstance(value, str):
                spec[key] = os.path.join(self.model_dir, os.path.basename(value))
        return spec

    def agent(self, name, overrides=None, section="agents"):
        """Tạo tác tử theo tên trong config.yaml. Trả về cặp (hàm chọn nước, hàm lấy thông tin)."""
        return build_agent(self.spec(name, section), overrides)

    def model_ready(self, name, section="agents"):
        """Mọi mô hình mà tác tử cần đã có file chưa (để bỏ qua cấp chưa huấn luyện)."""
        spec = self.spec(name, section)
        for key, value in spec.items():
            if key.endswith("model") and isinstance(value, str) and not os.path.exists(value):
                return False
        return True


def setup(quick=False, config_path="config.yaml"):
    exp = Experiment(quick, config_path)
    mode = "CHẠY THỬ NHANH (kết quả không dùng cho báo cáo)" if quick else "chạy thật"
    print(f"Chế độ: {mode}. Log: {exp.log_dir} | Hình: {exp.fig_dir} | Mô hình: {exp.model_dir}")
    return exp


def summaries_table(summaries):
    """Danh sách thống kê từ run_match thành bảng, mỗi dòng một cặp đấu."""
    rows = []
    for s in summaries:
        rows.append({
            "tác tử": s["agent"], "đối thủ": s["opponent"], "số ván": s["games"],
            "thắng": s["wins"], "hòa": s["draws"], "thua": s["losses"],
            "score": round(s["score"], 3),
            "Wilson 95%": f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]",
            "TB (ms)": round(s["time_mean"] * 1000, 1),
            "p95 (ms)": round(s["time_p95"] * 1000, 1),
            "xử thua": s["forfeits"],
        })
    return pd.DataFrame(rows)


def records_of(log_path, name_a, name_b):
    """Các ván của cặp (A, B) trong một file log."""
    result = []
    for r in read_log(log_path):
        if r["pair"] == [name_a, name_b]:
            result.append(r)
    return result


def info_mean(records, agent_name, key):
    """Trung bình một số liệu trong LAST_INFO (ví dụ "depth", "nodes", "simulations")
    trên các nước của tác tử agent_name. Không có số liệu thì trả về None."""
    values = []
    for r in records:
        color = "red" if r["red"] == agent_name else "yellow"
        for i, info in enumerate(r.get("move_info", [])):
            mover = "red" if i % 2 == 0 else "yellow"
            if mover == color and key in info:
                values.append(info[key])
    if not values:
        return None
    return float(np.mean(values))


def plot_scores(x_values, series, xlabel, title, path, ylabel="score"):
    """Vẽ score theo một trục (ngân sách, số mô phỏng...) kèm khoảng tin cậy.

    series: dict {tên đường: danh sách thống kê run_match theo thứ tự x_values}.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, summaries in series.items():
        scores = [s["score"] for s in summaries]
        low = [s["score"] - s["ci_low"] for s in summaries]
        high = [s["ci_high"] - s["score"] for s in summaries]
        ax.errorbar(x_values, scores, yerr=[low, high], marker="o", capsize=4, linewidth=2, label=name)
    ax.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_matrix(matrix, names, path, title="Score của hàng khi đấu cột"):
    """Ma trận đối đầu dạng ô màu, số in trong ô."""
    size = len(names)
    values = np.full((size, size), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a != b and b in matrix.get(a, {}):
                values[i, j] = matrix[a][b]
    fig, ax = plt.subplots(figsize=(1.0 * size + 2, 0.85 * size + 1.5))
    image = ax.imshow(values, cmap="RdBu", vmin=0, vmax=1)
    labels = [LEVEL_LABELS.get(n, n) for n in names]
    ax.set_xticks(range(size), labels, rotation=30, ha="right")
    ax.set_yticks(range(size), labels)
    for i in range(size):
        for j in range(size):
            if not np.isnan(values[i, j]):
                color = "white" if abs(values[i, j] - 0.5) > 0.3 else "black"
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def matrix_table(matrix, names):
    """Ma trận đối đầu dạng bảng pandas (ô chéo để trống)."""
    data = {}
    for b in names:
        column = []
        for a in names:
            if a == b or b not in matrix.get(a, {}):
                column.append(None)
            else:
                column.append(round(matrix[a][b], 3))
        data[LEVEL_LABELS.get(b, b)] = column
    return pd.DataFrame(data, index=[LEVEL_LABELS.get(a, a) for a in names])

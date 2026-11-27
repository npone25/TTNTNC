"""Hàm phục vụ cho mạng: mã hóa trạng thái, tạo mạng policy, policy gradient, value; huấn luyện; lưu và nạp.

Kế thừa: kiến trúc ở notebook ch11 mục 2.1 và ch15 mục 1.1, 2.3, sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: chuyển sang Keras 3 (TensorFlow 2.21): dùng keras.Input thay tham số input_shape; nối hai đầu vào bằng layers.Concatenate thay tf.concat (Keras 3 báo lỗi với tf.concat); lưu đuôi .keras; nhãn dạng số nguyên (sparse) thay one-hot.
Thêm: đánh giá trên tập validation và test (sách không có); dừng sớm theo val_loss và lấy lại trọng số tốt nhất; ghi đường cong học, thời gian huấn luyện và cấu hình máy; gọi mạng qua tf.function (nhanh hơn model.predict của sách khoảng 60 lần khi đánh giá từng thế cờ lẻ).
Phụ trách: Nguyễn Trọng Phúc.

Quy ước đầu vào: bàn cờ nhìn từ phía NGƯỜI SẮP ĐI (quân mình = 1, đối thủ = -1), hình dạng (7, 6, 1).
Đầu ra mạng value: 3 lớp [hòa, người sắp đi thắng, người sắp đi thua]; phiên bản 2 có thêm mạng value hồi quy
một đầu ra trong [-1, 1], học điểm của Alpha-Beta tìm sâu.
"""

import json
import os
import time

# tắt bớt log thông báo của TensorFlow (phải đặt trước khi import tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from utils.data import machine_info


# Mã hóa trạng thái

def player_view(env):
    """Bàn cờ (7, 6) nhìn từ phía người sắp đi: đỏ giữ nguyên, vàng nhân -1 (giống sách)."""
    if env.turn == "red":
        board = env.state
    else:
        board = -env.state
    return np.array(board, dtype=np.float32)


def encode_conv(env):
    """Đầu vào cho mạng tích chập: (1, 7, 6, 1)."""
    return player_view(env).reshape(1, 7, 6, 1)


def encode_flat(env):
    """Đầu vào dạng phẳng 42 ô cho mạng policy gradient: (1, 42)."""
    return player_view(env).reshape(1, 42)


# Tạo mạng

def build_policy_net(learning_rate=0.001):
    """Mạng policy cấp 5, cùng kiến trúc với fast_model ở ch11 của sách."""
    model = keras.Sequential([
        keras.Input(shape=(7, 6, 1)),
        layers.Conv2D(filters=128, kernel_size=(4, 4), padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(7, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    return model


def build_value_net(learning_rate=0.0005, regression=False):
    """Mạng value, cùng kiến trúc thân với value_model ở ch15 của sách.

    regression = False: 3 lớp đầu ra [hòa, người sắp đi thắng, người sắp đi thua] như sách.
    regression = True (phiên bản 2): một đầu ra tanh trong [-1, 1], học thẳng điểm của Alpha-Beta tìm sâu.
    """
    if regression:
        head = layers.Dense(1, activation="tanh")
        loss = "mse"
        metrics = ["mae"]
    else:
        head = layers.Dense(3, activation="softmax")
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model = keras.Sequential([
        keras.Input(shape=(7, 6, 1)),
        layers.Conv2D(filters=128, kernel_size=(4, 4), padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        head,
    ])
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)
    return model


def build_pg_net():
    """Mạng policy gradient cấp 6, cùng kiến trúc ch15 của sách (hai đầu vào).

    Đầu vào theo thứ tự [bàn cờ phẳng (42,), bàn cờ tích chập (7, 6, 1)] như sách.
    Không compile: cấp 6 tự cập nhật bằng tf.GradientTape.
    """
    conv_inputs = layers.Input(shape=(7, 6, 1))
    conv = layers.Conv2D(filters=128, kernel_size=(4, 4), padding="same", activation="relu")(conv_inputs)
    flat = layers.Flatten()(conv)
    inputs = layers.Input(shape=(42,))
    # Keras 3: dùng lớp Concatenate thay cho tf.concat của sách
    two_inputs = layers.Concatenate(axis=1)([flat, inputs])
    common = layers.Dense(256, activation="relu")(two_inputs)
    common = layers.Dense(64, activation="relu")(common)
    action = layers.Dense(7, activation="softmax")(common)
    return keras.Model(inputs=[inputs, conv_inputs], outputs=action)


# Gọi mạng nhanh

_compiled_calls = {}


def call_model(model, inputs):
    """Gọi mạng trên một thế cờ, trả về mảng numpy.

    Mạng Keras được bọc tf.function một lần rồi dùng lại (nhanh hơn predict rất nhiều).
    Vật khác (ví dụ mạng giả trong test) thì gọi trực tiếp.
    """
    if not isinstance(model, keras.Model):
        return np.array(model(inputs, training=False))
    key = id(model)
    if key not in _compiled_calls:
        @tf.function(reduce_retracing=True)
        def compiled(x):
            return model(x, training=False)
        _compiled_calls[key] = (model, compiled)
    compiled = _compiled_calls[key][1]
    if isinstance(inputs, list):
        tensors = []
        for part in inputs:
            tensors.append(tf.constant(part))
        return compiled(tensors).numpy()
    return compiled(tf.constant(inputs)).numpy()


def is_two_input(model):
    """Mạng policy gradient có hai đầu vào, mạng policy thường có một."""
    model_inputs = getattr(model, "inputs", None)
    return model_inputs is not None and len(model_inputs) == 2


def move_probabilities(env, model):
    """Xác suất cho các cột HỢP LỆ, đã chuẩn hóa tổng bằng 1. Trả về dict {cột: xác suất}.

    Tự nhận ra mạng policy (một đầu vào) hay mạng policy gradient (hai đầu vào).
    """
    if is_two_input(model):
        output = call_model(model, [encode_flat(env), encode_conv(env)])
    else:
        output = call_model(model, encode_conv(env))
    output = np.squeeze(output)
    probs = {}
    total = 0.0
    for move in env.validinputs:
        p = float(output[move - 1])
        probs[move] = p
        total += p
    if total <= 0:
        # mạng dồn hết xác suất vào cột không hợp lệ: chia đều
        for move in probs:
            probs[move] = 1.0 / len(probs)
    else:
        for move in probs:
            probs[move] = probs[move] / total
    return probs


def value_estimate(env, value_model):
    """Giá trị thế cờ theo góc người sắp đi, trong [-1, 1].

    Mạng 3 lớp: P(thắng) - P(thua). Mạng một đầu ra (phiên bản 2): chính giá trị đó.
    """
    output = np.squeeze(call_model(value_model, encode_conv(env)))
    if output.size == 1:
        return float(output)
    return float(output[1] - output[2])


# Huấn luyện, đánh giá, lưu và nạp

def train_model(model, dataset, epochs=30, batch_size=128, verbose=1, patience=None):
    """Huấn luyện mạng trên tập train, theo dõi trên val, đánh giá trên test.

    dataset là dict tạo bởi utils.data.make_dataset (hoặc load_dataset).
    patience: nếu có, dừng sớm khi val_loss không giảm trong patience epoch liền,
    rồi lấy lại trọng số ở epoch có val_loss thấp nhất (chỉ nhìn val, không nhìn test).
    Trả về dict báo cáo: đường cong học, loss và accuracy trên val/test, số giây huấn luyện.
    """
    callbacks = []
    if patience is not None:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                       restore_best_weights=True))
    start = time.time()
    history = model.fit(dataset["X_train"], dataset["y_train"],
                        validation_data=(dataset["X_val"], dataset["y_val"]),
                        epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
    seconds = time.time() - start
    val_loss, val_metric = model.evaluate(dataset["X_val"], dataset["y_val"], verbose=0)
    test_loss, test_metric = model.evaluate(dataset["X_test"], dataset["y_test"], verbose=0)
    curve = {}
    for key, values in history.history.items():
        curve[key] = []
        for v in values:
            curve[key].append(float(v))
    # mạng phân lớp đo accuracy, mạng hồi quy (value phiên bản 2) đo sai số tuyệt đối trung bình (mae)
    metric = "accuracy" if "accuracy" in curve else "mae"
    # epoch có val_loss thấp nhất (đánh số từ 1), là trọng số được giữ lại khi dừng sớm
    best_epoch = int(np.argmin(curve["val_loss"])) + 1
    return {
        "epochs": epochs, "epochs_run": len(curve["loss"]), "best_epoch": best_epoch,
        "patience": patience, "batch_size": batch_size, "train_seconds": round(seconds, 1),
        "val_loss": float(val_loss), f"val_{metric}": float(val_metric),
        "test_loss": float(test_loss), f"test_{metric}": float(test_metric),
        "n_train": int(len(dataset["y_train"])), "n_val": int(len(dataset["y_val"])),
        "n_test": int(len(dataset["y_test"])), "mirror": bool(dataset.get("mirror", False)),
        "curve": curve,
    }


def train_value(games, path="files/value_conn.keras", epochs=30, batch_size=128,
                learning_rate=0.0005, seed=0, verbose=1, mirror=False, patience=None, labeled=None,
                target="outcome"):
    """Huấn luyện mạng value cho cấp 7 từ ván chuyên gia (kết quả của data.generate_games).

    target = "outcome": nhãn là kết quả cuối ván nhìn từ người sắp đi (hòa / thắng / thua);
    có labeled thì thế đã được tìm kiếm chứng minh thắng thua lấy kết quả đó.
    target = "search_score" (phiên bản 2): hồi quy điểm Alpha-Beta tìm sâu của mọi thế trong labeled,
    điểm lượng giá đổi sang thang kết quả ván bằng tanh(k × điểm), k chọn trên tập train (data.fit_score_scale).
    Chia train/val/test theo ván để không lẫn thế của cùng một ván sang hai tập.
    Trả về (model, report); mô hình lưu ở path kèm file báo cáo .json.
    """
    from utils import data
    score_scale = None
    scale_errors = None
    if target == "search_score":
        kind = "search_score"
        score_scale, scale_errors = data.fit_score_scale(games, labeled, seed=seed)
    elif labeled is None:
        kind = "value"
    else:
        kind = "search_value"
    dataset = data.make_dataset(games, kind=kind, seed=seed, mirror=mirror, labeled=labeled,
                                score_scale=score_scale)
    model = build_value_net(learning_rate=learning_rate, regression=(kind == "search_score"))
    report = train_model(model, dataset, epochs=epochs, batch_size=batch_size, verbose=verbose,
                         patience=patience)
    report["learning_rate"] = learning_rate
    report["kind"] = kind
    report["labels_replaced"] = dataset.get("replaced")
    if kind == "search_score":
        report["score_scale"] = score_scale
        # json chỉ nhận khóa chuỗi
        report["score_scale_errors"] = {str(k): v for k, v in scale_errors.items()}
        # mốc: luôn đoán trung bình của tập train
        mean_train = float(np.mean(dataset["y_train"]))
        report["baseline_test_mse"] = float(np.mean((dataset["y_test"] - mean_train) ** 2))
        report["baseline_test_mae"] = float(np.mean(np.abs(dataset["y_test"] - mean_train)))
    save_model(model, path, report)
    return model, report


def save_model(model, path, report=None):
    """Lưu mạng đuôi .keras, kèm file path + ".json" ghi báo cáo huấn luyện và cấu hình máy."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    model.save(path)
    meta = {"tensorflow": tf.__version__, "keras": keras.__version__, "machine": machine_info()}
    if report is not None:
        meta["report"] = report
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def load_model(path):
    return keras.models.load_model(path)

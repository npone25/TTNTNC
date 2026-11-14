"""Hàm phục vụ cho mạng: mã hóa trạng thái, tạo mạng policy, policy gradient, value; huấn luyện; lưu và nạp.

Kế thừa: kiến trúc ở notebook ch11 mục 2.1 và ch15 mục 1.1, 2.3, sách AlphaGo Simplified (Mark Liu), giấy phép MIT.
Sửa: chuyển sang Keras 3 (TensorFlow 2.21): dùng keras.Input thay tham số input_shape; nối hai đầu vào bằng layers.Concatenate thay tf.concat (Keras 3 báo lỗi với tf.concat); lưu đuôi .keras; nhãn dạng số nguyên (sparse) thay one-hot.
Thêm: đánh giá trên tập validation và test (sách không có); ghi đường cong học, thời gian huấn luyện và cấu hình máy; gọi mạng qua tf.function (nhanh hơn model.predict của sách khoảng 60 lần khi đánh giá từng thế cờ lẻ).
Phụ trách: Nguyễn Trọng Phúc.

Quy ước đầu vào: bàn cờ nhìn từ phía NGƯỜI SẮP ĐI (quân mình = 1, đối thủ = -1), hình dạng (7, 6, 1).
Đầu ra mạng value: 3 lớp [hòa, người sắp đi thắng, người sắp đi thua].
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


def build_value_net(learning_rate=0.0005):
    """Mạng value, cùng kiến trúc với value_model ở ch15 của sách (3 lớp đầu ra)."""
    model = keras.Sequential([
        keras.Input(shape=(7, 6, 1)),
        layers.Conv2D(filters=128, kernel_size=(4, 4), padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
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
    """Giá trị thế cờ theo góc người sắp đi, trong [-1, 1]: P(thắng) - P(thua)."""
    output = np.squeeze(call_model(value_model, encode_conv(env)))
    return float(output[1] - output[2])


# Huấn luyện, đánh giá, lưu và nạp

def train_model(model, dataset, epochs=30, batch_size=128, verbose=1):
    """Huấn luyện mạng trên tập train, theo dõi trên val, đánh giá trên test.

    dataset là dict tạo bởi utils.data.make_dataset (hoặc load_dataset).
    Trả về dict báo cáo: đường cong học, loss và accuracy trên val/test, số giây huấn luyện.
    """
    start = time.time()
    history = model.fit(dataset["X_train"], dataset["y_train"],
                        validation_data=(dataset["X_val"], dataset["y_val"]),
                        epochs=epochs, batch_size=batch_size, verbose=verbose)
    seconds = time.time() - start
    val_loss, val_acc = model.evaluate(dataset["X_val"], dataset["y_val"], verbose=0)
    test_loss, test_acc = model.evaluate(dataset["X_test"], dataset["y_test"], verbose=0)
    curve = {}
    for key, values in history.history.items():
        curve[key] = []
        for v in values:
            curve[key].append(float(v))
    return {
        "epochs": epochs, "batch_size": batch_size, "train_seconds": round(seconds, 1),
        "val_loss": float(val_loss), "val_accuracy": float(val_acc),
        "test_loss": float(test_loss), "test_accuracy": float(test_acc),
        "n_train": int(len(dataset["y_train"])), "n_val": int(len(dataset["y_val"])),
        "n_test": int(len(dataset["y_test"])),
        "curve": curve,
    }


def train_value(games, path="files/value_conn.keras", epochs=30, batch_size=128,
                learning_rate=0.0005, seed=0, verbose=1):
    """Huấn luyện mạng value cho cấp 7 từ ván chuyên gia (kết quả của data.generate_games).

    Nhãn của mỗi thế cờ là kết quả cuối ván nhìn từ người sắp đi (hòa / thắng / thua),
    chia train/val/test theo ván để không lẫn thế của cùng một ván sang hai tập.
    Trả về (model, report); mô hình lưu ở path kèm file báo cáo .json.
    """
    from utils import data
    dataset = data.make_dataset(games, kind="value", seed=seed)
    model = build_value_net(learning_rate=learning_rate)
    report = train_model(model, dataset, epochs=epochs, batch_size=batch_size, verbose=verbose)
    report["learning_rate"] = learning_rate
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

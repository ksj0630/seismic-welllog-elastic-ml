import os
import re
import glob
import time
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


SAVE_DIR   = Path("../06.model_save_folder/101.code")
OUTPUT_DIR = Path("../05.result/101.code")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGE_TRAIN_PATH = Path("../16.LSTM_merge_data_from_15.folder/00.X_train_1A_5_11T2_14_15A_for_density.txt")
X_APPLY_PATH     = Path("../15.LSTM_X_train_for_density/02.4_X_train_index+10.txt")
WHOLE_PATH       = Path("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")


def load_datasets(sequence=20):
    d1 = np.loadtxt(MERGE_TRAIN_PATH)
    d2 = np.loadtxt(X_APPLY_PATH)
    d3 = np.loadtxt(WHOLE_PATH)

    x_train_raw, y_train_raw = d1[:, :3], d1[:, 3]
    x_apply_raw, y_apply_raw = d2[:, :3], d2[:, 3]
    tvd3 = d3[:, 0]

    def make_windows(x, y, seq):
        X, Y = [], []
        half = seq // 2
        for i in range(0, len(x) - seq + 1, seq):
            X.append(x[i:i+seq])
            Y.append(y[i + half])
        return np.asarray(X), np.asarray(Y)

    X_train, Y_train = make_windows(x_train_raw, y_train_raw, sequence)
    X_test,  Y_test  = make_windows(x_apply_raw, y_apply_raw, sequence)
    R_train, Z_train = make_windows(x_apply_raw, y_apply_raw, sequence)

    return X_train, Y_train, X_test, Y_test, R_train, Z_train, tvd3


def build_lstm_mha(sequence):
    inp = layers.Input(shape=(sequence, 3))
    att = layers.MultiHeadAttention(num_heads=2, key_dim=3)(inp, inp)
    x   = layers.LSTM(128, activation="leaky_relu", return_sequences=True)(att)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(64, activation="leaky_relu", return_sequences=False)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32)(x)
    x   = layers.Dense(16)(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"]
    )
    return model


def find_last_checkpoint(save_dir: Path) -> int:
    files = glob.glob(str(save_dir / "model_*.hdf5"))
    if not files:
        return 0
    epochs = []
    for f in files:
        m = re.findall(r".*model_(\d+)\.hdf5.*", f)
        if m:
            epochs.append(int(m[0]))
    return max(epochs) if epochs else 0


def train(sequence=20, epochs=1000, batch_size=128):
    X_train, Y_train, X_test, Y_test, R_train, Z_train, tvd3 = load_datasets(sequence)
    model = build_lstm_mha(sequence)

    initial_epoch = find_last_checkpoint(SAVE_DIR)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print(f"Resuming from epoch {initial_epoch:03d}")
        model = tf.keras.models.load_model(SAVE_DIR / f"model_{initial_epoch:03d}.hdf5", compile=True)

    ckpt_all  = ModelCheckpoint(SAVE_DIR / "model_{epoch:03d}.hdf5", verbose=1, save_weights_only=False, save_freq="epoch")
    ckpt_best = ModelCheckpoint(SAVE_DIR / "best_model.hdf5", verbose=0, monitor="val_loss", save_best_only=True, mode="min", save_weights_only=False)
    csv_log   = CSVLogger(SAVE_DIR / "model_log.csv", append=True, separator=",")

    t0 = time.time()
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs, initial_epoch=initial_epoch,
        batch_size=batch_size, shuffle=True, verbose=1,
        callbacks=[ckpt_all, ckpt_best, csv_log],
    )
    print("Training time:", str(datetime.timedelta(seconds=time.time() - t0)).split(".")[0])

    pred_train = model.predict(X_train)
    pred_apply = model.predict(R_train)

    np.savetxt(OUTPUT_DIR / "20.merge_true.txt", Y_train)
    np.savetxt(OUTPUT_DIR / "21.merge_pred.txt", pred_train.squeeze())

    plt.figure(figsize=(20, 2))
    plt.plot(Y_train, label="True", color="black")
    plt.plot(pred_train, label="Pred", color="red")
    plt.legend(); plt.xlabel("Index"); plt.ylabel("Density (g/cm³)"); plt.grid(linestyle=":")
    plt.savefig(OUTPUT_DIR / "101.merge.png"); plt.close()

    np.savetxt(OUTPUT_DIR / "22.F4_true.txt", Z_train)
    np.savetxt(OUTPUT_DIR / "23.F4_pred.txt", pred_apply.squeeze())

    depth = tvd3[20:-20]
    plt.figure(figsize=(20, 2))
    plt.plot(depth, Z_train, label="True", color="black")
    plt.plot(depth, pred_apply, label="Pred", color="red")
    plt.legend(); plt.xlim(depth[0], depth[-1]); plt.ylim(2.05, 2.85)
    plt.xlabel("Depth (km)"); plt.ylabel("Density (g/cm³)"); plt.grid(linestyle=":")
    plt.savefig(OUTPUT_DIR / "101.apply.png"); plt.close()

    half = R_train.shape[1] // 2
    dvec = R_train[:, half, 0]
    triplet = np.column_stack([dvec, Z_train, pred_apply.squeeze()])
    np.savetxt(OUTPUT_DIR / "101.apply_triplet.txt", triplet, fmt="%.6f")


if __name__ == "__main__":
    train(sequence=20, epochs=1000, batch_size=128)

import os
import re
import glob
import time
import datetime
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# -----------------------------
# Paths & constants
# -----------------------------
SAVE_DIR = Path("../06.model_save_folder/10.code/")
OUT_DIR = Path("../05.result/10.code/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = Path("../04.merge_data/01.X_train_1A_5_11T2_14_15A.txt")
R_TRAIN_PATH = Path("../02.X_train/03.4_X_train_index+10.txt")
WHOLE_PATH   = Path("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")

SEQ_LEN     = 20
EPOCHS      = 1000
BATCH_SIZE  = 128
LR          = 1e-4
GP_RESTARTS = 20
GP_ALPHA    = 2e-1


# -----------------------------
# Helpers
# -----------------------------
def find_last_checkpoint(save_dir: Path) -> int:
    files = glob.glob(str(save_dir / "model_*.hdf5"))
    if not files:
        return 0
    epochs = []
    for f in files:
        m = re.findall(r".*model_(\d+)\.hdf5$", f)
        if m:
            epochs.append(int(m[0]))
    return max(epochs) if epochs else 0


def build_sequences(x: np.ndarray, y: np.ndarray, seq_len: int):
    X, Y = [], []
    for idx in range(0, len(x) - seq_len + 1, seq_len):
        X.append(x[idx: idx + seq_len])
        Y.append(y[idx + (seq_len // 2)])
    return np.asarray(X), np.asarray(Y)


def load_data():
    d1 = np.loadtxt(X_TRAIN_PATH)
    x_train = np.column_stack([d1[:, 0], d1[:, 1], d1[:, 3]])
    y_train = d1[:, 6]

    d2 = np.loadtxt(R_TRAIN_PATH)
    r_train = np.column_stack([d2[:, 0], d2[:, 1], d2[:, 3]])
    z_train = d2[:, 6]

    d3 = np.loadtxt(WHOLE_PATH)
    tvd3, truee, init3 = d3[:, 0], d3[:, 6], d3[:, 1]

    return (x_train, y_train), (r_train, z_train), (tvd3, truee, init3)


def make_model(seq_len: int) -> tf.keras.Model:
    x_in = Input(shape=(seq_len, 3))
    attn = MultiHeadAttention(num_heads=2, key_dim=3)(x_in, x_in, x_in)
    x = LSTM(128, activation="leaky_relu", return_sequences=True)(attn)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation="leaky_relu", return_sequences=False)(x)
    x = Dense(64)(x)
    x = Dense(32)(x)
    x = Dense(16)(x)
    y_out = Dense(1)(x)
    model = Model(inputs=x_in, outputs=y_out)
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=["accuracy"])
    return model


def plot_train(y_true, y_pred, y_std, out_png: Path):
    plt.figure(figsize=(20, 2))
    plt.plot(y_true, label="True", linewidth=1.0, linestyle="solid", color="black")
    plt.plot(y_pred, label="Pred", linewidth=1.0, linestyle="solid", color="red")
    plt.fill_between(range(len(y_pred)), y_pred - 2 * y_std, y_pred + 2 * y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend()
    plt.xlabel("Depth (km)")
    plt.ylabel("P-wave velocity (km/s)")
    plt.grid(linestyle=":")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_test(depth, y_true, y_pred, y_std, initial_curve, out_png: Path):
    plt.figure(figsize=(20, 2))
    plt.plot(depth, y_true, label="True", linewidth=1.0, linestyle="solid", color="black")
    plt.plot(depth, y_pred, label="Pred", linewidth=1.0, linestyle="solid", color="red")
    plt.plot(depth, initial_curve, label="initial", linewidth=1.0, linestyle="dashed", color="blue")
    plt.fill_between(depth, y_pred - 2 * y_std, y_pred + 2 * y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend()
    plt.xlim(depth[0], depth[-1])
    plt.ylim([1, 5.65])
    plt.xlabel("Depth (km)")
    plt.ylabel("P-wave velocity (km/s)")
    plt.grid(linestyle=":")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(epochs=EPOCHS, batch_size=BATCH_SIZE, gpiter=GP_RESTARTS):
    (x_train, y_train), (r_train, z_train), (_, _, init3) = load_data()

    X_train, Y_train = build_sequences(x_train, y_train, SEQ_LEN)
    X_test,  Y_test  = build_sequences(r_train, z_train, SEQ_LEN)

    model = make_model(SEQ_LEN)

    initial_epoch = find_last_checkpoint(SAVE_DIR)
    if initial_epoch > 0:
        model = tf.keras.models.load_model(SAVE_DIR / f"model_{initial_epoch:03d}.hdf5", compile=True)

    save_each  = ModelCheckpoint(SAVE_DIR / "model_{epoch:03d}.hdf5", verbose=1, save_weights_only=False, save_freq="epoch")
    save_best  = ModelCheckpoint(SAVE_DIR / "best_model.hdf5", verbose=0, monitor="val_loss", save_best_only=True, mode="min")
    csv_logger = CSVLogger(SAVE_DIR / "model_log.csv", append=True, separator=",")

    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_split=0.2,
        initial_epoch=initial_epoch,
        callbacks=[save_each, save_best, csv_logger],
    )

    feat_extractor = Model(inputs=model.input, outputs=model.layers[-4].output)
    feats_train = feat_extractor.predict(X_train)
    feats_test  = feat_extractor.predict(X_test)

    gp_path = SAVE_DIR / "gaussian_process_model.pkl"
    if initial_epoch == epochs and gp_path.exists():
        gp = joblib.load(gp_path)
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gpiter, alpha=GP_ALPHA)
        gp.fit(feats_train, Y_train)
        joblib.dump(gp, gp_path)

    y_pred_tr, y_std_tr = gp.predict(feats_train, return_std=True)
    y_pred_te, y_std_te = gp.predict(feats_test, return_std=True)

    np.savetxt(OUT_DIR / "20.merge_true.txt",       Y_train)
    np.savetxt(OUT_DIR / "21.merge_pred.txt",       y_pred_tr)
    np.savetxt(OUT_DIR / "211.merge_standard.txt",  y_std_tr)

    plot_train(Y_train, y_pred_tr, y_std_tr, OUT_DIR / "10.merge_pred.png")

    half = X_test.shape[1] // 2
    depth = X_test[:, half, 0]
    pred_pack = np.column_stack([depth, Y_test, y_pred_te])
    np.savetxt(OUT_DIR / "10.pred_dtc.txt", pred_pack, delimiter=" ")

    np.savetxt(OUT_DIR / "22.F4_true.txt",      Y_test)
    np.savetxt(OUT_DIR / "23.F4_pred.txt",      y_pred_te)
    np.savetxt(OUT_DIR / "233.F4_standard.txt", y_std_te)

    plot_test(depth, Y_test, y_pred_te, y_std_te, init3[10:-10], OUT_DIR / "10.pred.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = str(datetime.timedelta(seconds=time.time() - t0)).split(".")[0]
    print("Elapsed:", dt)

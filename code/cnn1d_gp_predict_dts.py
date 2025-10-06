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
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# -----------------------------
# Config
# -----------------------------
SAVE_DIR   = Path("../06.model_save_folder/58.code/")
OUTPUT_DIR = Path("../05.result/58.code/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GBTC_PATH = Path("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")
TRAIN_X_PATH = Path("../30.1DCNN_GP_train_data_for_dts/x_train_dts.npy")
TRAIN_Y_PATH = Path("../30.1DCNN_GP_train_data_for_dts/y_train_dts.npy")
APPLY_X_PATH = Path("../30.1DCNN_GP_train_data_for_dts/x_test_dts.npy")
APPLY_Y_PATH = Path("../30.1DCNN_GP_train_data_for_dts/y_test_dts.npy")

EPOCHS      = 150
BATCH_SIZE  = 128
LR          = 1e-4
GP_ALPHA    = 2e-1
GP_RESTARTS = 20
N_CH        = 3


# -----------------------------
# Utils
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


class SaveLast(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params["epochs"] - 1:
            self.model.save(SAVE_DIR / "last_epoch_model.hdf5")


def load_arrays(x_path: Path, y_path: Path):
    X = np.load(x_path)
    y = np.load(y_path)
    nvert = X.shape[1]
    half = nvert // 2
    X_scaled = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y_central = y[:, half]
    return X_scaled, y_central


def load_data():
    X_tr_all, y_tr_all = load_arrays(TRAIN_X_PATH, TRAIN_Y_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tr_all, y_tr_all, test_size=0.1, random_state=42, shuffle=False
    )
    X_apply, y_apply = load_arrays(APPLY_X_PATH, APPLY_Y_PATH)
    gbtc = np.loadtxt(GBTC_PATH)
    tvd3 = gbtc[:, 0]
    return (X_tr, y_tr), (X_te, y_te), (X_apply, y_apply), tvd3


# -----------------------------
# Model
# -----------------------------
def build_cnn1d(n1: int, n_ch: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(n1, n_ch))

    x = layers.Conv1D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    x = layers.Dense(32, activation="leaky_relu")(x)
    x = layers.Dense(16, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Plots
# -----------------------------
def plot_merge_pred(y_true, y_pred, y_std, out_png: Path, ylabel: str):
    plt.figure(figsize=(20, 2))
    plt.plot(y_true, label="True", linewidth=1.0, linestyle="solid", color="black")
    plt.plot(y_pred, label="Pred", linewidth=1.0, linestyle="solid", color="red")
    plt.fill_between(range(len(y_pred)), y_pred - 2 * y_std, y_pred + 2 * y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend()
    plt.xlabel("Depth (km)")
    plt.ylabel(ylabel)
    plt.grid(linestyle=":")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_profile(depth, y_true, y_pred, y_std, out_png: Path, ylabel: str, ylim=None, crop=8):
    plt.figure(figsize=(20, 2))
    if crop and crop > 0:
        depth = depth[:-crop]
        y_true = y_true[:-crop]
        y_pred = y_pred[:-crop]
        y_std  = y_std[:-crop]
    plt.plot(depth, y_true, label="True", linewidth=1.0, linestyle="solid", color="black")
    plt.plot(depth, y_pred, label="Pred", linewidth=1.0, linestyle="solid", color="red")
    plt.fill_between(depth, y_pred - 2 * y_std, y_pred + 2 * y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend()
    plt.xlim(depth[0], depth[-1])
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Depth (km)")
    plt.ylabel(ylabel)
    plt.grid(linestyle=":")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# -----------------------------
# Train + GP + Predict
# -----------------------------
def main(n_ch=N_CH, epochs=EPOCHS, batch_size=BATCH_SIZE, gpiter=GP_RESTARTS):
    (X_train, y_train), (X_test, y_test), (X_apply, y_apply), tvd3 = load_data()

    model = build_cnn1d(X_train.shape[1], n_ch)

    initial_epoch = find_last_checkpoint(SAVE_DIR)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print(f"Resuming from epoch {initial_epoch:03d}")
        model = models.load_model(SAVE_DIR / f"model_{initial_epoch:03d}.hdf5", compile=False)
        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=["accuracy"])

    cb_each = ModelCheckpoint(SAVE_DIR / "model_{epoch:03d}.hdf5", verbose=1, save_weights_only=False, save_freq="epoch")
    cb_best = ModelCheckpoint(SAVE_DIR / "best_model.hdf5", verbose=0, monitor="val_loss", save_best_only=True, mode="min")
    cb_csv  = CSVLogger(SAVE_DIR / "model_log.csv", append=True, separator=",")
    cb_last = SaveLast()

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[cb_each, cb_best, cb_csv, cb_last],
        verbose=1,
    )

    feat_extractor = Model(inputs=model.input, outputs=model.layers[-6].output)
    feats_test  = feat_extractor.predict(X_test)
    feats_apply = feat_extractor.predict(X_apply)

    gp_path = SAVE_DIR / "gaussian_process_model.pkl"
    if initial_epoch == epochs and gp_path.exists():
        gp = joblib.load(gp_path)
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gpiter, alpha=GP_ALPHA)
        gp.fit(feat_extractor.predict(X_train), y_train)
        joblib.dump(gp, gp_path)

    y_pred_te, y_std_te = gp.predict(feats_test,  return_std=True)
    y_pred_ap, y_std_ap = gp.predict(feats_apply, return_std=True)

    np.savetxt(OUTPUT_DIR / "20.merge_true.txt",      y_test)
    np.savetxt(OUTPUT_DIR / "21.merge_pred.txt",      y_pred_te)
    np.savetxt(OUTPUT_DIR / "211.merge_standard.txt", y_std_te)
    plot_merge_pred(y_test, y_pred_te, y_std_te, OUTPUT_DIR / "58.merge_pred.png", ylabel="S-wave velocity (km/s)")

    half = X_apply.shape[1] // 2
    depth = X_apply[:, half, 0]
    pack = np.column_stack([depth, y_apply, y_pred_ap])
    np.savetxt(OUTPUT_DIR / "58.pred_dtc.txt", pack, delimiter=" ")

    np.savetxt(OUTPUT_DIR / "22.F4_true.txt",      y_apply)
    np.savetxt(OUTPUT_DIR / "23.F4_pred.txt",      y_pred_ap)
    np.savetxt(OUTPUT_DIR / "233.F4_standard.txt", y_std_ap)
    plot_profile(depth, y_apply, y_pred_ap, y_std_ap, OUTPUT_DIR / "58.pred.png", ylabel="S-wave velocity (km/s)", ylim=(1.2, 2.9), crop=8)


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = str(datetime.timedelta(seconds=time.time() - t0)).split(".")[0]
    print("Elapsed:", dt)

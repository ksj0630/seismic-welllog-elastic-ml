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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


# -----------------------------
# Paths
# -----------------------------
SAVE_DIR   = Path("../06.model_save_folder/23.code")
OUTPUT_DIR = Path("../05.result/23.code")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WELL_ALL_PATH = Path("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")
X_TRAIN_PATH  = Path("../13.1DCNN_input_from_prof/training/x_train.npy")
Y_TRAIN_PATH  = Path("../13.1DCNN_input_from_prof/training/y_train.npy")
X_TEST_PATH   = Path("../13.1DCNN_input_from_prof/training/x_test.npy")
Y_TEST_PATH   = Path("../13.1DCNN_input_from_prof/training/y_test.npy")


# -----------------------------
# Utils
# -----------------------------
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


class Dataset:
    def __init__(self, data_file, apply_file):
        X = np.load(data_file[0])
        y = np.load(data_file[1])

        nvert = X.shape[1]
        half = nvert // 2

        X_scaled = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        y_half   = y[:, half]

        # fixed split (shuffle=False) to match original behavior
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_half, test_size=0.1, random_state=42, shuffle=False
        )

        X2 = np.load(apply_file[0])
        y2 = np.load(apply_file[1])
        self.X_apply = X2.reshape(X2.shape[0], X2.shape[1], X2.shape[2])
        self.y_apply = y2[:, half]


def build_cnn1d(n1: int, n_ch: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n1, n_ch))
    x = layers.Conv1D(64, 3, padding="same")(inp); x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Conv1D(64, 3, padding="same")(x);   x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Conv1D(128, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Conv1D(256, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64,  activation="leaky_relu")(x)
    x = layers.Dense(32,  activation="leaky_relu")(x)
    x = layers.Dense(16,  activation="leaky_relu")(x)
    out = layers.Dense(1,  activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])
    return model


def main(n_ch=3, epochs=107, batch_size=128):
    well_all = np.loadtxt(WELL_ALL_PATH)
    tvd3  = well_all[:, 0]
    init3 = well_all[:, 1]

    data = Dataset(
        data_file=[X_TRAIN_PATH, Y_TRAIN_PATH],
        apply_file=[X_TEST_PATH, Y_TEST_PATH],
    )

    model = build_cnn1d(data.X_train.shape[1], n_ch)

    initial_epoch = find_last_checkpoint(SAVE_DIR)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print(f"Resuming from epoch {initial_epoch:03d}")
        model = tf.keras.models.load_model(SAVE_DIR / f"model_{initial_epoch:03d}.hdf5", compile=False)
        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])

    ckpt_every = ModelCheckpoint(SAVE_DIR / "model_{epoch:03d}.hdf5", verbose=1, save_weights_only=False, save_freq="epoch")
    ckpt_best  = ModelCheckpoint(SAVE_DIR / "best_model.hdf5", verbose=0, monitor="val_loss",
                                 save_best_only=True, mode="min", save_weights_only=False)
    csv_logger = CSVLogger(SAVE_DIR / "model_log.csv", append=True, separator=",")

    t0 = time.time()
    model.fit(
        data.X_train, data.y_train,
        epochs=epochs, initial_epoch=initial_epoch,
        batch_size=batch_size, validation_split=0.1,
        callbacks=[ckpt_every, ckpt_best, csv_logger], shuffle=True, verbose=1
    )
    print("Training time:", str(datetime.timedelta(seconds=time.time() - t0)).split(".")[0])

    # feature extractor: one dense layer before the output
    feat_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    feat_train = feat_extractor.predict(data.X_train)
    feat_test  = feat_extractor.predict(data.X_test)
    feat_apply = feat_extractor.predict(data.X_apply)

    gp_path = SAVE_DIR / "gaussian_process_model.pkl"
    if initial_epoch == epochs and gp_path.exists():
        gp = joblib.load(gp_path)
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=2e-1)
        gp.fit(feat_train, data.y_train)
        joblib.dump(gp, gp_path)

    y_pred,  y_std  = gp.predict(feat_test,  return_std=True)
    y_pred2, y_std2 = gp.predict(feat_apply, return_std=True)

    # save arrays
    np.savetxt(OUTPUT_DIR / "20.merge_true.txt", data.y_test)
    np.savetxt(OUTPUT_DIR / "21.merge_pred.txt", y_pred)
    np.savetxt(OUTPUT_DIR / "211.merge_standard.txt", y_std)

    # plot: validation
    plt.figure(figsize=(20, 2))
    plt.plot(data.y_test, label="True", linewidth=1.0, color="black")
    plt.plot(y_pred,      label="Pred", linewidth=1.0, color="red")
    xs = np.arange(len(y_pred))
    plt.fill_between(xs, y_pred - 2*y_std, y_pred + 2*y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend(); plt.xlabel("Index"); plt.ylabel("P-wave velocity (km/s)"); plt.grid(linestyle=":")
    plt.savefig(OUTPUT_DIR / "23.merge_pred.png"); plt.close()

    # plot: apply
    np.savetxt(OUTPUT_DIR / "22.F4_true.txt", data.y_apply)
    np.savetxt(OUTPUT_DIR / "23.F4_pred.txt", y_pred2)
    np.savetxt(OUTPUT_DIR / "233.F4_standard.txt", y_std2)

    plt.figure(figsize=(20, 2))
    depth = tvd3[10:-10]
    plt.plot(depth, data.y_apply,  label="True", linewidth=1.0, color="black")
    plt.plot(depth, y_pred2,       label="Pred", linewidth=1.0, color="red")
    plt.plot(depth, init3[10:-10], label="initial", linewidth=1.0, linestyle="--", color="blue")
    plt.fill_between(depth, y_pred2 - 2*y_std2, y_pred2 + 2*y_std2, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend(); plt.xlim([depth[0], depth[-1]]); plt.ylim([1, 5.75])
    plt.xlabel("Depth (km)"); plt.ylabel("P-wave velocity (km/s)"); plt.grid(linestyle=":")
    plt.savefig(OUTPUT_DIR / "23.pred.png"); plt.close()

    # save (depth, true, pred)
    half = data.X_apply.shape[1] // 2
    dvec = data.X_apply[:, half, 0]
    pred_pack = np.column_stack([dvec, data.y_apply, y_pred2])
    np.savetxt(OUTPUT_DIR / "23.pred_dtc.txt", pred_pack, fmt="%.6f")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="1D CNN + GP for P-wave velocity")
    p.add_argument("-input_channels", type=int, default=3)
    p.add_argument("-epochs",         type=int, default=107)
    p.add_argument("-batch_size",     type=int, default=128)
    args = p.parse_args()

    main(n_ch=args.input_channels, epochs=args.epochs, batch_size=args.batch_size)

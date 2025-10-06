import os
import glob
import re
import time
import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# -----------------------------
# Paths & I/O
# -----------------------------
SAVE_DIR   = os.path.join('../06.model_save_folder/43.code/')
OUTPUT_DIR = os.path.join("../05.result/43.code/")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

gbt = np.loadtxt("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")
tvd3 = gbt[:, 0]  # for plotting against depth

# -----------------------------
# Utilities
# -----------------------------
def find_last_checkpoint(save_dir: str) -> int:
    files = glob.glob(os.path.join(save_dir, "model_*.hdf5"))
    if not files:
        return 0
    epochs = []
    for f in files:
        m = re.findall(r".*model_(\d+)\.hdf5$", f)
        if m:
            epochs.append(int(m[0]))
    return max(epochs) if epochs else 0


class DataReader:
    def __init__(self, data_file, apply_file):
        X = np.load(data_file[0])
        y = np.load(data_file[1])

        nvert = X.shape[1]
        half  = nvert // 2

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        y = y[:, half]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=False
        )

        X2 = np.load(apply_file[0])
        y2 = np.load(apply_file[1])
        X2 = X2.reshape(X2.shape[0], X2.shape[1], X2.shape[2])
        self.X_apply = X2
        self.y_apply = y2[:, half]


# -----------------------------
# Model
# -----------------------------
def build_cnn1d(n1: int, n_ch: int, lr: float = 1e-4) -> tf.keras.Model:
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
    x = layers.Dense(64,  activation="leaky_relu")(x)
    x = layers.Dense(32,  activation="leaky_relu")(x)
    x = layers.Dense(16,  activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
    return model


# -----------------------------
# Plot helpers
# -----------------------------
def plot_merge(y_true, y_pred, y_std, out_png: str, ylabel: str):
    plt.figure(figsize=(20, 2))
    plt.plot(y_true, label="True", linewidth=1.0, color="black")
    plt.plot(y_pred, label="Pred", linewidth=1.0, color="red")
    if y_std is not None:
        x = np.arange(len(y_pred))
        plt.fill_between(x, y_pred - 2 * y_std, y_pred + 2 * y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend(); plt.xlabel("Index"); plt.ylabel(ylabel); plt.grid(linestyle=":")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()


def plot_profile(depth, y_true, y_pred, y_std, out_png: str, ylabel: str, ylim=None):
    plt.figure(figsize=(20, 2))
    plt.plot(depth, y_true, label="True", linewidth=1.0, color="black")
    plt.plot(depth, y_pred, label="Pred", linewidth=1.0, color="red")
    if y_std is not None:
        plt.fill_between(depth, y_pred - 2 * y_std, y_pred + 2 * y_std, color="gray", alpha=0.2, label="Uncertainty")
    plt.legend(); plt.xlim(depth[0], depth[-1])
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Depth (km)"); plt.ylabel(ylabel); plt.grid(linestyle=":")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()


# -----------------------------
# Train + GP + Predict
# -----------------------------
def main(n_ch=3, epochs=1000, batch_size=128, gpiter=20, structure='CNN1D'):
    data_file  = ['../29.1DCNN_GP_train_data_for_density/x_train_rhob.npy',
                  '../29.1DCNN_GP_train_data_for_density/y_train_rhob.npy']
    apply_file = ['../29.1DCNN_GP_train_data_for_density/x_test_rhob.npy',
                  '../29.1DCNN_GP_train_data_for_density/y_test_rhob.npy']

    data = DataReader(data_file=data_file, apply_file=apply_file)

    if structure == 'CNN1D':
        model = build_cnn1d(data.X_train.shape[1], n_ch)

    initial_epoch = find_last_checkpoint(SAVE_DIR)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print(f"Resuming from epoch {initial_epoch:03d}")
        model = models.load_model(os.path.join(SAVE_DIR, f"model_{initial_epoch:03d}.hdf5"), compile=False)
        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])

    cb_each = ModelCheckpoint(os.path.join(SAVE_DIR, "model_{epoch:03d}.hdf5"),
                              verbose=1, save_weights_only=False, save_freq="epoch")
    cb_best = ModelCheckpoint(os.path.join(SAVE_DIR, "best_model.hdf5"),
                              verbose=0, monitor="val_loss", save_best_only=True, mode="min")
    cb_csv  = CSVLogger(os.path.join(SAVE_DIR, "model_log.csv"), append=True, separator=",")

    model.fit(
        data.X_train, data.y_train,
        epochs=epochs, initial_epoch=initial_epoch,
        batch_size=batch_size, validation_split=0.1,
        callbacks=[cb_each, cb_best, cb_csv],
        shuffle=True, verbose=1,
    )

    # Feature extraction → GP training
    feat_extractor = Model(inputs=model.input, outputs=model.layers[-6].output)
    feats_train = feat_extractor.predict(data.X_train)
    feats_test  = feat_extractor.predict(data.X_test)
    feats_apply = feat_extractor.predict(data.X_apply)

    gp_path = os.path.join(SAVE_DIR, 'gaussian_process_model.pkl')
    if initial_epoch == epochs and os.path.isfile(gp_path):
        gp = joblib.load(gp_path)
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gpiter, alpha=2e-1)
        gp.fit(feats_train, data.y_train)
        joblib.dump(gp, gp_path)

    y_pred_te, y_std_te = gp.predict(feats_test,  return_std=True)
    y_pred_ap, y_std_ap = gp.predict(feats_apply, return_std=True)

    # Save & plots (test)
    np.savetxt(os.path.join(OUTPUT_DIR, "20.merge_true.txt"),      data.y_test)
    np.savetxt(os.path.join(OUTPUT_DIR, "21.merge_pred.txt"),      y_pred_te)
    np.savetxt(os.path.join(OUTPUT_DIR, "211.merge_standard.txt"), y_std_te)
    plot_merge(data.y_test, y_pred_te, y_std_te,
               os.path.join(OUTPUT_DIR, '43.merge_pred.png'),
               ylabel="Density (g/cm3)")

    # Save & plots (apply/inference)
    half  = data.X_apply.shape[1] // 2
    depth = data.X_apply[:, half, 0]
    pack  = np.column_stack([depth, data.y_apply, y_pred_ap])
    np.savetxt(os.path.join(OUTPUT_DIR, "43.pred_dtc.txt"), pack, delimiter=' ')

    np.savetxt(os.path.join(OUTPUT_DIR, "22.F4_true.txt"),      data.y_apply)
    np.savetxt(os.path.join(OUTPUT_DIR, "23.F4_pred.txt"),      y_pred_ap)
    np.savetxt(os.path.join(OUTPUT_DIR, "233.F4_standard.txt"), y_std_ap)
    plot_profile(depth, data.y_apply, y_pred_ap, y_std_ap,
                 os.path.join(OUTPUT_DIR, '43.pred.png'),
                 ylabel="Density (g/cm3)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='1D CNN + GP for Density (RHOB)')
    parser.add_argument('-input_channels', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-gpiter', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-structure', type=str, default='CNN1D')
    args = parser.parse_args()

    start = time.time()
    main(args.input_channels, args.epochs, args.batch_size, args.gpiter, args.structure)
    sec = time.time() - start
    print("Elapsed:", str(datetime.timedelta(seconds=sec)).split(".")[0])

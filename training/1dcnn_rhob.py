import os
import glob
import re
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split


# -----------------------------
# Paths
# -----------------------------
SAVE_DIR   = "../06.model_save_folder/114.code/"
OUTPUT_DIR = "../05.result/114.code/"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

gbt = np.loadtxt("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")
tvd3  = gbt[:, 0]
truee = gbt[:, 5]


# -----------------------------
# Utils
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
            X, y, test_size=0.1, random_state=42, shuffle=True
        )

        X2 = np.load(apply_file[0])
        y2 = np.load(apply_file[1])
        X2 = X2.reshape(X2.shape[0], X2.shape[1], X2.shape[2])
        self.X_apply = X2
        self.y_apply = y2[:, half]


# -----------------------------
# Model
# -----------------------------
def build_cnn1d(n1: int, n_ch: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(n1, n_ch))

    x = Conv1D(64, 7, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv1D(64, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(2, 2)(x)

    x = Conv1D(128, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv1D(128, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(2, 2)(x)

    x = Conv1D(256, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv1D(256, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation="leaky_relu")(x)
    x = Dense(64,  activation="leaky_relu")(x)
    x = Dense(32,  activation="leaky_relu")(x)
    x = Dense(16,  activation="leaky_relu")(x)
    outputs = Dense(1, activation="leaky_relu")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])
    return model


# -----------------------------
# Train + Predict
# -----------------------------
def main(n_ch=3, epochs=1000, batch_size=128, structure="CNN1D"):
    data_file  = ["../17.1DCNN_train_data_for_density/x_train_rhob.npy",
                  "../17.1DCNN_train_data_for_density/y_train_rhob.npy"]
    apply_file = ["../17.1DCNN_train_data_for_density/x_test_rhob.npy",
                  "../17.1DCNN_train_data_for_density/y_test_rhob.npy"]

    data = DataReader(data_file=data_file, apply_file=apply_file)

    if structure == "CNN1D":
        model = build_cnn1d(data.X_train.shape[1], n_ch)

    initial_epoch = find_last_checkpoint(SAVE_DIR)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print(f"Resuming from epoch {initial_epoch:03d}")
        model = models.load_model(os.path.join(SAVE_DIR, f"model_{initial_epoch:03d}.hdf5"), compile=False)
        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])

    cb_each = ModelCheckpoint(os.path.join(SAVE_DIR, "model_{epoch:03d}.hdf5"), verbose=1, save_weights_only=False, save_freq="epoch")
    cb_best = ModelCheckpoint(os.path.join(SAVE_DIR, "best_model.hdf5"), verbose=0, monitor="val_loss", save_best_only=True, mode="min")
    cb_csv  = CSVLogger(os.path.join(SAVE_DIR, "model_log.csv"), append=True, separator=",")

    model.fit(
        data.X_train, data.y_train,
        epochs=epochs, initial_epoch=initial_epoch,
        batch_size=batch_size, validation_split=0.1,
        callbacks=[cb_each, cb_best, cb_csv],
        shuffle=True, verbose=1,
    )

    y_pred  = model.predict(data.X_test)
    y_pred2 = model.predict(data.X_apply)

    np.savetxt(os.path.join(OUTPUT_DIR, "20.merge_true.txt"), data.y_test)
    np.savetxt(os.path.join(OUTPUT_DIR, "21.merge_pred.txt"), y_pred)

    plt.figure(figsize=(20, 2))
    plt.plot(data.y_test, label="True", linewidth=1.0, color="black")
    plt.plot(y_pred,      label="Pred", linewidth=1.0, color="red")
    plt.legend(); plt.grid(linestyle=":")
    plt.savefig(os.path.join(OUTPUT_DIR, "114.merge_pred.png")); plt.close()

    np.savetxt(os.path.join(OUTPUT_DIR, "22.F4_true.txt"), data.y_apply)
    np.savetxt(os.path.join(OUTPUT_DIR, "23.F4_pred.txt"), y_pred2)

    plt.figure(figsize=(20, 2))
    plt.plot(tvd3[20:-20], data.y_apply, label="True", linewidth=1.0, color="black")
    plt.plot(tvd3[20:-20], y_pred2,      label="Pred", linewidth=1.0, color="red")
    plt.legend(); plt.xlim(tvd3[20], tvd3[-20]); plt.ylim([2.05, 2.85])
    plt.xlabel("Depth (km)"); plt.ylabel("Density (g/cm3)"); plt.grid(linestyle=":")
    plt.savefig(os.path.join(OUTPUT_DIR, "114.pred.png")); plt.close()

    half  = data.X_apply.shape[1] // 2
    depth = data.X_apply[:, half, 0]
    pred_pack = np.column_stack([depth, data.y_apply, y_pred2.squeeze()])
    np.savetxt(os.path.join(OUTPUT_DIR, "114.pred_dtc.txt"), pred_pack, delimiter=" ")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="1D CNN for Density (RHOB)")
    parser.add_argument("-input_channels", type=int, default=3)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-model", type=str, default="CNN1D")
    parser.add_argument("-structure", type=str, default="CNN1D")
    args = parser.parse_args()

    t0 = time.time()
    main(args.input_channels, args.epochs, args.batch_size, args.structure)
    dt = str(datetime.timedelta(seconds=time.time() - t0)).split(".")[0]
    print("Elapsed:", dt)

import os
import glob
import re
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, Input
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ================================
# Directories
# ================================
save_dir = os.path.join('../06.model_save_folder/154.code/')
os.makedirs(save_dir, exist_ok=True)

output_dir = os.path.join("../05.result/154.code/")
os.makedirs(output_dir, exist_ok=True)

# ================================
# Load data
# ================================
data1 = np.loadtxt("../26.LSTM_merge_data_from_25.folder/00.merge_data_1A_11T2_14.txt")
xx = data1[:, :3]
yy = data1[:, 3]
tvd1 = data1[:, 0]

data2 = np.loadtxt('../25.LSTM_X_train_for_dts/02.4_X_train_index+10.txt')
r_train = data2[:, :3]
z_train = data2[:, 3]

data3 = np.loadtxt('../24.LSTM_train_data_for_dts/02.4_for_dts.txt')
tvd3 = data3[:, 0]
truee = data3[:, 3]

x_train, x_test = xx, r_train
y_train, y_test = yy, z_train

# ================================
# Sequence preparation
# ================================
sequence = 20

def to_sequences(x, y):
    X, Y = [], []
    for i in range(0, len(x) - sequence + 1, sequence):
        X.append(x[i: i + sequence])
        Y.append(y[i + 10])
    return np.asarray(X), np.asarray(Y)

X_train, Y_train = to_sequences(x_train, y_train)
X_test, Y_test   = to_sequences(x_test, y_test)
R_train, Z_train = to_sequences(r_train, z_train)

print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
print('X_test:',  X_test.shape,  'Y_test:',  Y_test.shape)
print('R_train:', R_train.shape, 'Z_train:', Z_train.shape)

# ================================
# Model
# ================================
def build_model():
    inp = Input(shape=(sequence, 3))
    attn = MultiHeadAttention(num_heads=2, key_dim=3)(inp, inp, inp)
    x = LSTM(128, activation='leaky_relu', return_sequences=True)(attn)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation='leaky_relu', return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='leaky_relu')(x)
    x = Dense(16, activation='leaky_relu')(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mae', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.summary()
    return model

# ================================
# Eval / plots
# ================================
def evaluate_and_plot(model):
    pred_train = model.predict(X_train)
    pred_apply = model.predict(R_train)

    # Train-like segment (merged)
    np.savetxt(os.path.join(output_dir, "20.merge_true.txt"), Y_train)
    np.savetxt(os.path.join(output_dir, "21.merge_pred.txt"), pred_train)
    plt.figure(figsize=(20, 2))
    plt.plot(Y_train, label="True", color="black", linewidth=1.0)
    plt.plot(pred_train, label="Pred", color="red", linewidth=1.0)
    plt.legend(); plt.grid(linestyle=':')
    plt.xlabel("Index")
    plt.ylabel("S-wave velocity (km/s)")
    plt.savefig(os.path.join(output_dir, '154.merge_code.png'))

    # Apply segment
    np.savetxt(os.path.join(output_dir, "22.F4_true.txt"), Z_train, fmt='%.6f')
    np.savetxt(os.path.join(output_dir, "23.F4_pred.txt"), pred_apply, fmt='%.6f')

    plt.figure(figsize=(20, 2))
    # Trim tail with NaNs/gaps if needed: original used [-18] and [-8] offsets
    plt.plot(tvd3[10:-18], Z_train[0:-8], label="True", color="black", linewidth=1.0)
    plt.plot(tvd3[10:-18], pred_apply[0:-8], label="Pred", color="red", linewidth=1.0)
    plt.legend()
    plt.xlim([tvd3[10], tvd3[-18]])
    plt.ylim([1.2, 2.9])
    plt.xlabel("Depth (km)")
    plt.ylabel("S-wave velocity (km/s)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '154.code.png'))

# ================================
# Training runner
# ================================
def main(epochs=1000, batch_size=128):
    model = build_model()

    def find_last_checkpoint(d):
        files = glob.glob(os.path.join(d, 'model_*.hdf5'))
        if not files:
            return 0
        return max(int(re.findall(r".*model_(\d+).hdf5.*", f)[0]) for f in files)

    initial_epoch = find_last_checkpoint(save_dir)
    if initial_epoch > 0:
        print(f"Resuming from epoch {initial_epoch:03d}")
        model = tf.keras.models.load_model(os.path.join(save_dir, f'model_{initial_epoch:03d}.hdf5'), compile=True)

    save_every = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                 verbose=1, save_weights_only=False, save_freq='epoch')
    save_best = ModelCheckpoint(os.path.join(save_dir, 'best_model.hdf5'),
                                verbose=0, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
    csv_logger = CSVLogger(os.path.join(save_dir, 'model_log.csv'), append=True, separator=',')

    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        initial_epoch=initial_epoch,
        callbacks=[save_every, save_best, csv_logger]
    )

    evaluate_and_plot(model)

if __name__ == '__main__':
    start = time.time()
    main()  # fixed default epochs=1000
    sec = time.time() - start
    print(str(datetime.timedelta(seconds=sec)).split(".")[0])

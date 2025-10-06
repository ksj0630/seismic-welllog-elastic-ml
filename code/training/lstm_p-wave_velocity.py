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


# =========================================
# Directory setup
# =========================================
save_dir = os.path.join('../06.model_save_folder/73.code/')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

output_dir = os.path.join("../05.result/73.code/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# =========================================
# Load data
# =========================================
fin1 = open("../04.merge_data/01.X_train_1A_5_11T2_14_15A.txt", "r")
data1 = np.loadtxt(fin1)
tvd1 = data1[:, 0]
initial_Vp1 = data1[:, 1]
seismic1 = data1[:, 3]
xx = np.stack([tvd1, initial_Vp1, seismic1], axis=1)
yy = data1[:, 6]

fin2 = open("../02.X_train/03.4_X_train_index+10.txt", "r")
data2 = np.loadtxt(fin2)
tvd2 = data2[:, 0]
initial_Vp2 = data2[:, 1]
seismic2 = data2[:, 3]
r_train = np.stack([tvd2, initial_Vp2, seismic2], axis=1)
z_train = data2[:, 6]
init = data2[:, 1]

fin3 = open('../01.GBT_whole_data/12.new_process_4_GBT_whole.txt', "r")
data3 = np.loadtxt(fin3)
tvd3 = data3[:, 0]
truee = data3[:, 6]
init3 = data3[:, 1]

x_train, x_test = xx, r_train
y_train, y_test = yy, z_train


# =========================================
# Sequence data preparation
# =========================================
sequence = 20

def make_sequence(x, y):
    X, Y = [], []
    for i in range(0, len(x) - sequence + 1, sequence):
        X.append(x[i: i + sequence])
        Y.append(y[i + 10])
    return np.array(X), np.array(Y)

X_train, Y_train = make_sequence(x_train, y_train)
X_test, Y_test = make_sequence(x_test, y_test)
R_train, Z_train = make_sequence(r_train, z_train)

print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)
print('R_train:', R_train.shape, 'Z_train:', Z_train.shape)


# =========================================
# Model definition
# =========================================
def MultiHeadAttention_LSTM_model():
    input_layer = Input(shape=(sequence, 3))
    attention = MultiHeadAttention(num_heads=2, key_dim=3)(input_layer, input_layer, input_layer)
    x = LSTM(128, activation='leaky_relu', return_sequences=True)(attention)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation='leaky_relu', return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='leaky_relu')(x)
    x = Dense(16, activation='leaky_relu')(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mae', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.summary()
    return model


# =========================================
# Plotting and saving predictions
# =========================================
def show_results(model):
    pred_train = model.predict(X_train)
    pred_test = model.predict(R_train)

    np.savetxt(os.path.join(output_dir, "20.merge_true.txt"), Y_train)
    np.savetxt(os.path.join(output_dir, "21.merge_pred.txt"), pred_train)
    plt.figure(figsize=(20, 2))
    plt.plot(Y_train, label="True", color="black")
    plt.plot(pred_train, label="Pred", color="red")
    plt.legend()
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '73.merge_code.png'))

    np.savetxt(os.path.join(output_dir, "22.F4_true.txt"), Z_train)
    np.savetxt(os.path.join(output_dir, "23.F4_pred.txt"), pred_test)
    np.savetxt(os.path.join(output_dir, "24.F4_init.txt"), init)

    plt.figure(figsize=(20, 2))
    plt.plot(tvd3[10:-10], Z_train[:], label="True", color="black")
    plt.plot(tvd3[10:-10], pred_test[:], label="Pred", color="red")
    plt.plot(tvd3[10:-10], init3[10:-10], label="Initial", linestyle='dashed', color="blue")
    plt.legend()
    plt.xlim([tvd3[10], tvd3[-10]])
    plt.ylim([1.75, 5.75])
    plt.xlabel("Depth (km)")
    plt.ylabel("P-wave velocity (km/s)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '73.code.png'))


# =========================================
# Main
# =========================================
def main(epochs=1000, batch_size=128):
    model = MultiHeadAttention_LSTM_model()

    def find_last_checkpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, 'model*.hdf5'))
        if file_list:
            epochs_exist = [int(re.findall(".*model_(.*).hdf5.*", f)[0]) for f in file_list]
            return max(epochs_exist)
        return 0

    initial_epoch = find_last_checkpoint(save_dir)
    if initial_epoch > 0:
        print(f'Resuming from epoch {initial_epoch}')
        model = load_model(os.path.join(save_dir, f'model_{initial_epoch:03d}.hdf5'), compile=True)

    save_freq = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                verbose=1, save_weights_only=False, save_freq='epoch')
    save_best = ModelCheckpoint(os.path.join(save_dir, 'best_model.hdf5'),
                                monitor='val_loss', save_best_only=True, mode='min', verbose=0)
    csv_logger = CSVLogger(os.path.join(save_dir, 'model_log.csv'), append=True, separator=',')

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              epochs=epochs, batch_size=batch_size, shuffle=True,
              verbose=1, initial_epoch=initial_epoch,
              callbacks=[save_freq, save_best, csv_logger])

    show_results(model)


if __name__ == '__main__':
    start = time.time()
    main()  # fixed default: 1000 epochs
    sec = time.time() - start
    result = datetime.timedelta(seconds=sec)
    print(str(result).split(".")[0])

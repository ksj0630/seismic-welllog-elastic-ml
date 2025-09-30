import os
import glob
import re
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, BatchNormalization, Activation
from keras.layers import Conv1DTranspose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback

fin3 = open('../01.GBT_whole_data/12.new_process_4_GBT_whole.txt',"r")
data3 = np.loadtxt(fin3)
tvd3 = data3[:,0]
truee = data3[:,5]

#################################
# save_model
#################################
save_dir = os.path.join('../06.model_save_folder/114.code/') 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

output_dir = os.path.join("../05.result/114.code/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.hdf5'))  # get name list of all .hdf5 files
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

class DATA_read():
    def __init__(self, data_file, apply_file):
        X = np.load(data_file[0])
        y = np.load(data_file[1])

        nvert = X.shape[1]
        half = int(nvert / 2)

        X_scaled = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        y_half = y[:, half]

        # 학습용, 테스트용 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_half, test_size=0.1, random_state=42, shuffle=True)

        # read apply data
        X2 = np.load(apply_file[0])
        y2 = np.load(apply_file[1])

        X_scaled2 = np.reshape(X2, (X2.shape[0], X2.shape[1], X2.shape[2]))
        y_half2 = y2[:, half]

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.X_apply = X_scaled2
        self.y_apply = y_half2


def CNN1D(n1, n_ch):
    inputs = layers.Input(shape=(n1, n_ch))

    # Encoding Path (Downsampling) 
    x = Conv1D(64, kernel_size=7, padding='same', input_shape=(n1, n_ch))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv1D(64, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    
    x = Conv1D(128, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv1D(128, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    
    x = Conv1D(256, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Conv1D(256, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation='leaky_relu')(x)
    x = Dense(64, activation='leaky_relu')(x)
    x = Dense(32, activation='leaky_relu')(x)
    x = Dense(16, activation='leaky_relu')(x)
    outputs = Dense(1, activation='leaky_relu')(x)  # Linear activation for regression output

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    model.summary()
    return model


class save_last(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params['epochs'] - 1:
            self.model.save(os.path.join(save_dir, 'last_epoch_model.hdf5'))

def main(n_ch=3, epochs=1000, batch_size=128, structure='CNN1D'):

    data_file = ['../17.1DCNN_train_data_for_density/x_train_rhob.npy', '../17.1DCNN_train_data_for_density/y_train_rhob.npy']
    apply_file = ['../17.1DCNN_train_data_for_density/x_test_rhob.npy', '../17.1DCNN_train_data_for_density/y_test_rhob.npy']

    data = DATA_read(data_file=data_file, apply_file=apply_file)

    # CNN 모델 설정
    if structure == 'CNN1D':
        model = CNN1D(data.X_train.shape[1], n_ch)


    # 이전 모델 로드
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > epochs:
        initial_epoch = epochs
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = models.load_model(os.path.join(save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])

    save_freq = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'), 
                                verbose=1, save_weights_only=False, save_freq='epoch')

    save_best = ModelCheckpoint(os.path.join(save_dir, 'best_model.hdf5'), 
                                verbose=0, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)

    csv_logger = CSVLogger(os.path.join(save_dir, 'model_log.csv'), append=True, separator=',')

    # CNN 모델 학습
    model.fit(data.X_train, data.y_train, 
              epochs=epochs, initial_epoch=initial_epoch,
              batch_size=batch_size, validation_split=0.1,
              callbacks=[save_freq, save_best, csv_logger])

    # CNN 모델을 사용해 테스트 데이터에 대한 예측 수행
    y_pred = model.predict(data.X_test)
    y_pred2 = model.predict(data.X_apply)
    print(y_pred.shape, y_pred2.shape)

    # 저장 경로가 없으면 생성
    output_dir = "../05.result/114.code/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 예측 결과 저장
    np.savetxt(os.path.join(output_dir, "20.merge_true.txt"), data.y_test)
    np.savetxt(os.path.join(output_dir, "21.merge_pred.txt"), y_pred)
    plt.figure(figsize=(20, 2))
    plt.plot(data.y_test, label='True', linewidth=1.0, linestyle='solid', color="black")
    plt.plot(y_pred, label='Pred', linewidth=1.0, linestyle='solid', color="red")
    plt.legend()
    # plt.xlim(2.556, 3.051)
    # plt.xlabel("Depth (km)")
    # plt.ylabel("P-wave velocity (km/s)")
    plt.grid(linestyle=':')
    # plt.show()
    plt.savefig(os.path.join(output_dir, '114.merge_pred.png'))

    np.savetxt(os.path.join(output_dir, "22.F4_true.txt"), data.y_apply)
    np.savetxt(os.path.join(output_dir, "23.F4_pred.txt"), y_pred2)
    plt.figure(figsize=(20, 2))
    plt.plot(tvd3[20:-20], data.y_apply, label='True', linewidth=1.0, linestyle='solid', color="black")
    plt.plot(tvd3[20:-20], y_pred2, label='Pred', linewidth=1.0, linestyle='solid', color="red")
    plt.legend()
    plt.xlim(tvd3[20], tvd3[-20])
    plt.ylim([2.05, 2.85])
    plt.xlabel("Depth (km)")
    plt.ylabel("Density (g/cm3)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '114.pred.png'))
    # plt.show()
    
    # 예측값 저장
    half = int(data.X_apply.shape[1] / 2)
    depth = data.X_apply[:, half, 0]
    pred_dtc = np.zeros((data.y_apply.shape[0], 3))
    print(pred_dtc.shape, data.y_apply.shape)
    pred_dtc[:, 0] = depth[:]
    pred_dtc[:, 1] = data.y_apply[:]
    pred_dtc[:, 2] = y_pred2[:,0]
    np.savetxt(os.path.join(output_dir, "114.pred_dtc.txt"), pred_dtc, delimiter=' ')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='1D CNN for Elastic parameter estimation')
    parser.add_argument('-input_channels', type=int, default=3, help='input channels (default: 3)')
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs (default: 300)')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size (default: 64)')
    parser.add_argument('-model', type=str, default='CNN1D', help='model name to save (default: CNN1D)')
    parser.add_argument('-structure', type=str, default='CNN1D', help='model structure')

    args = parser.parse_args()

    print("args:", args)
    
    start = time.time()
    main(args.input_channels, args.epochs, args.batch_size, args.structure)
    sec = time.time()-start
    result = datetime.timedelta(seconds=sec)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(result_list[0])

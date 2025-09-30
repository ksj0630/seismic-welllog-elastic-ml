import os
import glob
import re
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import *
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from keras.layers import LSTM, BatchNormalization
from tensorflow.keras.layers import MultiHeadAttention, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import Model, load_model, save_model
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


#################################
# save_model
#################################
save_dir = os.path.join('../06.model_save_folder/38.code/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

output_dir = os.path.join("../05.result/38.code/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


###########################
# Load_Data
###########################
fin1 = open("../36.LSTM_GP_merge_data_for_DTS/01.X_train_1A_11T2_14.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
xx = data1[:,:3]
yy = data1[:,3]
tvd1 = data1[:,0]


fin2 = open("../35.LSTM_GP_X_train_for_DTS/02.4_X_train_index+10.txt","r")
data2 = np.loadtxt(fin2)
datalen2 = len(data2)
r_train = data2[:,:3]
z_train = data2[:,3]


fin3 = open('../34.LSTM_GP_train_data_for_DTS/02.4_for_dts.txt',"r")
data3 = np.loadtxt(fin3)
tvd3 = data3[:,0]
truee = data3[:,3]

x_train, x_test = xx, r_train
y_train, y_test = yy, z_train

###########################
# Data_read
###########################
# LSTM train dataset
sequence = 20
X_train, Y_train = [], []
print(len(x_train))
for index in range(0, len(x_train) - sequence+1, sequence):
    X_train.append((x_train[index: index + sequence]))
    Y_train.append((y_train[index+(sequence//2)]))
    print(index, index+sequence, index+(sequence//2))
X_train, Y_train = np.array(X_train), np.array(Y_train)
print(X_train.shape, Y_train.shape)

X_test, Y_test = [], []
for index in range(0, len(x_test) - sequence+1, sequence):
    X_test.append((x_test[index: index + sequence]))
    Y_test.append((y_test[index+(sequence//2)])) 
    print(index, index+sequence, index+(sequence//2))
X_test, Y_test = np.array(X_test), np.array(Y_test)

## Retype and Reshape
X_train = X_train.reshape(X_train.shape[0], sequence, -1)
X_test = X_test.reshape(X_test.shape[0], sequence, -1)
print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)


# LSTM apply dataset
sequence = 20
R_train, Z_train = [], []
for index in range(0, len(r_train) - sequence+1, sequence):
    R_train.append((r_train[index: index + sequence]))
    Z_train.append((z_train[index+(sequence//2)]))
    print(index, index+sequence, index+(sequence//2))
R_train, Z_train = np.array(R_train), np.array(Z_train)

## Retype and Reshape
R_train = R_train.reshape(R_train.shape[0], sequence, -1)
print('R_train:', R_train.shape, 'Z_train:', Z_train.shape)


###########################
# Learning
###########################
def MultiHeadAttention_LSTM_model(sequence):
    input_layer = Input(shape=(sequence, 3))
    attention = MultiHeadAttention(num_heads=2, key_dim=3)(input_layer, input_layer, input_layer)
    x = LSTM(128, activation='leaky_relu', return_sequences=True)(attention)
    x = layers.Dropout(0.2)(x)
    x = LSTM(64, activation='leaky_relu', return_sequences=False)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(32)(x)
    x = layers.Dense(16)(x)
    output_layer = Dense(1)(x)

    # 모델 생성
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    model.summary()

    return model

print(MultiHeadAttention_LSTM_model)


#################################
# Load_the_last_model
#################################
def main(epochs=1000, batch_size=128, gpiter=20):
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    model = MultiHeadAttention_LSTM_model(sequence)


    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir,'model*.hdf5'))  # get name list of all .hdf5 files
        #file_list = os.listdir(save_dir)
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*model_(.*).hdf5.*",file_)
                #print(result[0])
                epochs_exist.append(int(result[0]))
            initial_epoch=max(epochs_exist)
        else:
            initial_epoch = 0
        return initial_epoch


    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5' %initial_epoch), compile=True)

    # use call back functions
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    save_freq = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'), 
                                verbose=1, save_weights_only=False, save_freq='epoch')

    save_best = ModelCheckpoint(os.path.join(save_dir, 'best_model.hdf5'), 
                                verbose=0, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)

    csv_logger = CSVLogger(os.path.join(save_dir, 'model_log.csv'), append=True, separator=',')

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=True,
                                verbose=1,
                                validation_split=0.2,
                                initial_epoch=initial_epoch,
                                callbacks=[save_freq, save_best, csv_logger])

    # 중간 특징 벡터를 추출하기 위해 마지막 Dense 레이어 이전까지 모델을 정의
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # LSTM 모델로 특징 벡터 추출
    features_train = feature_extractor.predict(X_train)
    features_test = feature_extractor.predict(R_train)

    # Gaussian Process 모델 저장 경로
    gpfile = os.path.join(save_dir, 'gaussian_process_model.pkl')

    # Gaussian Process 모델 불러오기 또는 학습
    if initial_epoch == epochs and os.path.isfile(gpfile):
        gp = joblib.load(gpfile)
    else:
        # Gaussian Process 모델링
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gpiter, alpha=2e-1)
        gp.fit(features_train, Y_train)

    # Gaussian Process 모델 저장
    joblib.dump(gp, gpfile)
    
    # GP로 예측 및 불확실성 추정
    y_pred, y_std = gp.predict(features_train, return_std=True)
    y_pred2, y_std2 = gp.predict(features_test, return_std=True)


    # 예측 결과 및 시각화(train data)
    np.savetxt(os.path.join(output_dir, "20.merge_true.txt"), Y_train)
    np.savetxt(os.path.join(output_dir, "21.merge_pred.txt"), y_pred)
    np.savetxt(os.path.join(output_dir, "211.merge_standard.txt"), y_std)

    plt.figure(figsize=(20, 3))
    plt.plot(Y_train, label='True', linewidth=1.0, linestyle='solid', color="black")
    plt.plot(y_pred, label='Pred', linewidth=1.0, linestyle='solid', color="red")
    plt.fill_between(range(len(y_pred)), y_pred - 2*y_std, y_pred + 2*y_std, color='gray', alpha=0.2, label='Uncertainty')
    plt.legend()
    # plt.xlim(2.556, 3.051)
    plt.xlabel("Depth (km)")
    plt.ylabel("S-wave velocity (km/s)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '38.merge_pred.png'))
    # plt.show()


    # 예측값 저장
    half = int(R_train.shape[1] / 2)
    depth = R_train[:, half, 0]
    pred_dtc = np.zeros((Z_train.shape[0], 3))
    print(pred_dtc.shape, Z_train.shape)
    pred_dtc[:, 0] = depth[:]
    pred_dtc[:, 1] = Z_train[:]
    pred_dtc[:, 2] = y_pred2[:]
    np.savetxt(os.path.join(output_dir, "38.pred_dtc.txt"), pred_dtc, delimiter=' ')


    # 예측 결과 및 시각화(test data)
    np.savetxt(os.path.join(output_dir, "22.F4_true.txt"), Z_train)
    np.savetxt(os.path.join(output_dir, "23.F4_pred.txt"), y_pred2)
    np.savetxt(os.path.join(output_dir, "233.F4_standard.txt"), y_std2)
    plt.figure(figsize=(20, 2))
    plt.plot(depth[0:-8], Z_train[0:-8], label='True', linewidth=1.0, linestyle='solid', color="black")
    plt.plot(depth[0:-8], y_pred2[0:-8], label='Pred', linewidth=1.0, linestyle='solid', color="red")
    plt.fill_between(depth[:], y_pred2 - 2*y_std2, y_pred2 + 2*y_std2, color='gray', alpha=0.2, label='Uncertainty')
    plt.legend()
    plt.xlim(depth[0], depth[-8])
    plt.ylim([1.2, 2.9])
    plt.xlabel("Depth (km)")
    plt.ylabel("S-wave velocity (km/s)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '38.pred.png'))
    # plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    sec = time.time()-start
    result = datetime.timedelta(seconds=sec)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print("총 실행 시간:", result_list[0])

    





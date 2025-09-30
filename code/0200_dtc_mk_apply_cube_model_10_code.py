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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, MultiHeadAttention, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


#################################
# save_model
#################################
save_dir = os.path.join('../06.model_save_folder/200.code/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

output_dir = os.path.join("../05.result/200.code/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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

    model = MultiHeadAttention_LSTM_model(sequence=20)

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

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])
    # use call back functions
    save_freq = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'), 
                                verbose=1, save_weights_only=False, save_freq='epoch')

    save_best = ModelCheckpoint(os.path.join(save_dir, 'best_model.hdf5'), 
                                verbose=0, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)

    csv_logger = CSVLogger(os.path.join(save_dir, 'model_log.csv'), append=True, separator=',')


    # 중간 특징 벡터를 추출하기 위해 마지막 Dense 레이어 이전까지 모델을 정의
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)


    ###########################
    # Load_Data
    ###########################
    n1 = 904
    n2 = 213
    n3 = 152
    sequence = 20

    fin2 = open("../38.for_mk_3d_cube_data/un2_cut_seis_904.bin","rb")
    seis = np.fromfile(fin2, dtype='float32')
    seis = seis.reshape(-1, n1)

    fin3 = open("../38.for_mk_3d_cube_data/un2_cut_velp_904.bin","rb")
    velp = np.fromfile(fin3, dtype='float32')
    velp = velp.reshape(-1, n1)

    cube = np.zeros((n3*n2, n1, 3))
    pred_cube = np.zeros((n3*n2, n1-sequence, 1))
    std_cube = np.zeros((n3*n2, n1-sequence, 1))
    print(seis.shape, velp.shape)

    for iz in range(n1):
        cube[:, iz, 0] = 2.3 + iz*0.001


    ##################################
    # Gaussian Process 모델 저장 경로
    ##################################
    gpfile = os.path.join(save_dir, 'gaussian_process_model.pkl')


    #################################
    # Data 처리
    #################################
    if os.path.isfile(gpfile):
        print("Gaussian Process 모델을 불러옵니다.")
        gp = joblib.load(gpfile)  # 이미 저장된 GP 모델 로드
    else:
        raise FileNotFoundError(f"Gaussian Process 모델 파일을 찾을 수 없습니다: {gpfile}")

    for ii in tqdm(range(n3 * n2)):   
        cube[ii, :, 1] = velp[ii, :] * 0.001
        cube[ii, :, 2] = seis[ii, :]
        r_train = cube[ii, :]
        
        # LSTM apply dataset
        R_train  = []
        for index in range(len(r_train) - sequence):
            R_train.append((r_train[index: index + sequence]))
        R_train = np.array(R_train)

        # Retype and Reshape
        R_train = R_train.reshape(R_train.shape[0], sequence, -1)

        # LSTM 모델로 특징 벡터 추출
        features_test = feature_extractor.predict(R_train)

        # Gaussian Process 모델을 통한 예측 및 불확실성 추정
        y_pred2, y_std2 = gp.predict(features_test, return_std=True)

        # 예측값과 표준편차를 별도의 큐브에 저장
        pred_cube[ii, :, 0] = y_pred2  # 예측 값 큐브
        std_cube[ii, :, 0] = y_std2    # 표준 편차 큐브

    fout1 = open(os.path.join(output_dir, 'pred_cube.bin'), 'wb')
    pred_cube = np.float32(pred_cube)
    pred_cube.tofile(fout1)
    fout1.close()

    fout2 = open(os.path.join(output_dir, 'std_cube.bin'), 'wb')
    std_cube = np.float32(std_cube)
    std_cube.tofile(fout2)
    fout2.close()

if __name__ == '__main__':
    start = time.time()
    main()
    sec = time.time()-start
    result = datetime.timedelta(seconds=sec)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print("총 실행 시간:", result_list[0])

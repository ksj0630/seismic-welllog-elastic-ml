import os
import glob
import re
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import *
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MultiHeadAttention, LSTM, Dense, Input
from keras.models import Model, load_model, save_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


#################################
# save_model
#################################
save_dir = os.path.join('../06.model_save_folder/154.code/') 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

output_dir = os.path.join("../05.result/154.code/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

###########################
# Load_Data
###########################
fin1 = open("../26.LSTM_merge_data_from_25.folder/00.merge_data_1A_11T2_14.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
xx = data1[:,:3]
yy = data1[:,3]
tvd1 = data1[:,0]

fin2 = open('../25.LSTM_X_train_for_dts/02.4_X_train_index+10.txt',"r")
data2 = np.loadtxt(fin2)
datalen2 = len(data2)
r_train = data2[:,:3]
z_train = data2[:,3]


fin3 = open('../24.LSTM_train_data_for_dts/02.4_for_dts.txt',"r")
data3 = np.loadtxt(fin3)
tvd3 = data3[:,0]
truee = data3[:,3]


x_train, x_test = xx, r_train
y_train, y_test = yy, z_train


###########################
# Data_read
###########################
sequence = 20
X_train, Y_train = [], []
print(len(x_train))
for index in range(0, len(x_train) - sequence+1, sequence):
    X_train.append((x_train[index: index + sequence]))
    Y_train.append((y_train[index+10]))
    print(index, index+sequence, index+10)
X_train, Y_train = np.array(X_train), np.array(Y_train)
print(X_train.shape, Y_train.shape)


X_test, Y_test = [], []
for index in range(0, len(x_test) - sequence+1, sequence):
    X_test.append((x_test[index: index + sequence]))
    Y_test.append((y_test[index+10])) 
    print(index, index+sequence, index+10)
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
    Z_train.append((z_train[index+10]))
    print(index, index+sequence, index+10)
R_train, Z_train = np.array(R_train), np.array(Z_train)

## Retype and Reshape
R_train = R_train.reshape(R_train.shape[0], sequence, -1)
print('R_train:', R_train.shape, 'Z_train:', Z_train.shape)


###########################
# Learning
###########################
def MultiHeadAttention_LSTM_model():
    input_layer = Input(shape=(sequence, 3))
    attention = MultiHeadAttention(num_heads=2, key_dim=3)(input_layer, input_layer, input_layer)
    x = LSTM(128, activation='leaky_relu', return_sequences=True)(attention)
    x = layers.Dropout(0.2)(x)
    x = LSTM(64, activation='leaky_relu', return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
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
# Predicted_and_save_test_data
#################################
def show_MultiHeadAttention_LSTM_model(SINGLE):
    predict = SINGLE.predict(X_train)
    predict2 = SINGLE.predict(R_train)
    print(predict.shape)

    print('잔차제곱합 : ', np.mean(np.square(Y_train - predict)))
    np.savetxt(os.path.join(output_dir, "20.merge_true.txt"), Y_train)
    np.savetxt(os.path.join(output_dir, "21.merge_pred.txt"), predict)
    plt.figure(figsize=(20, 2))
    plt.plot(Y_train[:], label="True", linewidth=1.0, linestyle='solid', color="black")
    plt.plot(predict[:], label="Pred", linewidth=1.0, linestyle='solid', color="red")
    plt.legend()
    # plt.xlim([tvd1[10], tvd1[-10]]) 
    # plt.ylim([]])
    plt.xlabel("Depth (km)")
    plt.ylabel("S-wave velocity (km/s)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '154.merge_code.png'))
    # plt.show()


    print('잔차제곱합테스트_pred : ', np.mean(np.square(Z_train - predict2)))
    # print('잔차제곱합테스트_init : ', np.mean(np.square(Z_train - init_whole)))
    np.savetxt(os.path.join(output_dir, "22.F4_true.txt"), Z_train, fmt='%.6f')
    np.savetxt(os.path.join(output_dir, "23.F4_pred.txt"), predict2, fmt='%.6f')
    print(tvd3.shape, Z_train.shape, predict2.shape)
    plt.figure(figsize=(20, 2))
    plt.plot(tvd3[10:-18], Z_train[0:-8], label="True", linewidth=1.0, linestyle='solid', color="black")
    plt.plot(tvd3[10:-18], predict2[0:-8], label="Pred", linewidth=1.0, linestyle='solid', color="red")
    plt.legend()
    plt.xlim([tvd3[10], tvd3[-18]]) # 마지막에 결측값 구간있어서 8개 추가로빼버림 
    plt.ylim([1.2, 2.9])
    plt.xlabel("Depth (km)")
    plt.ylabel("S-wave velocity (km/s)")
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(output_dir, '154.code.png'))
    # plt.show()

#################################
# Load_the_last_model
#################################
def main(epochs=432, batch_size=128):
    print(X_train.shape, Y_test.shape)

    model = MultiHeadAttention_LSTM_model()

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
                                initial_epoch=initial_epoch,
                                callbacks=[save_freq, save_best, csv_logger])
 
    
    show_MultiHeadAttention_LSTM_model(model)
    
    
if __name__ == '__main__':
    start = time.time()
    main()
    sec = time.time()-start
    result = datetime.timedelta(seconds=sec)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(result_list[0])
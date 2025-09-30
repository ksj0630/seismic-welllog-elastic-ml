import numpy as np


###########################
# Load_Data
###########################
fin1 = open("../01.well_data/20_1.new_process_4_GBT_whole.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
print(datalen1)

xx = data1[:,:]
yy = data1[:,3]


x_train, y_train = xx, yy


###########################
# Data_read
###########################
# LSTM train dataset
sequence = 10
X_train, Y_train = [], []
for index in range(len(x_train) - sequence):
    X_train.append((x_train[index: index + sequence]))
    Y_train.append((y_train[index+5]))
X_train, Y_train = np.array(X_train), np.array(Y_train)
print(X_train.shape, Y_train.shape)


fout1 = open('../01.well_data/21_1.new_LSTM_4_X_train_index+5.txt', 'w')
for ii in range(X_train.shape[0]):  # X_train의 각 행에 대해 반복
    for jj in range(X_train.shape[1]):  # X_train의 각 열에 대해 반복
        fout1.write(" ".join(map("{:.6f}".format, X_train[ii, jj])) + "\n") # 숫자를 문자열로 변환할 때 소수점 이하까지 출력하도록 포맷 지정
fout1.close()




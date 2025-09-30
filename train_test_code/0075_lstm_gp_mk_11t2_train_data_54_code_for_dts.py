import os
import numpy as np


#################################
# Load_data
#################################
fin1 = open("../../01.train_data_seismic_1m_mvavg_20_add_masking/17.new_15_9_F_11T2/01.well_data/20_5.new_process_11T2_GBT_whole.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
tvd = data1[10:-10,0]
true_dts = data1[10:-10,-1]

fin2 = open("../01.GBT_whole_data/12.new_process_11T2_GBT_whole.txt","r")
# TVD, initial_Vp, initial_Density, Seismic, Horizon_tracking, RHOB, DTC, DTS 순으로 데이터 정렬되있음.
data2 = np.loadtxt(fin2)
datalen2 = len(data2)
initial_vp = data2[10:-10,2]
initial_vs = initial_vp / 1.732
seismic = data2[10:-10,3]


xx = np.zeros((len(tvd),4))
xx[:,0] = tvd
xx[:,1] = initial_vs
xx[:,2] = seismic
xx[:,3] = true_dts
print(tvd.shape, initial_vs.shape, seismic.shape, true_dts.shape)
np.savetxt("../13.LSTM_GP_train_data_for_DTS/04.11T2_for_dts.txt", xx, fmt='%.6f', delimiter='      ')
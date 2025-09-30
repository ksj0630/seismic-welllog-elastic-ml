import os
import numpy as np


#################################
# Load_data
#################################
fin1 = open("../../01.train_data_seismic_1m_mvavg_20_add_masking/19.new_15_9_F_14/01.well_data/20.new_process_14_GBT_whole.txt","r")
# TVD, Pred_DTC, Pred_RHOB, True DTS 순으로 데이터 정렬되있음.
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
tvd = data1[:,0]
true_dts = data1[:,-1]

fin2 = open("../01.GBT_whole_data/12.new_process_14_GBT_whole.txt","r")
# TVD, initial_Vp, initial_Density, Seismic, Horizon_tracking, RHOB, DTC, DTS 순으로 데이터 정렬되있음.
data2 = np.loadtxt(fin2)
datalen2 = len(data2)
seismic = data2[:,3]

fin3 = open("../05.result/123.code/23.F4_pred.txt","r")
data3 = np.loadtxt(fin3)
datalen3 = len(data3)
pred_dtc = data3[:]

xx = np.zeros((datalen1,4))
xx[:,0] = tvd
xx[:,1] = pred_dtc
xx[:,2] = seismic
xx[:,3] = true_dts
print(tvd.shape, pred_dtc.shape, seismic.shape, true_dts.shape)
np.savetxt("../22.GBT_train_data_for_dts/05.14_for_dts.txt", xx, fmt='%.6f', delimiter='      ')
import numpy as np


fin1 = open("../05.result/122.code/23.F4_pred.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
pred_dtc = data1[:]


fin2 = open("../01.GBT_whole_data/12.new_process_11T2_GBT_whole.txt","r")
data2 = np.loadtxt(fin2)
datalen2 = len(data2)

tvd = data2[:,0]
seismic = data2[:,3]
rhob = data2[:,5]
dtc = data2[:,6]
dts = data2[:,7]
print(pred_dtc.shape, tvd.shape)


xx = np.zeros((datalen1,5))
xx[:,0] = tvd
xx[:,1] = pred_dtc
xx[:,2] = seismic
xx[:,3] = rhob
xx[:,4] = dtc
np.savetxt("../20.GBT_train_data_for_density/04.11T2_for_density.txt", xx, fmt='%.6f', delimiter='    ')

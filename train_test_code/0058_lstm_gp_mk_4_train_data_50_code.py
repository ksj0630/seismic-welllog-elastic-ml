import numpy as np
import os


fin1 = open("../05.result/50.code/23.F4_pred.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
pred_Vp = data1[:]


fin2 = open("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt","r")
data2 = np.loadtxt(fin2)
datalen2 = len(data2)

tvd = data2[10:-10,0]
initial_density = data2[10:-10,2]
seismic = data2[10:-10,3]
rhob = data2[10:-10,5]


xx = np.zeros((len(tvd), 4))
xx[:,0] = tvd
xx[:,1] = initial_density
xx[:,2] = seismic
xx[:,3] = rhob
np.savetxt("../10.depth_initial_dens_seismic_RHOB/02.4_for_density.txt", xx, fmt='%.6f', delimiter='    ')

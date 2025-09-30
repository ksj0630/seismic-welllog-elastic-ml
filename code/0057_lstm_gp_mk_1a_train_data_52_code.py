import numpy as np
import os


fin2 = open("../01.GBT_whole_data/12.new_process_1A_GBT_whole.txt","r")
data2 = np.loadtxt(fin2)
datalen2 = len(data2)

tvd = data2[10:-10,0]
initial_density = data2[10:-10,2]
seismic = data2[10:-10,3]
rhob = data2[10:-10,5]
print(tvd.shape)

xx = np.zeros((len(tvd), 4))
xx[:,0] = tvd
xx[:,1] = initial_density
xx[:,2] = seismic
xx[:,3] = rhob
np.savetxt("../10.depth_initial_dens_seismic_RHOB/01.1A_for_density.txt", xx, fmt='%.6f', delimiter='    ')


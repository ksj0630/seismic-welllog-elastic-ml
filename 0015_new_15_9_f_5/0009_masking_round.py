import numpy as np


###########################
# Load_Data
###########################
fin1 = open("../01.well_data/07.extract_mask_5.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
print(datalen1)
xx = data1[:,2]
nz = datalen1
print(xx)


rounded_xx = np.round(xx)
print(rounded_xx)


gbtdata = np.zeros((nz,3))
fout = open("../01.well_data/11.masking_round_5.txt","w")
#................................
kk = 0
for ii in range(nz):
    gbtdata[kk,0] = data1[ii,0]
    gbtdata[kk,1] = data1[ii,1]
    gbtdata[kk,2] = rounded_xx[ii]
    fout.write("%f %f %f \n" %(gbtdata[kk,0], gbtdata[kk,1], gbtdata[kk,2]))

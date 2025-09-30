import numpy as np
from tqdm import tqdm

fin1 = open("../01.well_data/10.new_process_14_MD_VD_intpl.txt","r")
well = np.loadtxt(fin1)
nwell = well.shape[0]

well[:,0] = well[:,0]*0.001

fin2 = open("../01.well_data/05.extract_veltrace_14.txt","r")
vel = np.loadtxt(fin2)
nvel = vel.shape[0]

fin3 = open("../01.well_data/06.extract_dentrace_14.txt","r")
den = np.loadtxt(fin3)
nden = den.shape[0]

fin4 = open("../01.well_data/04.extract_trace_14.txt","r")
seis = np.loadtxt(fin4)
nseis = seis.shape[0]

fin5 = open("../01.well_data/11.masking_round_14.txt","r")
mask = np.loadtxt(fin5)
nmask = mask.shape[0]


nz = nvel
gbtdata = np.zeros((nz,8))
fout = open("../01.well_data/12.new_process_14_GBT_whole.txt","w")
#................................
kk = 0
for ii in range(nz):
    if(vel[ii,1]>=well[0,1]):
        gbtdata[kk,0] = seis[ii,1]*0.001
        gbtdata[kk,1] = vel[ii,2]
        gbtdata[kk,2] = den[ii,2]
        gbtdata[kk,3] = seis[ii,2]
        gbtdata[kk,4] = mask[ii,2]
        gbtdata[kk,5] = well[kk,2]
        gbtdata[kk,6] = well[kk,3]
        gbtdata[kk,7] = well[kk,4]
        fout.write("%f %f %f %f %f %f %f %f \n" %(gbtdata[kk,0], gbtdata[kk,1], gbtdata[kk,2], gbtdata[kk,3], gbtdata[kk,4], gbtdata[kk,5], gbtdata[kk,6], gbtdata[kk,7]))

        kk = kk+1
        if(kk==nwell):
            break

import numpy as np
from tqdm import tqdm

fin1 = open("../01.well_data/19_5.new_process_11A_MD_VD_intpl.txt","r")
well = np.loadtxt(fin1)
nwell = well.shape[0]

well[:,0] = well[:,0]*0.001

fin2 = open("../01.well_data/22.extract_veltrace_11A.txt","r")
vel = np.loadtxt(fin2)
nvel = vel.shape[0]

fin3 = open("../01.well_data/23.extract_dentrace_11A.txt","r")
den = np.loadtxt(fin3)
nden = den.shape[0]

fin4 = open("../01.well_data/04.extract_trace_11A.txt","r")
seis = np.loadtxt(fin4)
nseis = seis.shape[0]

# fin5 = open("../01.well_data/11.masking_round_11A.txt","r")
# mask = np.loadtxt(fin5)
# nmask = mask.shape[0]


nz = nvel
gbtdata = np.zeros((nz,4))
fout = open("../01.well_data/20_5.new_process_11A_GBT_whole.txt","w")
#................................
kk = 0
for ii in range(nz):
    if(vel[ii,1]>=well[0,1]):
        gbtdata[kk,0] = seis[ii,1]*0.001
        gbtdata[kk,1] = vel[ii,2]
        gbtdata[kk,2] = den[ii,2]
        gbtdata[kk,3] = well[kk,4]
        fout.write("%f %f %f %f \n" %(gbtdata[kk,0], gbtdata[kk,1], gbtdata[kk,2], gbtdata[kk,3]))

        kk = kk+1
        if(kk==nwell):
            break

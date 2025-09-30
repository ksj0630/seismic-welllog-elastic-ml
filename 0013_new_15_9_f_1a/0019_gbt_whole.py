import numpy as np
from tqdm import tqdm

fin1 = open("../01.well_data/19.new_process_1A_MD_VD_intpl.txt","r")
well = np.loadtxt(fin1)
nwell = well.shape[0]

well[:,0] = well[:,0]*0.001

fin2 = open("../01.well_data/24.extract_low_dtstrace_1A.txt","r")
lowdts = np.loadtxt(fin2)
ndts = lowdts.shape[0]

fin4 = open("../01.well_data/04.extract_trace_1A.txt","r")
seis = np.loadtxt(fin4)
nseis = seis.shape[0]

fin5 = open("../01.well_data/11.masking_round_1A.txt","r")
mask = np.loadtxt(fin5)
nmask = mask.shape[0]


nz = ndts
gbtdata = np.zeros((nz,5))
fout = open("../01.well_data/25.lowdts_new_process_1A_GBT_whole.txt","w")
#................................
kk = 0
for ii in range(nz):
    if(lowdts[ii,1]>=well[0,1]):
        gbtdata[kk,0] = seis[ii,1]*0.001
        gbtdata[kk,1] = lowdts[ii,2]
        gbtdata[kk,2] = seis[ii,2]
        gbtdata[kk,3] = mask[ii,2]
        gbtdata[kk,4] = well[kk,4]
        fout.write("%f %f %f %f %f \n" %(gbtdata[kk,0], gbtdata[kk,1], gbtdata[kk,2], gbtdata[kk,3], gbtdata[kk,4]))

        kk = kk+1
        if(kk==nwell):
            break

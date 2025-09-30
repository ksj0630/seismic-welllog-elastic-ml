import numpy as np
from tqdm import tqdm

#fin1 = open("process_15_4_m.txt","r")
fin1 = open("../01.well_data/09.new_process_F11A_mvavg.txt","r")
well = np.loadtxt(fin1)
nwell = well.shape[0]

well[:,0] = well[:,0]*1000

fin2 = open("../01.well_data/05.extract_veltrace_11A.txt","r")
seis = np.loadtxt(fin2)
nseis = seis.shape[0]

nz = 2980 #변경해줘야함

intpl = np.zeros((nz,5))

#................................

for ii in tqdm(range(nz)):
    md = seis[ii,0]
    for jj in range(nwell-1):
        wmd1 = well[jj,0]-55
        wmd2 = well[jj+1,0]-55
        if(md >= wmd1 and md < wmd2):
            dist1 = md-wmd1
            dist2 = wmd2 - wmd1

            w2 = dist1 / dist2
            w1 = 1-w2

            intpl[ii,0] = w1*wmd1 + w2*wmd2
            intpl[ii,1] = seis[ii,1]
            intpl[ii,2] = w1*well[jj,1] + w2*well[jj+1,1]
            intpl[ii,3] = w1*well[jj,2] + w2*well[jj+1,2]
            intpl[ii,4] = w1*well[jj,3] + w2*well[jj+1,3]


np.savetxt("../01.well_data/10.new_process_11A_MD_VD_intpl.txt", intpl, fmt="%f %f %f %f %f")


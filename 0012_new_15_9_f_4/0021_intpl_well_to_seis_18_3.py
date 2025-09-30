import numpy as np
from tqdm import tqdm

#fin1 = open("process_15_4_m.txt","r")
fin1 = open("../01.well_data/18_3.new_process_F4_mvavg.txt","r")
well = np.loadtxt(fin1)
nwell = well.shape[0]
print("Well data shape:", well.shape)  # Well 데이터의 크기 출력


# km >> m 단위변경 위해 곱해줌
well[:,0] = well[:,0]*1000

fin2 = open("../01.well_data/22.extract_veltrace_4.txt","r")
seis = np.loadtxt(fin2)
nseis = seis.shape[0]

# 배열 크기 입력
nz = nseis
print(nz)

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


np.savetxt("../01.well_data/19_3.new_process_4_MD_VD_intpl.txt", intpl, fmt="%f %f %f %f %f")


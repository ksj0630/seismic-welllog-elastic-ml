import numpy as np
from tqdm import tqdm

ref = 55

z1 = 146 - ref
z2 = 3260 - ref
nz = z2-z1+1

fin1 = open("../01.well_data/01.15_1B_drilling.txt","r")
well = np.loadtxt(fin1)

well[:,0] = well[:,0]-ref
well[:,1] = well[:,1]-ref

nwell = well.shape[0]

intpl = np.zeros((nz,4))

for ii in tqdm(range(nz)):
    zz = z1+ii
    for jj in range(nwell-1):
        if(zz >= well[jj,1] and zz < well[jj+1,1]):
            dist1 = zz-well[jj,1]
            dist2 = well[jj+1,1]-well[jj,1]

            w2 = dist1 / dist2
            w1 = 1-w2

            intpl[ii,0] = w1*well[jj,0] + w2*well[jj+1,0]
            intpl[ii,1] = zz
            intpl[ii,2] = w1*well[jj,2] + w2*well[jj+1,2]
            intpl[ii,3] = w1*well[jj,3] + w2*well[jj+1,3]


intpl[nz-1,:] = well[nwell-1,:]

np.savetxt("../01.well_data/02.15_1B_intpl.txt", intpl, fmt="%f %f %f %f")


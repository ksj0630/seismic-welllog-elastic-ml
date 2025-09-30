import numpy as np


fin1 = open('../../25.padding_estimation_3d_cube/DTS_low_sm5_4500_cube_pad.bin',"rb")


seis = np.fromfile(fin1,dtype='float32')
n1 = 4500
n2 = 597
n3 = 401

seis = seis.reshape(-1,n1)

nwind = 5
nrange = nwind*2+1
extrs = np.zeros((nrange,n1))
extrt = np.zeros((n1,3))



fin2 = open("../../01.seismic_velocity_equal_size/seis_coord.txt","r")
scoor = np.loadtxt(fin2)
ntrace = scoor.shape[0]

fin3 = open("../01.well_data/03.15_4_match.txt","r")
match = np.loadtxt(fin3)
nwell = match.shape[0]

swind = scoor[:,3:]
print(swind[0])

print(swind.shape)


#nwell=100
#from 148m, skip:4m
kk = 0
for ii in range(1,nwell):
    iz = int(match[ii,1])
    wnco = int(match[ii,4])
    weco = int(match[ii,5])
    loc = np.where(swind==[weco,wnco])[0][0]

    loc1 = loc-nwind
    loc2 = loc+nwind+1

    print(ii,iz,wnco,weco)
    extrs[:,iz] = seis[loc1:loc2,iz]

    extrt[kk,0] = match[ii,0]
    extrt[kk,1] = match[ii,1]
    extrt[kk,2] = seis[loc,iz]
    kk = kk+1


extrs = np.float32(extrs)
fout = open("../01.well_data/24.extract_low_dts_4.bin","wb")
extrs.tofile(fout)

# extrt[:,2] = extrt[:,2]*0.001
np.savetxt("../01.well_data/24.extract_low_dtstrace_4.txt",extrt,fmt='%f')


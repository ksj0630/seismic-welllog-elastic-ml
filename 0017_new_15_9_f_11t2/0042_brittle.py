import numpy as np

###########################
# Load_Data
###########################
fin1 = open("../01.well_data/20_5.new_process_11T2_GBT_whole.txt","r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)
print(datalen1)
tvd = data1[:,0]
velp = data1[:,1]
dens = data1[:,2]
dts = data1[:,3]
tvdlen = len(tvd)
brittle = np.zeros((tvdlen, 4))


# Compute
lamb = dens*(velp**2)-2*((dts**2)*dens)
mmu = (dts**2) * dens
youngs = (mmu*(3*lamb + 2*mmu) / (lamb + mmu))
poisson = (lamb / (2*(lamb + mmu)))

# Compute λρ and μρ
ladens = lamb*dens
mudens = mmu*dens

youngs_brit = (youngs-np.min(youngs)) / (np.max(youngs)-np.min(youngs))
poisson_brit = (poisson - np.max(poisson)) / (np.min(poisson)-np.max(poisson))
brittle_avg = (youngs_brit + poisson_brit)/2
brittle[:,0] = tvd
brittle[:,1] = youngs
brittle[:,2] = poisson
brittle[:,3] = brittle_avg
np.savetxt("../01.well_data/30.brittle_with_TVD_11T2.txt", brittle, fmt='%.6f', delimiter='    ')
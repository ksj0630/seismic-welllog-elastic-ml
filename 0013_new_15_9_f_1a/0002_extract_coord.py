import numpy as np
from tqdm import tqdm

fin1 = open("../01.well_data/02.15_1A_intpl.txt","r")
well = np.loadtxt(fin1)

fin2 = open("../../00.seismic_velocity_equal_size/seis_coord.txt","r")
seis = np.loadtxt(fin2)

print(well.shape)
print(seis.shape)

nwell = well.shape[0]
nseis = seis.shape[0]

well_seis_match = np.zeros((nwell,7))


#nwell = 5
for ii in tqdm(range(nwell)):
    wnco = well[ii,2]*100
    weco = well[ii,3]*100  #well easting coordinate
    distb = 9999999999.

    for jj in range(nseis):
        snco = seis[jj,4]
        seco = seis[jj,3]

        dist = np.sqrt((wnco-snco)*(wnco-snco)+(weco-seco)*(weco-seco))*0.01
        if(dist < distb):
            well_seis_match[ii,0] = well[ii,0]
            well_seis_match[ii,1] = well[ii,1]
            well_seis_match[ii,2] = well[ii,2]
            well_seis_match[ii,3] = well[ii,3]
            well_seis_match[ii,4] = snco
            well_seis_match[ii,5] = seco
            well_seis_match[ii,6] = dist
            distb = dist


np.savetxt("../01.well_data/03.15_1A_match.txt", well_seis_match, fmt="%f %f %f %f %d %d %f")
        


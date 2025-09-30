import numpy as np
from tqdm import tqdm

fin1 = open("../01.well_data/02.15_4_intpl.txt","r") # 1M단위로 Interpolation한 well data(MD, TVD, X, Y좌표)
well = np.loadtxt(fin1)

fin2 = open("../../01.seismic_velocity_equal_size/seis_coord.txt","r") # 탄성파 자료 좌표 데이터
seis = np.loadtxt(fin2)

print(well.shape)
print(seis.shape)

nwell = well.shape[0]
nseis = seis.shape[0]

well_seis_match = np.zeros((nwell,7))

# Easting: The distance measured east from a reference point, or the x-coordinate
# Northing: The distance measured north from a reference point, or the y-coordinate

# nwell = 5
for ii in tqdm(range(nwell)):
    wnco = well[ii,2]*100  # well northing coordinate (m -> cm 단위로 변환) 
    weco = well[ii,3]*100  # well easting coordinate (m -> cm 단위로 변환)
    distb = 9999999999. # 현재 가장 가까운 거리 초기값을 아주 큰 값(9999999999.)으로 설정.
                        # 왜? (1) 첫 번째 거리 계산에서 무조건 업데이트 되도록 하기 위해서!
                        # (2) 가장 가까운 거리(minimum distance)를 찾기 위한 비교를 하기 위해서!
                        # (3) 모든 데이터 중에서 최소 거리를 선택하도록 유도하기 위해서!
                        # 즉, distb는 '임시로 설정한 초깃값'일 뿐, 최소 거리 찾는 과정에서 계속 작아짐.

    for jj in range(nseis):
        snco = seis[jj,4] # seismic data 5번째 열(좌표, Northing)
        seco = seis[jj,3] # seismic data 4번째 열(좌표, Easting)

        # 유클리드 거리 계산
        # 물리검층 좌표와 탄성파자료 좌표 사이의 유클리드 거리 계산
        # 0.01을 곱해줌으로써 다시 미터 단위로 변환
        dist = np.sqrt((wnco-snco)*(wnco-snco)+(weco-seco)*(weco-seco))*0.01
        # 가장 가까운 탄성파 자료 찾기
        if(dist < distb):
            # 현재 계산된 dist 값이 distb(현재까지의 최소 거리)보다 작으면 업데이트
            well_seis_match[ii,0] = well[ii,0]  # 원래 물리검층 자료
            well_seis_match[ii,1] = well[ii,1]
            well_seis_match[ii,2] = well[ii,2]
            well_seis_match[ii,3] = well[ii,3]
            well_seis_match[ii,4] = snco # 가장 가까운 탄성파 자료 좌표 (Northing)
            well_seis_match[ii,5] = seco # 가장 가까운 탄성파 자료 좌표 (Easting)
            well_seis_match[ii,6] = dist # 최소 거리
            distb = dist # 최소 거리 업데이트
                         # 현재 물리검층 자료의 좌표와 가장 가까운 탄성파 자료의 좌표 ㅓ장
                         # distb를 업데이트하여 이후 더 가까운 점을 찾을 때 비교.


np.savetxt("../01.well_data/03.15_4_match.txt", well_seis_match, fmt="%f %f %f %f %d %d %f")
        


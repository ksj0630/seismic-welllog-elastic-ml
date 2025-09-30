import numpy as np
from tqdm import tqdm

# fin1 = open("process_15_4_m.txt","r")
fin1 = open("../01.well_data/09.new_process_F4_mvavg.txt","r") # 실제 las파일에서 긁어온 DEPTH, RHOB, DTC, DTS (mvavg 적용함)
well = np.loadtxt(fin1)
nwell = well.shape[0]

well[:,0] = well[:,0]*1000 # MD*1000 == m (단위 보정)
                           # MD는 단위가 km
                           # seismic data 단위 == m

fin2 = open("../01.well_data/05.extract_veltrace_4.txt","r")
seis = np.loadtxt(fin2)
nseis = seis.shape[0]

nz = nseis # 탄성파 자료의 갯수를 기준으로 보간할 데이터 크기 결정
 
intpl = np.zeros((nz,5)) # MD, Seismic Depth, RHOB, DTC, DTS 순으로 저장하기 위해 (nz, 5) 크기의 배열 생성


for ii in tqdm(range(nz)): # 탄성파 자료의 depth 갯수만큼 반복
    md = seis[ii,0] # 현재 탄성파 자료의 MD 값(Measured Depth, MD)
    for jj in range(nwell-1): # 물리검층 자료의 depth 갯수만큼 반복
        wmd1 = well[jj,0]-55 # 현재 물리검층 자료의 MD 값에서 55m(reference depth*1000) 빼줌
        wmd2 = well[jj+1,0]-55 # 다음 물리검층 자료의 MD 값에서 55m(reference depth*1000) 빼줌

        if(md >= wmd1 and md < wmd2): # 현재 탄성파 md값이 wmd1과 wmd2 사이에 있는 경우 보간 수행
            dist1 = md-wmd1 # seismic md와 첫 번째 물리검층 자료 MD 간의 거리
            dist2 = wmd2 - wmd1 # 두 물리검층 자료 간의 거리

            w2 = dist1 / dist2 # 전체 거리(dist2) 중에서 md가 wmd1에서 얼마나 떨어져 있는지를 비율로 표현
                               #즉, w2가 클수록 wmd2에 가까운 값이고, 작을수록 wmd1에 가까운 값
                               # 보간 가중치(weight)
            w1 = 1-w2 # w2의 보완 값 [이렇게 하면 현재 MD에 대해 선형 보간을 수행할 수 있음]
                      # w2가 wmd2 로의 가중치라면, w1은 wmd1의 가중치
                      # 즉, w1과 w2를 조합해서 두 값 사이에서 적절한 중간값을 찾음
                      # wmd1에 가까운 정도
                      # 보간 가중치를 적용하는 이유
                        # > 탄성파 자료(MD)는 물리검층 자료(wmd1, wmd2)와 정확히 일치하지 않는 경우가 많음
                        # > 따라서, 탄성파 자료의 위치에 맞는 값을 생성하기 위해 보간을 수행함

            intpl[ii,0] = w1*wmd1 + w2*wmd2 # 보간된 MD
            intpl[ii,1] = seis[ii,1]
            intpl[ii,2] = w1*well[jj,1] + w2*well[jj+1,1] # RHOB 값 보간
            intpl[ii,3] = w1*well[jj,2] + w2*well[jj+1,2] # DTC 값 보간
            intpl[ii,4] = w1*well[jj,3] + w2*well[jj+1,3] # DTS 값 보간


np.savetxt("../01.well_data/10.new_process_4_MD_VD_intpl.txt", intpl, fmt="%f %f %f %f %f")


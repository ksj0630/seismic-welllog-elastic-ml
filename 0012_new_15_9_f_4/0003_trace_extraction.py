import numpy as np


fin1 = open('../../01.seismic_velocity_equal_size/coord_ST0202_PSDM_FULL_cdptwind_4500.bin',"rb")


seis = np.fromfile(fin1,dtype='float32')
n1 = 4500
n2 = 597
n3 = 401

seis = seis.reshape(-1,n1)

# window 크기 및 결과 저장을 위한 배열 생성
nwind = 5  # 윈도우 크기 설정
nrange = nwind*2+1 # 중심값을 포함한 11개(5*2+1)의 샘플을 추출하는 윈도우 크기
extrs = np.zeros((nrange,n1)) # 윈도우 내 데이터를 저장할 (11, 4500) 크기의 배열
extrt = np.zeros((n1,3)) # 최종적으로 물리검층 자료와 탄성파 자료를 매칭할 (4500, 3) 크기의 배열


# seismic data 좌표 읽기
fin2 = open("../../01.seismic_velocity_equal_size/seis_coord.txt","r")
scoor = np.loadtxt(fin2)
ntrace = scoor.shape[0]

# 물리검층 자료와 탄성파 자료 매칭 파일 읽기
fin3 = open("../01.well_data/03.15_4_match.txt","r")
match = np.loadtxt(fin3)
nwell = match.shape[0] # 물리검층 자료 z방향으로 갯수.

swind = scoor[:,3:] # seismic 좌표 추출
                    # scoor의 3번 열 이후의 데이터를 swind 배열에 저장
                    # seismic data의 easting, northing 좌표만 추출
# print(swind[0])
# print(swind.shape)


#nwell=100
#from 148m, skip:4m

# 물리검층 자료를 탄성파 자료와 매칭하는 과정
# 즉, 물리검층 자료를 기반으로 가장 가까운 탄성파 자료를 찾아서 연결하는 과정.
# 탄성파 자료와 물리검층 자료는 서로 다른 방식으로 측정되며, 물리검층 자료와 탄성파 자료를 비교/결합하기 위해 해당 과정 수행
# TVD를 사용하는 이유는, 해당 값을 이용하여 탄성파 자료의 동일한 깊이에 있는 값을 가져와야하기 때문임.
kk = 0
for ii in range(1,nwell): # 물리검층 자료 갯수만큼 반복
    iz = int(match[ii,1]) # 물리검층 자료 TVD
    wnco = int(match[ii,4]) # 물리검층과 가장 가까운 탄성파 자료의 Northing 좌표
    weco = int(match[ii,5]) # 물리검층과 가장 가까운 탄성파 자료의 Easting 좌표

    # 이미 계산된 seismic 좌표를 기준으로 loc 찾기
    loc = np.where(swind==[weco,wnco])[0][0] # seismic data에서 위치 찾기

    loc1 = loc-nwind
    loc2 = loc+nwind+1
    print(loc, loc1, loc2)
    print(ii,iz,wnco,weco)
    extrs[:,iz] = seis[loc1:loc2,iz] # seismic data에서 윈도우 영역 추출
                                     # 해당 위치의 iz 깊이에서 윈도우 범위(11개 데이터)를 추출하여 extrs에 저장

    extrt[kk,0] = match[ii,0] # well MD
    extrt[kk,1] = match[ii,1] # well TVD
    extrt[kk,2] = seis[loc,iz] # 해당 seismic data value (seismic amplitude)
                               # loc: well data와 가장 가까운 seismic data의 위치[인덱스]
                               # iz: 물리검층 자료의 TVD
                               # extrt는 TVD와 seismic data를 매칭한 결과를 저장하는 배열
                               # extrt[kk, 2]는 해당 위치의 seismic value를 저장함
                               # 물리검층 자료와 매칭한 탄성파 자료 값을 저장하여 분석에 활용
    kk = kk+1 # 다음 데이터로 이동


extrs = np.float32(extrs)
fout = open("../01.well_data/04.extract_seismic_4.bin","wb")
extrs.tofile(fout)

np.savetxt("../01.well_data/04.extract_trace_4.txt",extrt,fmt='%f')

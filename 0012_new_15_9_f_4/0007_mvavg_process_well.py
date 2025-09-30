import numpy as np

# well data moving average code
# mvavg 적용 목적
# 데이터 노이즈를 줄이고, smoothing된 변화를 분석하기 위해.
# 한 지점에서 갑자기 튀는 이상값(Outlier)들을 제거하는 효과가 있음.
# new_process_txt file은 (MD인가...)TVD, RHOB, DTC, DTS True well data로 excel에서 직접 가공한 자료

fin = open("../01.well_data/08.new_process_F4.txt","r")
data = np.loadtxt(fin)


xx = data[:,1] # True RHOB
yy = data[:,2] # True DTC
zz = data[:,3] # True DTS


lenxx = len(xx)
lenyy = len(yy)
lenzz = len(zz)


nxx = np.zeros((lenxx))
nyy = np.zeros((lenyy))
nzz = np.zeros((lenzz))


window_size = 10


window1 = []
window2 = []
window3 = []


mvavgdata = np.zeros((lenxx, 4))
fout = open("../01.well_data/09.new_process_F4_mvavg.txt","w")
#................................
kk=0 # mvavg data의 인덱스 카운터
for ii in range(0,lenxx):
    if window_size <= 0: # window_size가 0이하면 오류 발생
        raise ValueError("윈도우 크기는 1 이상이어야 합니다.")
    start_idx1 = max(0, ii-window_size) # 현재 인덱스 ii에서 window_size만큼 앞쪽을 포함하는 시작 인덱스.
    start_idx2 = max(0, ii-window_size) # 음수 인덱스를 방지하기 위해 'max(0, ii-window_size)' 사용
    start_idx3 = max(0, ii-window_size)
    print(start_idx1, start_idx2, start_idx3)

    end_idx1 = min(lenxx, ii+window_size+1) # 현재 인덱스 ii에서 window_size만큼 뒤쪽을 포함하는 종료 인덱스.
    end_idx2 = min(lenxx, ii+window_size+1) # 배열 길이를 초과하지 않도록 'min(lenxx, value)' 사용.
    end_idx3 = min(lenxx, ii+window_size+1) # +1을 하는 이유는 Python의 슬라이싱이 마지막 인덱스를 포함하지 않기 때문.
    print(end_idx1, end_idx2, end_idx3)

    window1 = data[start_idx1:end_idx1, 1] # mvavg를 적용할 범위 추출
    window2 = data[start_idx2:end_idx2, 2] 
    window3 = data[start_idx3:end_idx3, 3]
    print(window1, window2, window3)

    nxx[ii] = (sum(window1) / len(window1)) # mvavg 계산
    nyy[ii] = (sum(window2) / len(window2)) # 이렇게 하면 각 데이터 포인트를 중심으로 앞뒤 10개를 포함한 평균이 적용됨.
    nzz[ii] = (sum(window3) / len(window3)) # 데이터의 노이즈를 줄이고 smoothing된 값으로 변환됨
    # print(ma_values)
    mvavgdata[kk,0] = data[ii,0] # 원래 TVD value 유지
    mvavgdata[kk,1] = nxx[ii] # mvavg x 값
    mvavgdata[kk,2] = nyy[ii] # mvavg y 값
    mvavgdata[kk,3] = nzz[ii] # mvavg z 값
    fout.write("%f %f %f %f \n" %(mvavgdata[kk,0], mvavgdata[kk,1], mvavgdata[kk,2], mvavgdata[kk,3]))


import numpy as np


fin = open("../01.well_data/17_1.No_missing_DTS_F11A.txt","r")
data = np.loadtxt(fin)


xx = data[:,1]
yy = data[:,2]
zz = data[:,3]


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
fout = open("../01.well_data/18_1.new_process_F11A_mvavg.txt","w")
#................................
kk=0
for ii in range(0,lenxx):
    if window_size <= 0:
        raise ValueError("윈도우 크기는 1 이상이어야 합니다.")
    start_idx1 = max(0, ii-window_size)
    start_idx2 = max(0, ii-window_size)
    start_idx3 = max(0, ii-window_size)
    print(start_idx1, start_idx2, start_idx3)

    end_idx1 = min(lenxx, ii+window_size+1)
    end_idx2 = min(lenxx, ii+window_size+1)
    end_idx3 = min(lenxx, ii+window_size+1)
    print(end_idx1, end_idx2, end_idx3)

    window1 = data[start_idx1:end_idx1, 1]
    window2 = data[start_idx2:end_idx2, 2]
    window3 = data[start_idx3:end_idx3, 3]
    print(window1, window2, window3)

    nxx[ii] = (sum(window1) / len(window1))
    nyy[ii] = (sum(window2) / len(window2))
    nzz[ii] = (sum(window3) / len(window3))
    # print(ma_values)
    mvavgdata[kk,0] = data[ii,0]
    mvavgdata[kk,1] = nxx[ii]
    mvavgdata[kk,2] = nyy[ii]
    mvavgdata[kk,3] = nzz[ii]
    fout.write("%f %f %f %f \n" %(mvavgdata[kk,0], mvavgdata[kk,1], mvavgdata[kk,2], mvavgdata[kk,3]))


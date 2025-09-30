import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import datetime

#################################
# save_dir
#################################
save_dir = os.path.join('../06.model_save_folder/121.code/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

output_dir = os.path.join("../05.result/121.code/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#################################
# Load_data
#################################
fin1 = open("../04.merge_data/00.GBT_merge_1A_5_11T2_14_15A.txt", "r")
data1 = np.loadtxt(fin1)
datalen1 = len(data1)

tvd1 = data1[:, 0]
initial_Vp1 = data1[:, 1]
seismic1 = data1[:, 3]
xx = np.zeros((datalen1, 3))
xx[:, 0] = tvd1
xx[:, 1] = initial_Vp1
xx[:, 2] = seismic1
yy = data1[:, 6]
print(xx.shape, yy.shape)

fin2 = open("../01.GBT_whole_data/12.new_process_5_GBT_whole.txt", "r")
data2 = np.loadtxt(fin2)
datalen2 = len(data2)
print(datalen2)

tvd2 = data2[:, 0]
initial_Vp2 = data2[:, 1]
seismic2 = data2[:, 3]
r_train = np.zeros((datalen2, 3))
r_train[:, 0] = tvd2
r_train[:, 1] = initial_Vp2
r_train[:, 2] = seismic2
z_train = data2[:, 6]
print(r_train.shape, z_train.shape)

tvd3 = data2[:, 0]
init = data2[:, 1]

x_train, x_test = xx, r_train
y_train, y_test = yy, z_train

##############################################
# GradientBoostingRegressor + Training Time
##############################################
start_time = time.time()  # 학습 시작 시간 기록

sk_reg = GradientBoostingRegressor(
    loss='absolute_error',    # MAE 손실 함수 설정
    n_estimators=25000,        # 트리 개수
    learning_rate=1e-4,        # 학습률
    max_depth=25,              # 트리 깊이
    min_samples_split=5,       # 노드 분할을 위한 최소 샘플 수
    min_samples_leaf=7,        # 리프 노드의 최소 샘플 수
    subsample=0.8,             # 사용할 데이터 비율
    random_state=42            # 재현성을 위한 시드 고정
).fit(x_train, y_train)

# 학습 소요 시간 계산
elapsed_time = time.time() - start_time
training_duration = datetime.timedelta(seconds=elapsed_time)
training_duration_str = str(training_duration).split(".")[0]  # 소수점 제거

#################################
# save_model
#################################
joblib.dump(sk_reg, os.path.join(save_dir, "gbr_model.pkl"))

# Load the saved GradientBoostingRegressor model
model_path = os.path.join(save_dir, 'gbr_model.pkl')
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    gbr_model = joblib.load(model_path)
else:
    print("Model file not found.")

# 모델 로드 소요 시간 계산
model_start_time = time.time()
model_elapsed_time = time.time() - model_start_time
model_load_time = datetime.timedelta(seconds=model_elapsed_time)
model_load_time_str = str(model_load_time).split(".")[0]  # 소수점 제거
print(f"Model loading time: {model_load_time_str}")


# Train 데이터 예측
predict_train = gbr_model.predict(x_train)
predict_test = gbr_model.predict(x_test)

# print("Loaded model predictions on test data:", predict_test)

# 잔차제곱합 계산 및 저장
print('잔차제곱합 (Train): ', np.mean(np.square(y_train - predict_train)))
np.savetxt(os.path.join(output_dir, "20.merge_true.txt"), y_train)
np.savetxt(os.path.join(output_dir, "21.merge_pred.txt"), predict_train)

# Train 데이터 시각화
plt.figure(figsize=(20, 2))
plt.plot(y_train, label="True", linewidth=1.0, linestyle='solid', color="black")
plt.plot(predict_train, label="Pred DTC", linewidth=1.0, linestyle='solid', color="blue")
plt.legend()
plt.grid(linestyle=':')
plt.savefig(os.path.join(output_dir, '121.merge_code.png'))

# Test 데이터 예측
print('잔차제곱합 (Test): ', np.mean(np.square(y_test - predict_test)))
np.savetxt(os.path.join(output_dir, "22.F4_true.txt"), y_test)
np.savetxt(os.path.join(output_dir, "23.F4_pred.txt"), predict_test)
np.savetxt(os.path.join(output_dir, "24.F4_init.txt"), init)

# Test 데이터 시각화
plt.figure(figsize=(20, 2))
plt.plot(tvd3[:], y_test, label="True", linewidth=1.0, linestyle='solid', color="black")
plt.plot(tvd3[:], predict_test, label="Pred DTC", linewidth=1.0, linestyle='solid', color="red")
plt.plot(tvd3[:], init, label="initial", linewidth=1.0, linestyle='dotted', color="blue")
plt.legend()
plt.xlim([tvd3[0], tvd3[-1]])
plt.ylim([1, 5.75])
plt.xlabel("Depth (km)")
plt.ylabel("P-wave velocity (km/s)")
plt.grid(linestyle=':')
plt.savefig(os.path.join(output_dir, '121.code.png'))

print(f"Total training time: {training_duration_str}")

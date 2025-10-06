import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import joblib


# -----------------------------
# Paths
# -----------------------------
SAVE_DIR   = Path("../06.model_save_folder/114.code")
OUTPUT_DIR = Path("../05.result/114.code")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PATH   = Path("../04.merge_data/00.GBT_merge_1A_5_11T2_14_15A.txt")
WHOLE_PATH    = Path("../01.GBT_whole_data/12.new_process_4_GBT_whole.txt")
MODEL_PATH    = SAVE_DIR / "gbr_model.pkl"

# -----------------------------
# Load data
# -----------------------------
data1 = np.loadtxt(MERGED_PATH)   # train (supervised pairs)
data2 = np.loadtxt(WHOLE_PATH)    # apply/eval over full depth

x_train = np.column_stack([data1[:, 0], data1[:, 1], data1[:, 3]])  # TVD, initial Vp, seismic
y_train = data1[:, 6]                                                # target Vp

x_test  = np.column_stack([data2[:, 0], data2[:, 1], data2[:, 3]])  # TVD, initial Vp, seismic
y_test  = data2[:, 6]
tvd3    = data2[:, 0]
init    = data2[:, 1]

print("x_train/y_train:", x_train.shape, y_train.shape)
print("x_test/y_test  :", x_test.shape,  y_test.shape)

# -----------------------------
# Train
# -----------------------------
t0 = time.time()
gbr = GradientBoostingRegressor(
    loss="absolute_error",
    n_estimators=25_000,
    learning_rate=1e-4,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=7,
    subsample=0.8,
    random_state=42,
)
gbr.fit(x_train, y_train)
elapsed = str(datetime.timedelta(seconds=time.time() - t0)).split(".")[0]
print(f"Training time: {elapsed}")

# -----------------------------
# Save & (optionally) reload
# -----------------------------
joblib.dump(gbr, MODEL_PATH)
print(f"Saved model → {MODEL_PATH.resolve()}")
gbr = joblib.load(MODEL_PATH)

# -----------------------------
# Predict
# -----------------------------
pred_train = gbr.predict(x_train)
pred_test  = gbr.predict(x_test)

# -----------------------------
# Metrics (simple)
# -----------------------------
mse_train = np.mean((y_train - pred_train) ** 2)
mse_test  = np.mean((y_test  - pred_test)  ** 2)
print(f"MSE (train): {mse_train:.6f}")
print(f"MSE (test) : {mse_test:.6f}")

# -----------------------------
# Save artifacts
# -----------------------------
np.savetxt(OUTPUT_DIR / "20.merge_true.txt", y_train)
np.savetxt(OUTPUT_DIR / "21.merge_pred.txt", pred_train)

plt.figure(figsize=(20, 2))
plt.plot(y_train, label="True", linewidth=1.0, color="black")
plt.plot(pred_train, label="Pred Vp", linewidth=1.0, color="blue")
plt.legend(); plt.grid(linestyle=":")
plt.savefig(OUTPUT_DIR / "114.merge_code.png"); plt.close()

np.savetxt(OUTPUT_DIR / "22.F4_true.txt", y_test)
np.savetxt(OUTPUT_DIR / "23.F4_pred.txt", pred_test)
np.savetxt(OUTPUT_DIR / "24.F4_init.txt", init)

plt.figure(figsize=(20, 2))
plt.plot(tvd3, y_test,     label="True",    linewidth=1.0, color="black")
plt.plot(tvd3, pred_test,  label="Pred Vp", linewidth=1.0, color="red")
plt.plot(tvd3, init,       label="Initial", linewidth=1.0, linestyle="dashed", color="blue")
plt.legend()
plt.xlim([tvd3[0], tvd3[-1]])
plt.ylim([1.0, 5.75])
plt.xlabel("Depth (km)")
plt.ylabel("P-wave velocity (km/s)")
plt.grid(linestyle=":")
plt.savefig(OUTPUT_DIR / "114.code.png"); plt.close()

print(f"Done. Artifacts in: {OUTPUT_DIR.resolve()}")

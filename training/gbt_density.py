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
SAVE_DIR   = Path("../06.model_save_folder/131.code")
OUTPUT_DIR = Path("../05.result/131.code")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = Path("../21.GBT_merge_data_from_20.folder/00.merge_data_1A_5_11T2_14_15A.txt")
APPLY_PATH = Path("../20.GBT_train_data_for_density/02.4_for_density.txt")
MODEL_PATH = SAVE_DIR / "gbr_model.pkl"

# -----------------------------
# Load data
# -----------------------------
train = np.loadtxt(TRAIN_PATH)
apply = np.loadtxt(APPLY_PATH)

# features: TVD, predicted_DTC, seismic; target: density
x_train = np.column_stack([train[:, 0], train[:, 1], train[:, 2]])
y_train = train[:, 3]

x_test  = np.column_stack([apply[:, 0], apply[:, 1], apply[:, 2]])
y_test  = apply[:, 3]
tvd2    = apply[:, 0]

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
# Save & reload
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
# Metrics
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
plt.plot(pred_train, label="Pred", linewidth=1.0, color="red")
plt.legend(); plt.grid(linestyle=":")
plt.savefig(OUTPUT_DIR / "131.merge_code.png"); plt.close()

np.savetxt(OUTPUT_DIR / "22.F4_true.txt", y_test)
np.savetxt(OUTPUT_DIR / "23.F4_pred.txt", pred_test)

plt.figure(figsize=(20, 2))
mask = slice(20, -20)  # crop edges if needed
plt.plot(tvd2[mask], y_test[mask],     label="True", linewidth=1.0, color="black")
plt.plot(tvd2[mask], pred_test[mask],  label="Pred", linewidth=1.0, color="red")
plt.legend()
plt.xlim([tvd2[20], tvd2[-20]])
plt.ylim([2.05, 2.85])
plt.xlabel("Depth (km)")
plt.ylabel("Density (g/cm³)")
plt.grid(linestyle=":")
plt.savefig(OUTPUT_DIR / "131.code_300.png"); plt.close()

print("Done.")

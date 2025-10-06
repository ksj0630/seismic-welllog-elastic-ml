import numpy as np
from tqdm import tqdm
from pathlib import Path

well_path = Path("../01.well_data/02.15_4_intpl.txt")
seis_path = Path("../../01.seismic_velocity_equal_size/seis_coord.txt")
out_path = Path("../01.well_data/03.15_4_match.txt")

well = np.loadtxt(well_path)
seis = np.loadtxt(seis_path)

print(well.shape)
print(seis.shape)

nwell = well.shape[0]
snco = seis[:, 4]
seco = seis[:, 3]

result = np.zeros((nwell, 7), dtype=float)

for i in tqdm(range(nwell)):
    wnco_cm = well[i, 2] * 100.0
    weco_cm = well[i, 3] * 100.0
    d2 = (snco - wnco_cm) ** 2 + (seco - weco_cm) ** 2
    j_min = np.argmin(d2)
    dist_m = np.sqrt(d2[j_min]) * 0.01
    result[i, 0] = well[i, 0]
    result[i, 1] = well[i, 1]
    result[i, 2] = well[i, 2]
    result[i, 3] = well[i, 3]
    result[i, 4] = snco[j_min]
    result[i, 5] = seco[j_min]
    result[i, 6] = dist_m

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(out_path, result, fmt="%f %f %f %f %d %d %f")
print(f"Saved → {out_path.resolve()}")

import numpy as np
from tqdm import tqdm
from pathlib import Path

well_path = Path("../01.well_data/09.new_process_F4_mvavg.txt")
seis_path = Path("../01.well_data/05.extract_veltrace_4.txt")
out_path = Path("../01.well_data/10.new_process_4_MD_VD_intpl.txt")

well = np.loadtxt(well_path)
seis = np.loadtxt(seis_path)

nwell = well.shape[0]
nseis = seis.shape[0]

well[:, 0] *= 1000  # MD 단위 km → m
nz = nseis

intpl = np.zeros((nz, 5), dtype=float)

for i in tqdm(range(nz)):
    md = seis[i, 0]
    for j in range(nwell - 1):
        wmd1 = well[j, 0] - 55
        wmd2 = well[j + 1, 0] - 55
        if wmd1 <= md < wmd2:
            dist1 = md - wmd1
            dist2 = wmd2 - wmd1
            w2 = dist1 / dist2
            w1 = 1 - w2
            intpl[i, 0] = w1 * wmd1 + w2 * wmd2
            intpl[i, 1] = seis[i, 1]
            intpl[i, 2] = w1 * well[j, 1] + w2 * well[j + 1, 1]
            intpl[i, 3] = w1 * well[j, 2] + w2 * well[j + 1, 2]
            intpl[i, 4] = w1 * well[j, 3] + w2 * well[j + 1, 3]
            break

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(out_path, intpl, fmt="%.6f %.6f %.6f %.6f %.6f")

print(f"Saved → {out_path.resolve()}")

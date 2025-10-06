import numpy as np
from pathlib import Path

bin_path = Path("../../01.seismic_velocity_equal_size/coord_ST0202_PSDM_FULL_cdptwind_4500.bin")
seis_coord_path = Path("../../01.seismic_velocity_equal_size/seis_coord.txt")
match_path = Path("../01.well_data/03.15_4_match.txt")
out_bin = Path("../01.well_data/04.extract_seismic_4.bin")
out_txt = Path("../01.well_data/04.extract_trace_4.txt")

seis = np.fromfile(bin_path, dtype="float32")
n1, n2, n3 = 4500, 597, 401
seis = seis.reshape(-1, n1)

nwind = 5
nrange = nwind * 2 + 1
extrs = np.zeros((nrange, n1), dtype=float)
extrt = np.zeros((n1, 3), dtype=float)

scoor = np.loadtxt(seis_coord_path)
match = np.loadtxt(match_path)

swind = scoor[:, 3:]
nwell = match.shape[0]

kk = 0
for ii in range(1, nwell):
    iz = int(match[ii, 1])
    wnco = int(match[ii, 4])
    weco = int(match[ii, 5])

    loc = np.where((swind[:, 0] == weco) & (swind[:, 1] == wnco))[0][0]
    loc1, loc2 = loc - nwind, loc + nwind + 1

    extrs[:, iz] = seis[loc1:loc2, iz]
    extrt[kk, 0] = match[ii, 0]
    extrt[kk, 1] = match[ii, 1]
    extrt[kk, 2] = seis[loc, iz]
    kk += 1

extrs = np.float32(extrs)
out_bin.parent.mkdir(parents=True, exist_ok=True)
extrs.tofile(out_bin)

np.savetxt(out_txt, extrt, fmt="%f")
print(f"Saved binary → {out_bin.resolve()}")
print(f"Saved text   → {out_txt.resolve()}")

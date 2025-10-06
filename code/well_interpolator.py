import numpy as np
from tqdm import tqdm
from pathlib import Path

# Parameters
ref = 55
z1 = 146 - ref
z2 = 3138 - ref
nz = z2 - z1 + 1

# Load data
input_path = Path("../01.well_data/01.15_4_drilling.txt")
well = np.loadtxt(input_path)

# Adjust reference
well[:, 0] -= ref
well[:, 1] -= ref

# Sort by depth (for safety)
well = well[np.argsort(well[:, 1])]

# Prepare interpolation grid
depth_grid = np.arange(z1, z2 + 1, 1)
intpl = np.zeros((len(depth_grid), 4), dtype=float)
intpl[:, 1] = depth_grid

# Perform linear interpolation
for col in [0, 2, 3]:
    intpl[:, col] = np.interp(depth_grid, well[:, 1], well[:, col])

# Match the last row with the last well sample
intpl[-1, :] = well[-1, :4]

# Generate output filename automatically
output_path = input_path.with_name(input_path.stem + "_intpl.txt")

# Save interpolated result
np.savetxt(output_path, intpl, fmt="%.6f %.6f %.6f %.6f")

print(f"✅ Interpolation complete → {output_path.resolve()}")

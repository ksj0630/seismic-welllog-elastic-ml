# seismic-welllog-elastic-ml

This project studies how to predict subsurface elastic properties—density (ρ), P-wave velocity (Vp), and S-wave velocity (Vs)—by fusing seismic information with well-log measurements. Instead of relying solely on physics-driven inversion, we evaluate data-driven models that learn mappings from seismic/well-log features to elastic properties, and we quantify predictive uncertainty where appropriate.

Why this matters

Accurate elastic property estimation supports reservoir characterization, geohazard assessment, and safe subsurface operations. Well logs provide high-resolution but sparse depth coverage, while seismic offers broad coverage at lower resolution. Combining both with ML can deliver high-fidelity, spatially consistent property estimates at reduced cost.

Data & problem setup

Dataset: Public Volve field (North Sea)–aligned seismic attributes and well logs along the well path.

Targets: ρ, Vp, Vs (continuous regression).

Inputs (examples): depth (TVD), low-resolution/initial velocity cues, and seismic attributes sampled along the well trajectory.

Sampling: Each depth point is a labeled training instance; sequence models additionally use sliding windows to capture vertical context.

Models evaluated

Gradient Boosting Trees (GBT): Treats each depth independently; strong tabular baseline.

LSTM (sequence model): Captures vertical continuity in logs and seismic.

1D-CNN (sequence model): Learns local patterns with convolutional context.

Uncertainty: LSTM+GP and 1D-CNN+GP pair a deterministic backbone with Gaussian Process regression for calibrated predictive uncertainty.

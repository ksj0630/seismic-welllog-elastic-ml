# seismic-welllog-elastic-ml

This repository explores data-driven prediction of subsurface elastic properties—density (ρ), P-wave velocity (Vp), and S-wave velocity (Vs)—by fusing seismic information with well-log measurements. Rather than relying solely on physics-driven inversion, we evaluate machine-learning models that learn mappings from seismic/well-log features to elastic properties, and we quantify predictive uncertainty where appropriate.

**Keywords:** seismic, well logs, elastic properties, Vp, Vs, density, machine learning, uncertainty, Volve

---

## Why this matters
Accurate elastic property estimation supports reservoir characterization, geohazard assessment, and safe subsurface operations. Well logs provide high-resolution but sparse depth coverage, while seismic offers broad coverage at lower resolution. Combining both with ML can deliver high-fidelity, spatially consistent property estimates at reduced cost.

---

## Data & problem setup
- **Dataset:** Public *Volve* field (North Sea) — aligned seismic attributes and well logs along the well path.  
- **Targets:** ρ, Vp, Vs (continuous regression).  
- **Inputs (examples):** depth (TVD), low-resolution/initial velocity cues, and seismic attributes sampled along the well trajectory.  
- **Sampling:** Each depth point is a labeled training instance; sequence models use sliding windows to capture vertical context.

> **Note:** This repo does not redistribute Volve data. Follow the instructions in [`data/README.md`](data/README.md) to obtain and preprocess the dataset.

---

## Models evaluated
- **Gradient Boosting Trees (GBT):** Treats each depth independently; strong tabular baseline.  
- **LSTM (sequence model):** Captures vertical continuity in logs and seismic.  
- **1D-CNN (sequence model):** Learns local patterns with convolutional context.  
- **Uncertainty:** **LSTM+GP** and **1D-CNN+GP** pair a deterministic backbone with Gaussian Process regression for calibrated predictive uncertainty.

---

## Evaluation
We report:
- **RMSE** (accuracy), **MRE** (relative error), **PCC** (Pearson correlation)
- Depth-wise profiles comparing **True vs. Pred vs. Initial/Low-res**
- Ablations on **learning rate** and **tree depth** (for GBT) to assess robustness



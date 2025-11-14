import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Load CSV
# ============================================================
csv_filename = "cvqkd_keyrate+phasemse_sweep_model_mismatch.csv"
df = pd.read_csv(csv_filename)

# Unique measurement noise values
meas_var_list = sorted(df["MeasVar"].unique())

# Models, colors, and column mappings
models = ["KalmanNet", "RNN", "UKF"]
colors = {"KalmanNet": "C0", "RNN": "C1", "UKF": "C2"}
mse_col_map = {"KalmanNet": "MSE_KalmanNet_Last", "RNN": "MSE_RNN_Last", "UKF": "MSE_UKF_Last"}
ci_col_map = {"KalmanNet": "CI_KalmanNet", "RNN": "CI_RNN", "UKF": "CI_UKF"}
key_col_map = {"KalmanNet": "K_KNet", "RNN": "K_RNN", "UKF": "K_UKF"}

# ============================================================
# Figure 1: Phase MSE (Last Symbol)
# ============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
axes1 = axes1.flatten()

for i, meas_val in enumerate(meas_var_list):
    if i >= len(axes1): break
    ax = axes1[i]
    subset = df[df["MeasVar"] == meas_val].copy()

    # Remove the last phase point
    subset = subset.iloc[:]

    for model in models:
        mse_col = mse_col_map[model]
        ci_col = ci_col_map[model]
        # Convert std to variance
        phase_var = subset["PhaseVar"]

        if mse_col in subset.columns:
            ax.errorbar(
                phase_var, subset[mse_col],
                yerr=subset[ci_col] if ci_col in subset.columns else None,
                fmt='o-', capsize=4, color=colors[model], label=model
            )

    ax.set_xlabel("Phase Noise Variance (Q) ")
    ax.set_ylabel("MSE")
    ax.set_title(f"Measurement Noise Variance (σ²) = {meas_val}")
    ax.grid(True)
    ax.legend()

plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.suptitle("Pilot Phase Estimation Last-Symbol MSE vs Phase Noise", fontsize=16, y=1.02)
plt.show()

# ============================================================
# Figure 2: Secret Key Rate vs Phase Noise
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
axes2 = axes2.flatten()

for i, meas_val in enumerate(meas_var_list):
    if i >= len(axes2): break
    ax = axes2[i]
    subset = df[df["MeasVar"] == meas_val].copy()

    # Remove the last phase point
    subset = subset.iloc[:]

    for model in models:
        key_col = key_col_map[model]
        # Convert std to variance
        phase_var = subset["PhaseVar"]

        if key_col in subset.columns:
            ax.plot(
                phase_var, subset[key_col],
                'o-', color=colors[model], label=model
            )

    ax.set_xlabel("Phase Noise Variance (σ²)")
    ax.set_ylabel("Asymptotic Secret Key Rate")
    ax.set_title(f"Measurement Noise Variance = {meas_val}")
    ax.grid(True)
    ax.legend()

plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.suptitle("Secret Key Rate vs Phase Noise", fontsize=16, y=1.02)
plt.show()

# ============================================================
# Figure 3: Excess Noise vs Phase Noise
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 12))
axes3 = axes3.flatten()

excess_col_map = {"KalmanNet": "ExcessNoise_KNet", "RNN": "ExcessNoise_RNN", "UKF": "ExcessNoise_UKF"}

for i, meas_val in enumerate(meas_var_list):
    if i >= len(axes3): break
    ax = axes3[i]
    subset = df[df["MeasVar"] == meas_val].copy()

    phase_var = subset["PhaseVar"]

    for model in models:
        ex_col = excess_col_map[model]
        if ex_col in subset.columns:
            ax.plot(
                phase_var, subset[ex_col],
                'o-', color=colors[model], label=model
            )

    ax.set_xlabel("Phase Noise Variance (σ²)")
    ax.set_ylabel("Excess Noise ε")
    ax.set_title(f"Measurement Noise Variance = {meas_val}")
    ax.grid(True)
    ax.legend()

plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.suptitle("Excess Noise vs Phase Noise", fontsize=16, y=1.02)
plt.show()

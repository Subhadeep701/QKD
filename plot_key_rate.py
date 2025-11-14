import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV file
csv_filename = "cvqkd_keyrate+phasemse_sweep_new.csv"

# Load CSV
df = pd.read_csv(csv_filename)

# Get first 4 unique measurement variances
meas_vars = sorted(df['MeasVar'].unique())[:4]

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Key Rate vs Phase Noise Variance for Different Measurement Variances", fontsize=16)

for ax, meas_var in zip(axes.flatten(), meas_vars):
    df_subset = df[df['MeasVar'] == meas_var]

    # Convert stored std to variance
    phase_std = df_subset['PhaseVar'].values
    phase_var = phase_std ** 2

    K_kn = df_subset['K_KNet'].values
    K_rnn = df_subset['K_RNN'].values
    K_ukf = df_subset['K_UKF'].values

    ax.plot(phase_var, K_kn, marker='o', label='KalmanNet')
    ax.plot(phase_var, K_rnn, marker='s', label='RNN')
    ax.plot(phase_var, K_ukf, marker='^', label='UKF')

    ax.set_xscale('log')  # Log scale for variance
    ax.set_xlabel('Phase Noise Variance')
    ax.set_ylabel('Key Rate (bits/pulse)')
    ax.set_title(f'Measurement Noise Variance = {meas_var}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

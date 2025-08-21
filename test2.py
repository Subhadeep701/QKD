import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from filterpy.kalman import ExtendedKalmanFilter as EKF

# ------------------ Simulation parameters ------------------
#np.random.seed(0)

n_steps = 600           # longer run to let errors accumulate
pilot_period = 10       # fewer pilots -> harder interpolation
use_two_axis_pilots = True
A_pilot = 2           # weaker pilots (noisier pilot info)

# Channel & receiver
eta_ch   = 0.20
eta_det  = 0.70
eta_tot  = eta_ch * eta_det
g_amp    = np.sqrt(eta_tot)

# Phase process
process_var = 5e-3      # much larger phase diffusion -> stronger nonlinearity
process_std = np.sqrt(process_var)

# Noise (shot-noise units)
meas_var = 3       # higher measurement noise
meas_std = np.sqrt(meas_var)

# Modulation variance
V_A = 6.0
# -----------------------------------------------------------

# ---- Build pilot mask & pilot symbols (xA, pA) ----
is_pilot = np.zeros(n_steps, dtype=bool)
pilot_indices = []
i = 0
while i < n_steps:
    is_pilot[i] = True
    pilot_indices.append(i)
    if use_two_axis_pilots and i+1 < n_steps:
        is_pilot[i+1] = True
        pilot_indices.append(i+1)
        i += pilot_period
    else:
        i += pilot_period

xA = np.zeros(n_steps)
pA = np.zeros(n_steps)

if use_two_axis_pilots:
    toggle = 0
    for k in pilot_indices:
        if toggle == 0:
            xA[k], pA[k] = A_pilot, 0.0
            toggle = 1
        else:
            xA[k], pA[k] = 0.0, A_pilot
            toggle = 0
else:
    xA[is_pilot] = A_pilot
    pA[is_pilot] = 0.0

# Data (unknown to Bob)
xA[~is_pilot] = np.random.normal(0, np.sqrt(V_A), size=np.sum(~is_pilot))
pA[~is_pilot] = np.random.normal(0, np.sqrt(V_A), size=np.sum(~is_pilot))

# ---- Bob randomly chooses quadrature each slot ----
bob_basis = np.random.choice([0, 1], size=n_steps)  # 0 = X, 1 = P

def homodyne(phi, xAk, pAk, basis, g=g_amp):
    if basis == 0:  # X quadrature
        return g * (xAk * np.cos(phi) + pAk * np.sin(phi))
    else:           # P quadrature
        return g * (-xAk * np.sin(phi) + pAk * np.cos(phi))

# ---- Generate measurements ----
phi_true = np.zeros(n_steps)
y = np.zeros(n_steps)

phi = 0.0
for k in range(n_steps):
    phi += np.random.normal(0, process_std)   # phase drift
    phi_true[k] = phi
    y[k] = homodyne(phi, xA[k], pA[k], bob_basis[k]) + np.random.normal(0, meas_std)

# Ideal measurements (without noise/phase drift)
y_ideal = np.array([homodyne(0, xA[k], pA[k], bob_basis[k]) for k in range(n_steps)])

# ------------------ UKF ------------------
def fx(phi, dt):
    return phi  # random walk

def hx(phi, xAk, pAk, basis):
    return homodyne(phi, xAk, pAk, basis)

sigmas = MerweScaledSigmaPoints(n=1, alpha=0.5, beta=2, kappa=1.)
ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=None, dt=1, points=sigmas)
ukf.x = np.array([0.0])  # initial phase
ukf.P *= 0.1
ukf.R = meas_var
ukf.Q = process_var

phi_est_ukf = np.zeros(n_steps)
y_corr_ukf = np.zeros(n_steps)

for k in range(n_steps):
    ukf.predict()
    if is_pilot[k]:
        ukf.update(y[k], hx=lambda phi: hx(phi, xA[k], pA[k], bob_basis[k]))
    phi_est_ukf[k] = ukf.x[0]
    y_corr_ukf[k] = y[k] * np.cos(phi_est_ukf[k])

# ------------------ EKF ------------------
def H_jacobian(phi, xAk, pAk, basis, g=g_amp):
    if basis == 0:  # X quadrature
        return np.array([[g * (-xAk * np.sin(phi) + pAk * np.cos(phi))]])
    else:           # P quadrature
        return np.array([[-g * (xAk * np.cos(phi) + pAk * np.sin(phi))]])

def h_meas(phi, xAk, pAk, basis):
    return np.array([homodyne(phi, xAk, pAk, basis)])

ekf = EKF(dim_x=1, dim_z=1)
ekf.x = np.array([0.0])
ekf.P *= 0.1
ekf.R = np.array([[meas_var]])
ekf.Q = np.array([[process_var]])

phi_est_ekf = np.zeros(n_steps)
y_corr_ekf = np.zeros(n_steps)

for k in range(n_steps):
    ekf.F = np.array([[1]])
    ekf.predict()
    if is_pilot[k]:
        H = H_jacobian(ekf.x[0], xA[k], pA[k], bob_basis[k])
        ekf.update(y[k], HJacobian=lambda x: H,
                         Hx=lambda x: h_meas(x[0], xA[k], pA[k], bob_basis[k]))
    phi_est_ekf[k] = ekf.x[0]
    y_corr_ekf[k] = y[k] * np.cos(phi_est_ekf[k])

# ------------------ Metrics ------------------
def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

# Phase tracking error
rmse_phi_ukf = rmse(phi_est_ukf, phi_true)
rmse_phi_ekf = rmse(phi_est_ekf, phi_true)

# Measurement correction error
mse_corr_ukf = np.mean((y_corr_ukf - y_ideal)**2)
mse_corr_ekf = np.mean((y_corr_ekf - y_ideal)**2)

# SNR
def snr(signal, noise):
    return 10*np.log10(np.var(signal)/np.var(noise))

snr_noisy = snr(y_ideal, y - y_ideal)
snr_ukf   = snr(y_ideal, y_corr_ukf - y_ideal)
snr_ekf   = snr(y_ideal, y_corr_ekf - y_ideal)

print("----- Metrics -----")
print(f"Phase RMSE: UKF={rmse_phi_ukf:.4f}, EKF={rmse_phi_ekf:.4f}")
print(f"Correction MSE: UKF={mse_corr_ukf:.4f}, EKF={mse_corr_ekf:.4f}")
print(f"SNR (dB): Noisy={snr_noisy:.2f}, UKF={snr_ukf:.2f}, EKF={snr_ekf:.2f}")

# ------------------ Plots ------------------
fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)
axs[0].plot(phi_true, label="True phase")
axs[0].plot(phi_est_ukf, '--', label="UKF est.")
axs[0].plot(phi_est_ekf, '--', label="EKF est.")
axs[0].set_title("Phase Tracking")
axs[0].legend()

axs[1].plot(y, label='Noisy', alpha=0.6)
axs[1].plot(y_ideal, label='Ideal', lw=2)
axs[1].set_title("Noisy vs Ideal")
axs[1].legend()

axs[2].plot(y_ideal, label='Ideal', lw=2)
axs[2].plot(y_corr_ukf, '--', label='UKF corrected')
axs[2].plot(y_corr_ekf, '--', label='EKF corrected')
axs[2].set_title("Ideal vs Corrected")
axs[2].legend()

plt.tight_layout()
plt.show()

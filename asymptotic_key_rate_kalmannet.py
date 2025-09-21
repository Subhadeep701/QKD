import os
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Import repo modules
import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from KNet.KalmanNet_nn import KalmanNetNN
from Pipelines.Pipeline_EKF import Pipeline_EKF
import torch.nn as nn

# --------------------
# Time string utility
# --------------------
def now_string():
    today = datetime.today().strftime("%m.%d.%y")
    clock = datetime.now().strftime("%H:%M:%S")
    return f"{today}_{clock}"

print("CV-QKD Phase Tracking Pipeline Start")
print("Current Time =", now_string())

# --------------------
# General settings
# --------------------
args = config.general_settings()
args.N_E = 100 # number of training sequences
args.N_CV = 50  # number validation sequences
args.N_T = 500    # number test sequences
args.T = 5     # sequence length
args.T_test = 5

args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True

args.use_cuda = False
args.n_steps = 15
args.n_batch = 25
args.lr = 1e-3
args.wd = 1e-4

device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
print("Using", device)

# --------------------
# CV-QKD Specific Parameters
# --------------------
m = 1  # State dimension (phase offset)
n = 2  # Measurement dimension (I and Q quadratures)

# Laser phase noise (Wiener process)
phase_noise_variance = 0.05

# --------------------
# State transition function
# --------------------
def f(x, jacobian=False):
    if jacobian:
        x_next = x
        F = torch.ones_like(x).view(x.shape[0], m, m)
        return x_next, F
    else:
        return x

# --------------------
# Measurement function
# --------------------
def h(x, jacobian=False):
    A = 2.0
    phi_signal = 0.0
    if jacobian:
        I = A * torch.cos(x + phi_signal)
        Q = A * torch.sin(x + phi_signal)
        y = torch.cat([I, Q], dim=1)
        dI_dtheta = -A * torch.sin(x + phi_signal).view(x.shape[0], 1, m)
        dQ_dtheta = A * torch.cos(x + phi_signal).view(x.shape[0], 1, m)
        H = torch.cat([dI_dtheta, dQ_dtheta], dim=1)
        return y, H
    else:
        I = A * torch.cos(x + phi_signal)
        Q = A * torch.sin(x + phi_signal)
        return torch.cat([I, Q], dim=1)

# --------------------
# Noise parameters
# --------------------
Q = phase_noise_variance * torch.eye(m)
measurement_noise_variance = 0.01
R = measurement_noise_variance * torch.eye(n)

# --------------------
# System model
# --------------------
sys_model = SystemModel(f=f, Q=Q, h=h, R=R, T=args.T, T_test=args.T_test, m=m, n=n)
m1x_0 = torch.zeros(m, 1)
m2x_0 = 0.01 * torch.eye(m)
sys_model.InitSequence(m1x_0, m2x_0)

# --------------------
# Data generation
# --------------------
data_dir = 'Simulations/CVQKD/data'
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(data_dir, 'data_cvqkd_phase.pt')

print("Start CV-QKD Data Gen")
DataGen(args, sys_model, data_file)
print("Load Data:", data_file)

# Load the data
train_input, train_target, cv_input, cv_target, test_input, test_target, _, _, _ = torch.load(
    data_file, map_location=device
)
print(f"Train input shape: {train_input.shape}, Train target shape: {train_target.shape}")

# --------------------
# KalmanNet
# --------------------
print("\n=== Train KalmanNet ===")
KNet = KalmanNetNN()
KNet.NNBuild(sys_model, args)

pipe = Pipeline_EKF(now_string(), "KNet", "KNet")
pipe.setssModel(sys_model)
pipe.setModel(KNet)
pipe.setTrainingParams(args)

MSE_cv_lin_ep, MSE_cv_dB_ep, MSE_tr_lin_ep, MSE_tr_dB_ep = pipe.NNTrain(
    sys_model, cv_input, cv_target, train_input, train_target, path_results='KNet/'
)

print("\n=== Test KalmanNet ===")
MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime = pipe.NNTest(
    sys_model, test_input, test_target, path_results='KNet/'
)
print("KNet MSE avg [lin]:", MSE_test_linear_avg.item())
print("KNet MSE avg [dB]: ", MSE_test_dB_avg.item())
# =====================================================================
# === CV-QKD Payload Continuation + KalmanNet-based Correction & Key Rate
# =====================================================================
print("\n=== CV-QKD Payload Generation & Key Rate (KalmanNet) ===")

# Convenience copies of pilot results
test_input_np  = test_input.cpu().numpy()
test_target_np = test_target.cpu().numpy().squeeze()   # [N_test, T]
KNet_out_np    = KNet_out.detach().numpy().squeeze()      # [N_test, T]

N_test = test_target_np.shape[0]
T_pilot = test_target_np.shape[1]
T_payload = 2  # payload length equals pilot test length

# -------------------------------------------------
# 1) Generate payload true phases as AR(1) continuation
# -------------------------------------------------
phase_var = float(phase_noise_variance)
rng = np.random.default_rng(2025)

phi_payload_true = np.zeros((N_test, T_payload))
for i in range(N_test):
    phi_payload_true[i, 0] = test_target_np[i, -1]  # first payload = last pilot
    phi = phi_payload_true[i, 0]
    for t in range(1, T_payload):                  # start from t=1
        phi += rng.normal(0.0, phase_var)
        phi_payload_true[i, t] = phi

# -------------------------------------------------
# 2) Simulate Alice payload symbols & Bob measurements
# -------------------------------------------------
Vmod = 3.0    # Alice modulation variance (shot-noise units)
A    = 2.0    # channel amplitude (consistent with h(x))
meas_var = float(measurement_noise_variance)

alice_payload = (rng.normal(scale=np.sqrt(Vmod), size=(N_test, T_payload))
                 + 1j * rng.normal(scale=np.sqrt(Vmod), size=(N_test, T_payload)))

meas_payload = np.zeros((N_test, 2, T_payload))
for i in range(N_test):
    s = alice_payload[i]
    phi = phi_payload_true[i]
    r = s * np.exp(1j * phi)                      # channel rotation
    meas_payload[i,0,:] = np.real(r) + rng.normal(0.0, np.sqrt(meas_var), T_payload)
    meas_payload[i,1,:] = np.imag(r) + rng.normal(0.0, np.sqrt(meas_var), T_payload)

# -------------------------------------------------
# 3) KalmanNet payload phase estimate (linear extrapolation from pilots)
# -------------------------------------------------
# def extrapolate_pilot(phi_pilot):
#     if len(phi_pilot) >= 2:
#         slope = phi_pilot[-1] - phi_pilot[-2]
#     else:
#         slope = 0.0
#     t = np.arange(1, T_payload + 1)
#     return phi_pilot[-1] + slope * t
#
# phi_kn_payload = np.vstack([extrapolate_pilot(KNet_out_np[i]) for i in range(N_test)])
phi_kn_payload = np.tile(KNet_out_np[:, -1][:, None], (1, T_payload))

# -------------------------------------------------
# 4) Rotate Bob’s payload measurements using KalmanNet estimate
# -------------------------------------------------
I = meas_payload[:,0,:]
Q = meas_payload[:,1,:]
c = np.cos(phi_kn_payload)
s = np.sin(phi_kn_payload)
Xb_corr = I * c + Q * s   # corrected X quadrature (Bob)
Pb_corr = Q * c - I * s   # corrected P quadrature

Xa = np.real(alice_payload)  # Alice X quadrature

# -------------------------------------------------
# 5) CV-QKD statistics & key rate
# -------------------------------------------------
def G(x):
    if x <= 1e-12:
        return 0.0
    return (x+1)*np.log2(x+1) - x*np.log2(x)

def mutual_info_and_channel(Xa, Xb):
    a = Xa.ravel()
    b = Xb.ravel()
    V_A = np.var(a, ddof=0)
    V_B = np.var(b, ddof=0)
    C_AB = np.cov(a, b, ddof=0)[0,1]
    V_B_given_A = max(V_B - (C_AB**2)/(V_A + 1e-12), 1e-12)
    Iab = 0.5 * np.log2(V_B / V_B_given_A)
    # Channel parameter estimates
    sqrtT = C_AB / (V_A + 1e-12)
    T_est = sqrtT**2
    xi_est = (V_B - T_est*V_A - 1.0) / max(T_est,1e-12)
    return Iab, V_A, V_B, C_AB, T_est, xi_est

Iab, V_A, V_B, C_AB, T_est, xi_est = mutual_info_and_channel(Xa, Xb_corr)

# Holevo bound (Gaussian approx.)
Va = Vmod + 1.0
a = Va
b = T_est * Va + 1.0 + T_est * xi_est
c_AB = np.sqrt(T_est) * np.sqrt(max(Va**2 - 1.0, 0.0))
det = a*b - c_AB**2
term_inside = max((a + b)**2 - 4*det, 0.0)
term = np.sqrt(term_inside)
nu_plus  = 0.5 * ((a + b) + term)
nu_minus = 0.5 * ((a + b) - term)
s1 = max((nu_plus  - 1.0)/2.0, 0.0)
s2 = max((nu_minus - 1.0)/2.0, 0.0)
nu3 = np.sqrt( max( b * (b - (c_AB**2)/a), 0.0 ) )
s3 = max((nu3 - 1.0)/2.0, 0.0)
chi_BE = G(s1) + G(s2) - G(s3)

beta = 0.95  # reconciliation efficiency
key_rate = beta * Iab - chi_BE
print("\n--- Pilot vs Payload Phase ---")
print("1st sequence pilot phase           :", test_target_np[0, :])
print("1st sequence pilot estimate  :", KNet_out_np[0, :])
print("1st sequence true payload phase    :", phi_payload_true[0, :])
print("1st sequence extrapolated payload  :", phi_kn_payload[0, :])  # ← Added line
print()
print("3rd sequence pilot phase           :", test_target_np[2, :])
print("3rd sequence pilot estimate  :", KNet_out_np[2, :])
print("3rd sequence true payload phase    :", phi_payload_true[2, :])
print("3rd sequence extrapolated payload  :", phi_kn_payload[2, :])  # ← Added line




# -------------------------------------------------
# 6) Performance summaries
# -------------------------------------------------
rmse_payload = np.sqrt(np.mean((phi_payload_true - phi_kn_payload)**2))
print(f"Payload KalmanNet phase RMSE      : {rmse_payload:.6e} rad")
print(f"Mutual information I_AB           : {Iab:.6f} bits/use")
print(f"Estimated channel T               : {T_est:.6f}")
print(f"Estimated excess noise xi         : {xi_est:.6e} SNU")
print(f"Holevo bound chi_BE               : {chi_BE:.6f} bits/use")
print(f"Asymptotic key rate K (beta=0.95) : {key_rate:.6f} bits/use")

# # Optional: scatter plot for quick sanity check
# plt.figure(figsize=(5,5))
# plt.scatter(Xa.ravel(), Xb_corr.ravel(), s=8, alpha=0.5)
# plt.xlabel("Alice X")
# plt.ylabel("Bob corrected X (KalmanNet)")
# plt.title("Payload: Alice vs Bob corrected X")
# plt.grid(True); plt.axis('equal')
# plt.show()

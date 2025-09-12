import os
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import repo modules
import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from KNet.KalmanNet_nn import KalmanNetNN
from Pipelines.Pipeline_EKF import Pipeline_EKF


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
args.N_E = 600000
args.N_CV = 60000
args.N_T = 1000
args.T = 6
args.T_test = 6

args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True

args.use_cuda = False
args.n_steps = 60
args.n_batch = 500
args.lr = 1e-3
args.wd = 1e-4

device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
print("Using", device)

# --------------------
# CV-QKD Specific Parameters
# --------------------
m = 1  # State dimension (phase offset)
n = 2  # Measurement dimension (I and Q quadratures)

phase_noise_variance = 0.01
measurement_noise_variance = 0.01

# State transition (Wiener process for phase noise)
def f(x, jacobian=False):
    if jacobian:
        x_next = x
        F = torch.ones_like(x).view(x.shape[0], m, m)
        return x_next, F
    else:
        return x

# Measurement function (IQ demodulation)
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

# Noise parameters
Q = phase_noise_variance * torch.eye(m)
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
Q_wrong = 0.01 * torch.eye(m)   # underestimated process noise
R_wrong = 0.1 * torch.eye(n)   # overestimated measurement noise

sys_model_wrong = SystemModel(f=f, Q=Q_wrong, h=h, R=R_wrong,
                             T=args.T, T_test=args.T_test, m=m, n=n)
sys_model_wrong.InitSequence(m1x_0, m2x_0)
data_dir = 'Simulations/CVQKD/data'
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(data_dir, 'data_cvqkd_phase.pt')

print("Start CV-QKD Data Gen")
DataGen(args, sys_model_wrong, data_file)
print("Load Data:", data_file)

train_input, train_target, cv_input, cv_target, test_input, test_target, _, _, _ = torch.load(
    data_file, map_location=device
)

print(f"Train input shape: {train_input.shape}")
print(f"Train target shape: {train_target.shape}")

# --------------------
# KalmanNet for CV-QKD
# --------------------
print("\n=== Train KalmanNet for CV-QKD ===")
KNet = KalmanNetNN()
KNet.NNBuild(sys_model, args)

pipe = Pipeline_EKF(now_string(), "KNet", "KNet")
pipe.setssModel(sys_model)
pipe.setModel(KNet)
pipe.setTrainingParams(args)

MSE_cv_lin_ep, MSE_cv_dB_ep, MSE_tr_lin_ep, MSE_tr_dB_ep = pipe.NNTrain(
    sys_model, cv_input, cv_target, train_input, train_target, path_results='KNet/'
)

print("\n=== Test KalmanNet for CV-QKD ===")
MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime = pipe.NNTest(
    sys_model, test_input, test_target, path_results='KNet/'
)
print("KNet MSE avg [lin]:", MSE_test_linear_avg.item())
print("KNet MSE avg [dB]: ", MSE_test_dB_avg.item())

# --------------------
# EKF / UKF baselines
# --------------------
from filterpy.kalman import ExtendedKalmanFilter as EKF_fp
from filterpy.kalman import UnscentedKalmanFilter as UKF_fp, MerweScaledSigmaPoints

A_amp = 2.0
phi_signal = 0.0
Q_proc = float(phase_noise_variance)
R_meas = float(measurement_noise_variance)

test_input_np  = test_input.cpu().numpy()
test_target_np = test_target.cpu().numpy()
KNet_out_np    = np.stack([o.cpu().numpy() for o in KNet_out]) if isinstance(KNet_out, (list,tuple)) else KNet_out.detach().cpu().numpy()

N_test = test_input_np.shape[0]
T_len  = test_input_np.shape[2]

def ekf_hx(x):
    phi = float(x[0])
    return np.array([ A_amp * np.cos(phi + phi_signal),
                      A_amp * np.sin(phi + phi_signal) ])

def ekf_H_jacobian(x):
    phi = float(x[0])
    dI = -A_amp * np.sin(phi + phi_signal)
    dQ =  A_amp * np.cos(phi + phi_signal)
    return np.array([[dI],[dQ]])

def ukf_hx(x):
    phi = float(x[0])
    return np.array([ A_amp * np.cos(phi + phi_signal),
                      A_amp * np.sin(phi + phi_signal) ])

def ukf_fx(x, dt):
    return x.copy()

phi_est_ekf_all = np.zeros((N_test, T_len))
phi_est_ukf_all = np.zeros((N_test, T_len))

P0 = 1e-2
for i in range(N_test):
    meas_seq = test_input_np[i]
    true_phi = test_target_np[i].reshape(-1)

    ekf = EKF_fp(dim_x=1, dim_z=2)
    ekf.x = np.array([0.0])
    ekf.P = np.array([[P0]])
    ekf.R = np.eye(2) * R_meas
    ekf.Q = np.array([[Q_proc]])

    phi_ekf_est = np.zeros(T_len)
    for k in range(T_len):
        ekf.F = np.array([[1.0]])
        ekf.predict()
        z = meas_seq[:, k]
        ekf.update(z, HJacobian=ekf_H_jacobian, Hx=ekf_hx)
        phi_ekf_est[k] = float(ekf.x[0])
    phi_est_ekf_all[i] = phi_ekf_est

    points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=0.0)
    ukf = UKF_fp(dim_x=1, dim_z=2, fx=ukf_fx, hx=ukf_hx, dt=1.0, points=points)
    ukf.x = np.array([0.0])
    ukf.P = np.array([[P0]])
    ukf.R = np.eye(2) * R_meas
    ukf.Q = np.array([[Q_proc]])

    phi_ukf_est = np.zeros(T_len)
    for k in range(T_len):
        ukf.predict()
        z = meas_seq[:, k]
        ukf.update(z)
        phi_ukf_est[k] = float(ukf.x[0])
    phi_est_ukf_all[i] = phi_ukf_est

true_phi_all = test_target_np.reshape(N_test, T_len)

mse_kn = np.mean((true_phi_all - KNet_out_np.reshape(N_test, T_len))**2)
mse_ekf = np.mean((true_phi_all - phi_est_ekf_all)**2)
mse_ukf = np.mean((true_phi_all - phi_est_ukf_all)**2)

print("\n--- Baseline comparison ---")
print(f"KalmanNet RMSE : {np.sqrt(mse_kn):.6f} rad")
print(f"EKF       RMSE : {np.sqrt(mse_ekf):.6f} rad")
print(f"UKF       RMSE : {np.sqrt(mse_ukf):.6f} rad")

# --------------------
# Plot comparison
# --------------------
idx = 0
t = np.arange(T_len)
plt.figure(figsize=(12,5))
plt.plot(t, true_phi_all[idx], label='True phase', color='k', linewidth=2)
plt.plot(t, KNet_out_np[idx].reshape(T_len), label='KalmanNet', linestyle='--')
plt.plot(t, phi_est_ekf_all[idx], label='EKF', linestyle='-.')
plt.plot(t, phi_est_ukf_all[idx], label='UKF', linestyle=':')
plt.xlabel('time index')
plt.ylabel('phase (rad)')
plt.title('Phase tracking comparison (sequence 0)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

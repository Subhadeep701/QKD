import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

# ==============================================================
# Import repo modules
# ==============================================================
import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from KNet.KalmanNet_nn import KalmanNetNN
from Pipelines.Pipeline_EKF import Pipeline_EKF
from filterpy.kalman import ExtendedKalmanFilter as EKF_fp
from filterpy.kalman import UnscentedKalmanFilter as UKF_fp, MerweScaledSigmaPoints


# ==============================================================
# Utility
# ==============================================================
def now_string():
    today = datetime.today().strftime("%m.%d.%y")
    clock = datetime.now().strftime("%H:%M:%S")
    return f"{today}_{clock}"

print("CV-QKD Phase Tracking Pipeline Start")
print("Current Time =", now_string())


# ==============================================================
# General settings
# ==============================================================
args = config.general_settings()
args.N_E = 100    # training sequences
args.N_CV = 50    # validation sequences
args.N_T = 10000     # test sequences
args.T = 7      # sequence length
args.T_test = 7
args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True
args.use_cuda = False
args.n_steps = 10
args.n_batch = 25
args.lr = 2*1e-3
args.wd = 1e-4

device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
print("Using", device)


# ==============================================================
# CV-QKD Specific Parameters
# ==============================================================
m = 1  # state dimension (phase)
n = 2  # measurement dimension (I,Q)

phase_noise_variance = 0.02
measurement_noise_variance = 0.02
Q_wrong = 0.02 * torch.eye(m)
R_wrong = 0.02 * torch.eye(n)

def f(x, jacobian=False):
    if jacobian:
        x_next = x
        F = torch.ones_like(x).view(x.shape[0], m, m)
        return x_next, F
    else:
        return x

def h(x, jacobian=False):
    A = 2.0
    phi_signal = 0.0
    if jacobian:
        I = A * torch.cos(x + phi_signal)
        Q = A * torch.sin(x + phi_signal)
        y = torch.cat([I, Q], dim=1)
        dI_dtheta = -A * torch.sin(x + phi_signal).view(x.shape[0], 1, m)
        dQ_dtheta =  A * torch.cos(x + phi_signal).view(x.shape[0], 1, m)
        H = torch.cat([dI_dtheta, dQ_dtheta], dim=1)
        return y, H
    else:
        I = A * torch.cos(x + phi_signal)
        Q = A * torch.sin(x + phi_signal)
        return torch.cat([I, Q], dim=1)

Q = phase_noise_variance * torch.eye(m)
R = measurement_noise_variance * torch.eye(n)


# ==============================================================
# System model
# ==============================================================
sys_model = SystemModel(f=f, Q=Q, h=h, R=R, T=args.T, T_test=args.T_test, m=m, n=n)
m1x_0 = 0.01 * torch.ones(m, 1)
m2x_0 = 0.001 * torch.eye(m)
sys_model.InitSequence(m1x_0, m2x_0)

sys_model_wrong = SystemModel(f=f, Q=Q_wrong, h=h, R=R_wrong,
                              T=args.T, T_test=args.T_test, m=m, n=n)
sys_model_wrong.InitSequence(m1x_0, m2x_0)


# ==============================================================
# Data generation
# ==============================================================
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


# ==============================================================
# KalmanNet training/testing
# ==============================================================
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


# ==============================================================
# Simple RNN (LSTM) baseline
# ==============================================================
print("\n=== Train RNN baseline ===")

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# prepare data for RNN: [batch, seq, feature]
train_input_rnn = train_input.permute(0,2,1)
train_target_rnn = train_target.permute(0,2,1)
cv_input_rnn = cv_input.permute(0,2,1)
cv_target_rnn = cv_target.permute(0,2,1)
test_input_rnn = test_input.permute(0,2,1)
test_target_rnn = test_target.permute(0,2,1)

rnn = SimpleRNN(input_dim=n, hidden_dim=32, output_dim=m, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=args.wd)

train_dataset = torch.utils.data.TensorDataset(train_input_rnn, train_target_rnn)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.n_batch,
                                           shuffle=True)

for epoch in range(args.n_steps):
    rnn.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = rnn(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"[RNN] Epoch {epoch+1}/{args.n_steps}, Loss: {epoch_loss/len(train_loader):.6f}")

rnn.eval()
with torch.no_grad():
    test_input_device = test_input_rnn.to(device)
    test_target_device = test_target_rnn.to(device)
    rnn_out = rnn(test_input_device)
    mse_rnn = criterion(rnn_out, test_target_device).item()
    mse_rnn_dB = 10 * np.log10(mse_rnn)
    print(f"RNN MSE [lin]: {mse_rnn:.6e}")
    print(f"RNN MSE [dB]: {mse_rnn_dB:.6f} dB")


# ==============================================================
# EKF / UKF baselines
# ==============================================================
print("\n=== EKF/UKF Baselines ===")

A_amp = 2.0
phi_signal = 0.0
Q_proc = float(phase_noise_variance**2)
R_meas = float(measurement_noise_variance**2)

test_input_np  = test_input.cpu().numpy()
test_target_np = test_target.cpu().numpy()
KNet_out_np    = (np.stack([o.cpu().numpy() for o in KNet_out])
                  if isinstance(KNet_out,(list,tuple))
                  else KNet_out.detach().cpu().numpy())
N_test = test_input_np.shape[0]
T_len  = test_input_np.shape[2]

def ekf_hx(x):
    phi = float(x[0])
    return np.array([A_amp*np.cos(phi+phi_signal),
                     A_amp*np.sin(phi+phi_signal)])

def ekf_H_jacobian(x):
    phi = float(x[0])
    dI = -A_amp*np.sin(phi+phi_signal)
    dQ =  A_amp*np.cos(phi+phi_signal)
    return np.array([[dI],[dQ]])

def ukf_hx(x):
    phi = float(x[0])
    return np.array([A_amp*np.cos(phi+phi_signal),
                     A_amp*np.sin(phi+phi_signal)])

def ukf_fx(x, dt):
    return x.copy()

phi_est_ekf_all = np.zeros((N_test, T_len))
phi_est_ukf_all = np.zeros((N_test, T_len))

P0 = 1e-2
for i in range(N_test):
    meas_seq = test_input_np[i]
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
KNet_out_np = KNet_out_np.reshape(N_test, T_len)
rnn_out_np  = rnn_out.cpu().numpy().reshape(N_test, T_len)

mse_kn  = np.mean((true_phi_all - KNet_out_np)**2)
mse_ekf = np.mean((true_phi_all - phi_est_ekf_all)**2)
mse_ukf = np.mean((true_phi_all - phi_est_ukf_all)**2)
mse_rnn = np.mean((true_phi_all - rnn_out_np)**2)

print("\n--- Baseline comparison ---")
print(f"KalmanNet RMSE : {np.sqrt(mse_kn):.6f} rad")
print(f"EKF       RMSE : {np.sqrt(mse_ekf):.6f} rad")
print(f"UKF       RMSE : {np.sqrt(mse_ukf):.6f} rad")
print(f"RNN       RMSE : {np.sqrt(mse_rnn):.6f} rad")


# ==============================================================
# Plot comparison
# ==============================================================
idx = 0
t = np.arange(T_len)
plt.figure(figsize=(12,5))
plt.plot(t, true_phi_all[idx], label='True phase', color='k', linewidth=2)
plt.plot(t, KNet_out_np[idx], label='KalmanNet', linestyle='--')
plt.plot(t, phi_est_ekf_all[idx], label='EKF', linestyle='-.')
plt.plot(t, phi_est_ukf_all[idx], label='UKF', linestyle=':')
plt.plot(t, rnn_out_np[idx], label='RNN (LSTM)', linestyle='-', marker='o', markersize=4, linewidth=1)
plt.xlabel('time index')
plt.ylabel('phase (rad)')
plt.title('Phase tracking comparison (sequence 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

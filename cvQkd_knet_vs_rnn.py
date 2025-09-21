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
args.N_E = 1000 # number of training sequences
args.N_CV = 500  # number validation sequences
args.N_T = 500    # number test sequences
args.T = 10      # sequence length
args.T_test = 10

args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True

args.use_cuda = False
args.n_steps = 15
args.n_batch = 100
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
phase_noise_variance = 0.01

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

# --------------------
# Define RNN
# --------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# --------------------
# Prepare RNN data (LSTM expects [batch, seq, feature])
# --------------------
train_input_rnn = train_input.permute(0,2,1)
train_target_rnn = train_target.permute(0,2,1)
cv_input_rnn = cv_input.permute(0,2,1)
cv_target_rnn = cv_target.permute(0,2,1)
test_input_rnn = test_input.permute(0,2,1)
test_target_rnn = test_target.permute(0,2,1)

# --------------------
# Train RNN
# --------------------
rnn = SimpleRNN(input_dim=n, hidden_dim=32, output_dim=m, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = args.n_steps
batch_size = args.n_batch

train_dataset = torch.utils.data.TensorDataset(train_input_rnn, train_target_rnn)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("\n=== Train RNN ===")
for epoch in range(num_epochs):
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.6f}")

# --------------------
# Test RNN
# --------------------
rnn.eval()
with torch.no_grad():
    test_input_device = test_input_rnn.to(device)
    test_target_device = test_target_rnn.to(device)
    rnn_out = rnn(test_input_device)
    mse_rnn = criterion(rnn_out, test_target_device).item()
    mse_rnn_dB = 10 * np.log10(mse_rnn)

print("\n=== Comparison ===")
print(f"KalmanNet MSE [lin]: {MSE_test_linear_avg.item():.6f}, [dB]: {MSE_test_dB_avg:.3f}")
print(f"RNN       MSE [lin]: {mse_rnn:.6f}, [dB]: {mse_rnn_dB:.3f}")
# --------------------
# Compare first 10 sequences
# --------------------
print("\n=== First 10 Test Sequences Comparison ===")
num_show = 4

# KalmanNet outputs are already obtained in KNet_out
# KNet_out shape: [num_sequences, m, T]
# Convert to CPU for printing
KNet_out_cpu = KNet_out[:num_show].detach().squeeze().cpu().numpy()
test_target_cpu = test_target[:num_show].cpu().numpy()
rnn_out_cpu = rnn_out[:num_show].cpu().numpy().transpose(0,2,1)  # reshape [batch, m, T]

# for i in range(num_show):
#     print(f"\nSequence {i+1}:")
#     print("True Target:       ", np.round(test_target_cpu[i], 4))
#     print("KalmanNet Predicted:", np.round(KNet_out_cpu[i], 4))
#     print("RNN Predicted:     ", np.round(rnn_out_cpu[i], 4))

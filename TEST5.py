"""
cvqkd_kalmannet_tsp.py

Single-file experiment:
 - Generate CV-QKD-like data with AR(1) phase drift
 - Train a KalmanNet-TSP-style MLP to estimate phase at pilot instants
 - Compute Phase MSE and corrected-measurement MSE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------
# Repro & device
# ---------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Simulation params
# ---------------------------
n = 20000           # total symbols
m_param = 20        # modulation parameter (used for generating q,p)
T = 0.9
h = 0.85
gain = np.sqrt(T * h)
sigma_meas = 0.05   # measurement noise variance (per quadrature)
phase_var = 1e-1
ar_coeff = 0.99
pilot_interval = 20
window_size = 30    # sliding window used by KalmanNet
train_fraction = 0.6
val_fraction = 0.2
# rest is test

# ---------------------------
# Generate CV-QKD-like data
# ---------------------------
q = np.random.normal(0, np.sqrt(m_param - 1), n)
p = np.random.normal(0, np.sqrt(m_param - 1), n)

phi = np.zeros(n)
phi[0] = np.random.normal(0, np.sqrt(phase_var))
for i in range(1, n):
    phi[i] = ar_coeff * phi[i-1] + np.random.normal(0, np.sqrt(phase_var))

zq = np.random.normal(0, np.sqrt(sigma_meas), n)
zp = np.random.normal(0, np.sqrt(sigma_meas), n)

qB = np.zeros(n)
pB = np.zeros(n)
for i in range(n):
    q_rot = q[i] * np.cos(phi[i]) - p[i] * np.sin(phi[i])
    p_rot = q[i] * np.sin(phi[i]) + p[i] * np.cos(phi[i])
    qB[i] = gain * q_rot + zq[i]
    pB[i] = gain * p_rot + zp[i]

# pilot indices
pilot_idxs = np.arange(window_size, n, pilot_interval)

# ---------------------------
# Prepare dataset
# Input at pilot k:
#   [qB_window, pB_window, prev_phi, q_k, p_k]
# Target: phi[k]
# ---------------------------
def build_dataset(pilot_idxs):
    X, Y = [], []
    for k in pilot_idxs:
        q_win = qB[k-window_size:k]
        p_win = pB[k-window_size:k]
        prev_phi = phi[k-1]
        qk, pk = q[k], p[k]
        x = np.concatenate([q_win, p_win, [prev_phi, qk, pk]])
        X.append(x)
        Y.append(phi[k])
    return np.array(X), np.array(Y)

X_all, Y_all = build_dataset(pilot_idxs)

# split train/val/test
num = X_all.shape[0]
i_train = int(num * train_fraction)
i_val = int(num * (train_fraction + val_fraction))

X_train, Y_train = X_all[:i_train], Y_all[:i_train]
X_val, Y_val = X_all[i_train:i_val], Y_all[i_train:i_val]
X_test, Y_test = X_all[i_val:], Y_all[i_val:]

# normalize inputs
X_mean, X_std = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-9
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# to torch
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val_t = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)

# ---------------------------
# KalmanNet-TSP style MLP
# ---------------------------
class KalmanNetTSP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train_t.shape[1]
model = KalmanNetTSP(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.MSELoss()

# ---------------------------
# Train
# ---------------------------
batch_size = 256
num_epochs = 200
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_t, Y_train_t),
    batch_size=batch_size, shuffle=True
)

best_val = 1e12
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    # val
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), Y_val_t).item()
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_kalmannet.pth")
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}  Val MSE: {val_loss:.3e}")

# load best
model.load_state_dict(torch.load("best_kalmannet.pth", map_location=device))

# ---------------------------
# Test
# ---------------------------
model.eval()
with torch.no_grad():
    pred_test = model(X_test_t).cpu().numpy().ravel()

# Phase MSE
phase_mse_kn = np.mean((Y_test - pred_test)**2)

# Measurement MSE
test_pilot_idxs = pilot_idxs[i_val:]
q_corr, p_corr = np.zeros(len(test_pilot_idxs)), np.zeros(len(test_pilot_idxs))
for ii, k in enumerate(test_pilot_idxs):
    ph = pred_test[ii]
    q_corr[ii] = (qB[k] * np.cos(ph) + pB[k] * np.sin(ph)) / gain
    p_corr[ii] = (-qB[k] * np.sin(ph) + pB[k] * np.cos(ph)) / gain
mse_meas_kn = np.mean((q[test_pilot_idxs] - q_corr)**2 + (p[test_pilot_idxs] - p_corr)**2)

# ---------------------------
# Results
# ---------------------------
print("\n===== RESULTS (KalmanNet-TSP) =====")
print(f"Phase MSE: {phase_mse_kn:.6e}")
print(f"Measurement MSE: {mse_meas_kn:.6e}")

# ---------------------------
# Plots
# ---------------------------
plt.figure(figsize=(12,5))
plt.plot(test_pilot_idxs, Y_test, label="True Phase", linewidth=2)
plt.plot(test_pilot_idxs, pred_test, label="KalmanNet-TSP Estimate", alpha=0.8)
plt.xlabel("Symbol Index (pilots)")
plt.ylabel("Phase (rad)")
plt.legend(); plt.grid(); plt.title("Phase Tracking with KalmanNet-TSP")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(test_pilot_idxs, Y_test - pred_test, label="Estimation Error")
plt.xlabel("Symbol Index (pilots)")
plt.ylabel("Error (rad)")
plt.legend(); plt.grid(); plt.title("Phase Estimation Error")
plt.show()

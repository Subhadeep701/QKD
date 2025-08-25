import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- Simulation parameters ----------------
n = 50000
m = 20
t = 0.9
h = 0.85
s = 0.05
phase_var = 1e-3
ar_coeff = 0.99
pilot_interval = 10

# ---------------- Generate Alice-Bob data ----------------
# Alice's states
q = np.random.normal(0, np.sqrt(m - 1), n)
p = np.random.normal(0, np.sqrt(m - 1), n)

# Phase noise AR(1)
phi = np.empty(n)
phi[0] = np.random.normal(0, np.sqrt(phase_var))
for i in range(1, n):
    phi[i] = ar_coeff * phi[i - 1] + np.random.normal(0, np.sqrt(phase_var))

# Channel output
gain = np.sqrt(t * h)
zq = np.random.normal(0, np.sqrt(s), n)
zp = np.random.normal(0, np.sqrt(s), n)

qB = np.zeros(n)
pB = np.zeros(n)
for i in range(n):
    q_rot = q[i] * np.cos(phi[i]) - p[i] * np.sin(phi[i])
    p_rot = q[i] * np.sin(phi[i]) + p[i] * np.cos(phi[i])
    qB[i] = gain * q_rot + zq[i]
    pB[i] = gain * p_rot + zp[i]

# Bob's random basis
b = np.random.randint(0, 2, n)
vals_q = np.where(b == 0, qB, pB)


# ---------------- KalmanNet-TSP ----------------
class KalmanNetTSP(nn.Module):
    def __init__(self, state_dim=1, measurement_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + measurement_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, state_dim)
        self.activation = nn.ReLU()

    def forward(self, measurement, prev_state):
        x = torch.cat([measurement, prev_state], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


# ---------------- Prepare training data (pilots) ----------------
pilot_idx = np.arange(0, n, pilot_interval)
x_train = []
y_train = []

for k in pilot_idx:
    # measurement = rotated quadrature at pilot
    meas = np.array([vals_q[k]])
    # prev_state = 0 placeholder (will be fed recursively during inference)
    x_train.append(np.concatenate([meas, [0.0]]))
    y_train.append(np.array([phi[k]]))  # true phase

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

# ---------------- Train KalmanNet-TSP ----------------
model = KalmanNetTSP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
num_epochs = 200

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train[:, :1], x_train[:, 1:])  # measurement + previous_state
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')

# ---------------- Evaluate on full sequence ----------------
model.eval()
phi_est_kn = np.zeros(n)
prev_state = torch.zeros(1, 1)

with torch.no_grad():
    for k in range(n):
        meas = torch.FloatTensor([[vals_q[k]]])
        phi_hat = model(meas, prev_state)
        phi_est_kn[k] = phi_hat.item()
        prev_state = phi_hat  # recursive

# ---------------- Measurement-domain correction ----------------
q_corr = np.zeros(n)
p_corr = np.zeros(n)
for i in range(n):
    phi_hat = phi_est_kn[i]
    q_corr[i] = (qB[i] * np.cos(phi_hat) + pB[i] * np.sin(phi_hat)) / gain
    p_corr[i] = (-qB[i] * np.sin(phi_hat) + pB[i] * np.cos(phi_hat)) / gain

# ---------------- MSE ----------------
mse_phase = np.mean((phi - phi_est_kn) ** 2)
mse_qp = np.mean((q - q_corr) ** 2 + (p - p_corr) ** 2)
print(f"Phase MSE: {mse_phase:.6f}")
print(f"Corrected measurement MSE: {mse_qp:.6f}")

# ---------------- Plots ----------------
plt.figure(figsize=(12, 5))
plt.plot(phi, label='True Phase', linewidth=2)
plt.plot(phi_est_kn, label='KalmanNet-TSP Estimate', alpha=0.7)
plt.xlabel('Symbol Index')
plt.ylabel('Phase (rad)')
plt.title('Pilot-aided Phase Tracking using KalmanNet-TSP')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(phi - phi_est_kn, label='Phase Estimation Error')
plt.xlabel('Symbol Index')
plt.ylabel('Error (rad)')
plt.title('Phase Estimation Error')
plt.legend()
plt.grid()
plt.show()

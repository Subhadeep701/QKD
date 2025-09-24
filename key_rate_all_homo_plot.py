import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ============================================================
# Repo modules (assumed to be in your PYTHONPATH as in original code)
# ============================================================
import Simulations.config as config
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from KNet.KalmanNet_nn import KalmanNetNN
from Pipelines.Pipeline_EKF import Pipeline_EKF

# ============================================================
# Utility
# ============================================================
def now_string():
    today = datetime.today().strftime("%m.%d.%y")
    clock = datetime.now().strftime("%H:%M:%S")
    return f"{today}_{clock}"

print("=== CV-QKD Homodyne Phase Tracking Demo ===")
print("Start Time:", now_string())

# ============================================================
# General settings
# ============================================================
args = config.general_settings()
args.N_E = 120      # training sequences
args.N_CV = 50      # validation sequences
args.N_T = 8000     # test sequences
args.T = 6        # pilot sequence length
args.T_test = 6
args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True
args.use_cuda = False
args.n_steps = 16
args.n_batch = int(args.N_E / 4)
args.lr = 1e-3
args.wd = 1e-4

device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============================================================
# CV-QKD model parameters
# ============================================================
m = 1  # state dimension (phase offset)
n = 2  # measurement dimension (I,Q for pilot estimation)
measurement_noise_variance = 0.07

# ============================================================
# State transition f(x) and measurement h(x)
# ============================================================
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

# ============================================================
# Phase variance sweep
# ============================================================
phase_var_list = np.linspace(0.1, 0.008, 10)
phase_var_list = np.round(phase_var_list, 3).tolist()
results = []

for phase_noise_variance in phase_var_list:
    print("\n=======================================")
    print(f" Running with phase_noise_variance = {phase_noise_variance}")
    print("=======================================\n")

    # ============================================================
    # System model
    # ============================================================
    Q = phase_noise_variance * torch.eye(m)
    R = measurement_noise_variance * torch.eye(n)

    sys_model = SystemModel(f=f, Q=Q, h=h, R=R,
                            T=args.T, T_test=args.T_test,
                            m=m, n=n)
    m1x_0 = torch.zeros(m, 1)
    m2x_0 = 0.01 * torch.eye(m)
    sys_model.InitSequence(m1x_0, m2x_0)

    # ============================================================
    # Data generation
    # ============================================================
    data_dir = 'Simulations/CVQKD/data'
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f'data_cvqkd_phase_var{phase_noise_variance:.2f}.pt')
    print("Generating pilot data...")
    DataGen(args, sys_model, data_file)
    print("Loading data:", data_file)

    train_input, train_target, cv_input, cv_target, \
    test_input, test_target, _, _, _ = torch.load(data_file, map_location=device)
    print("Train input shape:", train_input.shape)

    # ============================================================
    # KalmanNet training on pilot data
    # ============================================================
    KNet = KalmanNetNN()
    KNet.NNBuild(sys_model, args)

    pipe = Pipeline_EKF(now_string(), "KNet", "KNet")
    pipe.setssModel(sys_model)
    pipe.setModel(KNet)
    pipe.setTrainingParams(args)

    pipe.NNTrain(sys_model, cv_input, cv_target,
                 train_input, train_target, path_results='KNet/')
    _, _, _, KNet_out, _ = pipe.NNTest(sys_model, test_input, test_target, path_results='KNet/')

    # ============================================================
    # Define and Train RNN
    # ============================================================
    class SimpleRNN(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=32, output_dim=1, num_layers=1):
            super(SimpleRNN, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out)
            return out

    train_input_rnn = train_input.permute(0,2,1)
    train_target_rnn = train_target.permute(0,2,1)
    cv_input_rnn    = cv_input.permute(0,2,1)
    cv_target_rnn   = cv_target.permute(0,2,1)
    test_input_rnn  = test_input.permute(0,2,1)
    test_target_rnn = test_target.permute(0,2,1)

    rnn = SimpleRNN(input_dim=n, hidden_dim=32, output_dim=m, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = args.n_steps
    batch_size = args.n_batch
    train_dataset = torch.utils.data.TensorDataset(train_input_rnn, train_target_rnn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        rnn.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = rnn(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    rnn.eval()
    with torch.no_grad():
        test_input_device = test_input_rnn.to(device)
        test_target_device = test_target_rnn.to(device)
        rnn_out = rnn(test_input_device)

    # ============================================================
    # Payload generation for homodyne CV-QKD
    # (unchanged from your original code — using KNet_out, rnn_out)
    # ============================================================
    rng = np.random.default_rng(1)
    test_input_np=test_input.cpu().numpy().squeeze()
    test_target_np = test_target.cpu().numpy().squeeze()
    KNet_out_np    = KNet_out.detach().numpy().squeeze()
    rnn_out_np     = rnn_out.cpu().numpy().squeeze()

    N_test, T_pilot = test_target_np.shape
    T_payload = 6

    # === True payload phases ===
    phi_payload_true = np.zeros((N_test, T_payload))
    for i in range(N_test):
        phi_payload_true[i, 0] = test_target_np[i, -1]
        for t in range(1, T_payload):
            inc = rng.normal(0.0, phase_noise_variance)
            phi_payload_true[i, t] = phi_payload_true[i, t - 1] + inc

    # === Alice modulates coherent states ===
    Vmod   = 3.0
    A_chan = 2.0
    alice_payload = (rng.normal(scale=np.sqrt(Vmod), size=(N_test,T_payload))
                     + 1j*rng.normal(scale=np.sqrt(Vmod), size=(N_test,T_payload)))

    # === Channel: apply phase and noise ===
    meas_var = measurement_noise_variance
    meas_payload = np.zeros((N_test,2,T_payload))
    for i in range(N_test):
        s   = alice_payload[i]
        phi = phi_payload_true[i]
        r   = s * np.exp(1j*phi)
        meas_payload[i,0,:] = np.real(r) + rng.normal(0.0, np.sqrt(meas_var), T_payload)
        meas_payload[i,1,:] = np.imag(r) + rng.normal(0.0, np.sqrt(meas_var), T_payload)

    # === Extrapolated pilot estimates ===
    phi_kn_payload  = np.tile(KNet_out_np[:,-1][:,None], (1,T_payload))
    phi_rnn_payload = np.tile(rnn_out_np[:,-1][:,None], (1,T_payload))

    # ============================================================
    # Homodyne + key rate functions (same as your code)
    # ============================================================
    def homodyne_measure(phi_est):
        hom_meas = np.zeros((N_test, T_payload))
        alice_quadrature = np.zeros((N_test, T_payload))
        quadrature_choice = rng.integers(0, 2, size=(N_test, T_payload))
        for i in range(N_test):
            for t in range(T_payload):
                I = meas_payload[i, 0, t]
                Q = meas_payload[i, 1, t]
                if quadrature_choice[i, t] == 0:  # measure X
                    measured = I*np.cos(phi_est[i,t]) + Q*np.sin(phi_est[i,t])
                    alice_quadrature[i, t] = np.real(alice_payload[i, t])
                else:  # measure P
                    measured = Q*np.cos(phi_est[i,t]) - I*np.sin(phi_est[i,t])
                    alice_quadrature[i, t] = np.imag(alice_payload[i, t])
                measured += rng.normal(0.0, np.sqrt(meas_var))
                hom_meas[i, t] = measured
        return hom_meas, alice_quadrature, quadrature_choice

    def empirical_mutual_info(a_samples, b_samples):
        var_a = np.var(a_samples, ddof=1)
        var_b = np.var(b_samples, ddof=1)
        cov_ab = np.cov(a_samples, b_samples, ddof=1)[0,1]
        if var_b <= 1e-12:
            return 0.0
        var_a_given_b = max(var_a - cov_ab**2 / var_b, 1e-12)
        return max(0.5 * math.log2(var_a / var_a_given_b), 0.0)

    def compute_key_rate(hom_meas, alice_quadrature, quadrature_choice):
        mask_X = (quadrature_choice == 0)
        mask_P = (quadrature_choice == 1)
        I_X = empirical_mutual_info(alice_quadrature[mask_X], hom_meas[mask_X]) if mask_X.sum()>5 else 0.0
        I_P = empirical_mutual_info(alice_quadrature[mask_P], hom_meas[mask_P]) if mask_P.sum()>5 else 0.0
        count_X, count_P = mask_X.sum(), mask_P.sum()
        I_AB = (I_X*count_X + I_P*count_P) / max(count_X + count_P, 1)
        beta = 0.95
        T_est = A_chan**2
        var_b = np.var(hom_meas, ddof=1)
        eps_est = (var_b - T_est*Vmod - 1.0 - meas_var) / max(T_est, 1e-12)
        eps_est = max(eps_est, 0.0)
        chi_BE = 0.5 * math.log2(1.0 + (T_est*Vmod) / (1.0 + meas_var + T_est*eps_est))
        K = max(beta * I_AB - chi_BE, 0.0)
        return I_AB, chi_BE, K

    # KalmanNet key rate
    hom_kn, alice_kn, quad_kn = homodyne_measure(phi_kn_payload)
    I_AB_kn, chi_BE_kn, K_kn = compute_key_rate(hom_kn, alice_kn, quad_kn)

    # RNN key rate
    hom_rnn, alice_rnn, quad_rnn = homodyne_measure(phi_rnn_payload)
    I_AB_rnn, chi_BE_rnn, K_rnn = compute_key_rate(hom_rnn, alice_rnn, quad_rnn)

    # UKF payload estimate
    points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=1.0)
    phi_ukf_all = np.zeros((N_test, T_pilot + T_payload))
    Q_ukf = np.eye(1) * (phase_noise_variance)
    R_ukf = np.eye(2) * (measurement_noise_variance)

    def fx(x, dt): return x
    def hx(x):
        A = 2.0
        phi_signal = 0.0
        return np.array([A*np.cos(x[0]+phi_signal), A*np.sin(x[0]+phi_signal)])

    for i in range(N_test):
        ukf = UKF(dim_x=1, dim_z=2, fx=fx, hx=hx, dt=1.0, points=points)
        ukf.x = np.random.normal(0.0, np.sqrt(phase_noise_variance), size=(1,))
        ukf.P = np.eye(1) * 0.01
        ukf.Q = Q_ukf
        ukf.R = R_ukf
        for t in range(T_pilot):
            z = test_input_np[i,:,t]
            ukf.predict()
            ukf.update(z)
            phi_ukf_all[i, t] = ukf.x[0]
        phi_ukf_all[i, T_pilot:] = ukf.x[0]

    phi_ukf_payload = phi_ukf_all[:, T_pilot:]
    hom_ukf, alice_ukf, quad_ukf = homodyne_measure(phi_ukf_payload)
    I_AB_ukf, chi_BE_ukf, K_ukf = compute_key_rate(hom_ukf, alice_ukf, quad_ukf)

    # Store results
    results.append((phase_noise_variance, K_kn, K_rnn, K_ukf))

# ============================================================
# Summary results
# ============================================================
def plot_key_rate_vs_phase_var(results):
    """
    Plot Key Rate vs Phase Noise Variance for KalmanNet, RNN, and UKF.

    Args:
        results: list of tuples [(phase_var, K_kn, K_rnn, K_ukf), ...]
    """
    phase_vars = [r[0] for r in results]
    K_kn = [r[1] for r in results]
    K_rnn = [r[2] for r in results]
    K_ukf = [r[3] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(phase_vars, K_kn, marker='o', label='KalmanNet')
    plt.plot(phase_vars, K_rnn, marker='s', label='RNN')
    plt.plot(phase_vars, K_ukf, marker='^', label='UKF')

    plt.xlabel('Phase Noise Variance')
    plt.ylabel('Key Rate')
    plt.title('Key Rate vs Phase Noise Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage after computing `results`:
print("\n=== Key Rate vs Phase Noise Variance ===")
for pv, k_kn, k_rnn, k_ukf in results:
    print(f"PhaseVar={pv:.3f} -> KalmanNet={k_kn:.6f}, RNN={k_rnn:.6f}, UKF={k_ukf:.6f}")

plot_key_rate_vs_phase_var(results)
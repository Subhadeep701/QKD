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
import pandas as pd

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

def mean_ci(values, confidence=0.95):
    arr = np.array(values)
    mean = arr.mean()
    ci = stats.t.ppf(1 - (1 - confidence)/2, df=arr.shape[0]-1) * arr.std(ddof=1) / np.sqrt(arr.shape[0])
    return mean, ci
# ============================================================
# General settings
# ============================================================
args = config.general_settings()
args.N_E = 3600   # training sequences good result 3600
args.N_CV = 50       # validation sequences
args.N_T = 15000     # test sequences
args.T = 5     # pilot sequence length
args.T_test = 5
args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True
args.use_cuda = False
args.n_steps = 150
args.n_batch = int(args.N_E / 28)
args.lr = 1e-3
args.wd = 1e-4
T_payload = 1
device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============================================================
# CV-QKD model parameters
# ============================================================
m = 1  # state dimension (phase offset)
n = 2  # measurement dimension (I,Q for pilot estimation)
# measurement_noise_variance will be set in outer loop (sweep)

# ============================================================
# State transition f(x) and measurement h(x)
# ============================================================
def f_true(x, jacobian=False):
    batch_size = x.shape[0]
    # Strong nonlinear drift
    drift_sin = 0.2 * torch.sin(0.5 * torch.arange(x.shape[1], device=x.device, dtype=x.dtype))
    # Random walk
    noise_rw = 0.05 * torch.randn_like(x)
    # Occasional large jumps
    jump_prob = 0.2
    jump = torch.zeros_like(x)
    jump_mask = torch.rand(batch_size, x.shape[1], device=x.device) < jump_prob
    jump[jump_mask] = 0.3 * torch.randn_like(jump[jump_mask])
    x_next = x + drift_sin + noise_rw + jump
    if jacobian:
        # UKF still sees identity Jacobian (linear)
        F = torch.eye(x.shape[1], device=x.device).repeat(batch_size, 1, 1)
        return x_next, F
    else:
        return x_next

def h_true(x, jacobian=False):
    """
    True CV-QKD measurement model (homodyne):
    - Detector nonlinearity (saturation)
    - Phase-dependent amplitude distortion
    - Weak cross-talk between I/Q channels
    """
    A = 2.0
    alpha_sat = 2.5  # saturation level
    cross_talk = 0.08  # weak I<->Q coupling

    # Ideal measurement
    I_ideal = A * torch.cos(x)
    Q_ideal = A * torch.sin(x)

    # Detector saturation (tanh model)
    I_sat = alpha_sat * torch.tanh(I_ideal / alpha_sat)
    Q_sat = alpha_sat * torch.tanh(Q_ideal / alpha_sat)

    # Phase-dependent amplitude distortion (nonlinear)
    I_dist = I_sat + 0.1 * torch.sin(2 * x)
    Q_dist = Q_sat + 0.1 * torch.sin(2 * x)

    # Weak I-Q cross-talk
    I_meas = I_dist + cross_talk * Q_dist
    Q_meas = Q_dist + cross_talk * I_dist

    if jacobian:
        y = torch.cat([I_meas, Q_meas], dim=1)

        # Approximate Jacobians (derivatives)
        dI_dtheta = (-A * torch.sin(x) * (1 - (I_ideal/alpha_sat)**2)   # derivative through tanh
                     + 0.2 * torch.cos(2*x) + cross_talk * (1 - (Q_ideal/alpha_sat)**2))
        dQ_dtheta = ( A * torch.cos(x) * (1 - (Q_ideal/alpha_sat)**2)
                     + 0.2 * torch.cos(2*x) + cross_talk * (1 - (I_ideal/alpha_sat)**2))

        H = torch.cat([dI_dtheta.view(x.shape[0],1,m),
                       dQ_dtheta.view(x.shape[0],1,m)], dim=1)
        return y, H
    else:
        return torch.cat([I_meas, Q_meas], dim=1)


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
phase_var_list = np.linspace(0.1, 0.01, 5) #6
phase_var_list = np.round(phase_var_list, 3).tolist()


# ============================================================
# Measurement variance sweep (user-specified defaults)
# ============================================================

meas_var_list = np.linspace(0.08, 0.01, 4)#4
meas_var_list = np.round(meas_var_list, 3).tolist()
# Results accumulator and CSV setup
results = []
csv_filename = "cvqkd_keyrate+phasemse_sweep_new.csv"
# If file exists remove it to start fresh (optional; comment out if you prefer append)
if os.path.exists(csv_filename):
    os.remove(csv_filename)

# ============================================================
# Main nested sweep: measurement noise outer, phase noise inner
# ============================================================
for measurement_noise_variance in meas_var_list:
    print("\n#################################################")
    print(f"Measurement noise variance sweep: {measurement_noise_variance}")
    print("#################################################\n")

    # keep the original inner loop over phase noise variance
    for phase_noise_variance in phase_var_list:
        print("\n=======================================")
        print(f" Running with phase_noise_variance = {phase_noise_variance}")
        print("=======================================\n")

        # ============================================================
        # System model
        # ============================================================
        Q = phase_noise_variance * torch.eye(m)
        R = measurement_noise_variance * torch.eye(n)
        Q2 = phase_noise_variance * torch.eye(m)
        R2 = measurement_noise_variance * torch.eye(n)

        sys_model = SystemModel(f=f, Q=Q, h=h, R=R,
                                T=args.T, T_test=args.T_test,
                                m=m, n=n)
        m1x_0 = torch.zeros(m, 1)
        m2x_0 = 0.000001 * torch.eye(m)
        sys_model.InitSequence(m1x_0, m2x_0)

        # ============================================================
        # Data generation
        # ============================================================
        data_dir = 'Simulations/CVQKD/data'
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, f'data_cvqkd_phase_var{phase_noise_variance:.2f}_meas{measurement_noise_variance:.2f}.pt')
        print("Generating pilot data...")
        DataGen(args, sys_model, data_file)
        print("Loading data:", data_file)

        train_input, train_target, cv_input, cv_target, \
        test_input, test_target, _, _, _ = torch.load(data_file, map_location=device)
        print("Train input shape:", train_input.shape)

        # ============================================================
        # KalmanNet training on pilot data
        # ============================================================
        sys_model2 = SystemModel(f=f, Q=Q, h=h, R=R,
                                 T=args.T, T_test=args.T_test,
                                 m=m, n=n)
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
            def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_layers=1):
                super(SimpleRNN, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out)
                return out

        # Permute to (batch, seq_len, features)
        train_input_rnn = train_input.permute(0, 2, 1)
        train_target_rnn = train_target.permute(0, 2, 1)
        cv_input_rnn = cv_input.permute(0, 2, 1)
        cv_target_rnn = cv_target.permute(0, 2, 1)
        test_input_rnn = test_input.permute(0, 2, 1)
        test_target_rnn = test_target.permute(0, 2, 1)

        # ---- SHIFT TARGETS TO ENFORCE CAUSALITY ----
        train_input_rnn_shift = train_input_rnn[:, :-1, :]  # y0..y_{T-2}
        train_target_rnn_shift = train_target_rnn[:, 1:, :]  # phi1..phi_{T-1}
        cv_input_rnn_shift = cv_input_rnn[:, :-1, :]
        cv_target_rnn_shift = cv_target_rnn[:, 1:, :]
        test_input_rnn_shift = test_input_rnn[:, :-1, :]
        test_target_rnn_shift = test_target_rnn[:, 1:, :]


        # Instantiate model
        rnn = SimpleRNN(input_dim=n, hidden_dim=30, output_dim=m, num_layers=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3, weight_decay=1e-4)

        # DataLoader with shifted sequences
        train_dataset = torch.utils.data.TensorDataset(train_input_rnn_shift, train_target_rnn_shift)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_batch, shuffle=True)

        # ---- Training ----
        for epoch in range(args.n_steps):
            rnn.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = rnn(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

        # ---- Evaluation ----
        rnn.eval()
        with torch.no_grad():
            rnn_out = rnn(test_input_rnn_shift.to(device)).cpu()
            # ============================================================
            # ---- Compute MSE on pilot ----
            # ============================================================
            mse_knet = torch.mean((KNet_out - test_target) ** 2).item()
            mse_rnn = torch.mean((rnn_out - test_target_rnn_shift) ** 2).item()

            # ==========================
            # UKF on pilot and payload
            # ==========================
            UKF_est = np.zeros_like(test_target.cpu().numpy())
            phi_ukf_payload = np.zeros((test_target.shape[0], T_payload))

            for i in range(UKF_est.shape[0]):
                # Initialize UKF once per sample
                ukf = UKF(dim_x=1, dim_z=2,
                          fx=lambda x, dt: x,
                          hx=lambda x: np.array([2.0 * np.cos(x[0]), 2.0 * np.sin(x[0])]),
                          dt=1.0,
                          points=MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=1.0))
                ukf.x = np.array([0.0])
                ukf.P = np.eye(1) * 0.01
                ukf.Q = np.eye(1) * (phase_noise_variance)
                ukf.R = np.eye(2) * (measurement_noise_variance)

                # --- Pilot phase estimation with updates ---
                for t in range(test_target.shape[2]):  # T_pilot steps
                    z = test_input[i, :, t].numpy()
                    ukf.predict()
                    ukf.update(z)
                    UKF_est[i, :, t] = ukf.x[0]

                # --- Payload phase estimation: continue predictions only ---
                phi_ukf_payload[i, 0] = ukf.x[0]  # starting from last pilot estimate
                for t in range(1, T_payload):
                    #ukf.predict()  # no measurement update
                    phi_ukf_payload[i, t] = ukf.x[0]

            # ==========================
            # Compute MSEs for pilot
            # ==========================
            # === Extract scalar last component consistently for all methods ===
            phi_true_scalar = test_target.cpu().numpy()[:, -1, -1]  # True φ_last (scalar)
            phi_knet_scalar = KNet_out.detach().cpu().numpy()[:, -1, -1]  # KNet estimated φ_last
            phi_rnn_scalar = rnn_out.cpu().numpy()[:, -1, 0]  # RNN output already scalar
            phi_ukf_scalar = UKF_est[:, -1, -1]  # UKF estimated φ_last

            # === Compute scalar MSEs ===
            mse_knet_last = np.mean((phi_knet_scalar - phi_true_scalar) ** 2)
            mse_rnn_last = np.mean((phi_rnn_scalar - phi_true_scalar) ** 2)
            mse_ukf_last = np.mean((phi_ukf_scalar - phi_true_scalar) ** 2)

            # # === DEBUG PRINT ===
            # seq = 0  # Inspect first sequence
            # print("\n=== DEBUG: Single Sequence Scalar Comparison at Last Time-Step ===")
            # print(f"Sequence {seq}:")
            # print(f"  True φ_last (scalar):        {phi_true_scalar[seq]:+.6f}")
            # print(f"  KNet φ_last estimate:        {phi_knet_scalar[seq]:+.6f}")
            # print(f"  RNN φ_last estimate:         {phi_rnn_scalar[seq]:+.6f}")
            # print(f"  UKF φ_last estimate:         {phi_ukf_scalar[seq]:+.6f}")
            # print(f"Scalar MSEs: KNet={mse_knet_last:.6e}, RNN={mse_rnn_last:.6e}, UKF={mse_ukf_last:.6e}")

            # === Per-sequence scalar MSE distribution for variance & CI ===
            mse_knet_samples_last = (phi_knet_scalar - phi_true_scalar) ** 2
            mse_rnn_samples_last = (phi_rnn_scalar - phi_true_scalar) ** 2
            mse_ukf_samples_last = (phi_ukf_scalar - phi_true_scalar) ** 2

            # Mean & CI
            mse_knet_mean, mse_knet_ci = mean_ci(mse_knet_samples_last)
            mse_rnn_mean, mse_rnn_ci = mean_ci(mse_rnn_samples_last)
            mse_ukf_mean, mse_ukf_ci = mean_ci(mse_ukf_samples_last)

            # Variance
            mse_knet_var = np.var(mse_knet_samples_last, ddof=1)
            mse_rnn_var = np.var(mse_rnn_samples_last, ddof=1)
            mse_ukf_var = np.var(mse_ukf_samples_last, ddof=1)

            # === Print ===
            print(f"MSE ± CI: KalmanNet={mse_knet_mean:.6f} ± {mse_knet_ci:.6f}, Var={mse_knet_var:.6f}")
            print(f"           RNN={mse_rnn_mean:.6f} ± {mse_rnn_ci:.6f}, Var={mse_rnn_var:.6f}")
            print(f"           UKF={mse_ukf_mean:.6f} ± {mse_ukf_ci:.6f}, Var={mse_ukf_var:.6f}")

            # ============================================================
            # Payload generation for homodyne CV-QKD (unchanged logic)
            # ============================================================
            rng = np.random.default_rng(1)
            test_input_np = test_input.cpu().numpy().squeeze()
            test_target_np = test_target.cpu().numpy().squeeze()
            KNet_out_np = KNet_out.detach().numpy().squeeze()
            rnn_out_np = rnn_out.cpu().numpy().squeeze()
            N_test, T_pilot = test_target_np.shape

            # --- True payload phases ---
            phi_payload_true = np.zeros((N_test, T_payload))
            for i in range(N_test):
                phi_payload_true[i, 0] = test_target_np[i, -1]
                for t in range(1, T_payload):
                    inc = rng.normal(0.0, np.sqrt(phase_noise_variance))
                    phi_payload_true[i, t] = phi_payload_true[i, t - 1] + inc

            # --- Alice modulates coherent states ---
            Vmod = 2.0
            A_chan = 0.9
            alice_payload = (rng.normal(scale=np.sqrt(Vmod), size=(N_test, T_payload))
                             + 1j * rng.normal(scale=np.sqrt(Vmod), size=(N_test, T_payload)))

            # --- Channel: transmitted field (no measurement noise yet) ---
            meas_payload = np.zeros((N_test, 2, T_payload))
            for i in range(N_test):
                s = alice_payload[i]
                phi = phi_payload_true[i]
                r = s * np.exp(1j * phi)  # transmitted complex field
                meas_payload[i, 0, :] = np.real(r)
                meas_payload[i, 1, :] = np.imag(r)

            # --- Extrapolated pilot estimates ---
            phi_kn_payload = np.tile(KNet_out_np[:, -1][:, None], (1, T_payload))
            phi_rnn_payload = np.tile(rnn_out_np[:, -1][:, None], (1, T_payload))


            # ==========================
            # Homodyne measurement with phase compensation
            # ==========================
            def homodyne_measure(phi_est, meas_payload, alice_payload, meas_var, rng):
                """
                Performs homodyne measurement with phase compensation.
                Phase correction: multiply by exp(-i * phi_est)
                Noise is added AFTER measurement.
                Returns:
                    hom_meas: measured quadratures at Bob
                    alice_quadrature: corresponding Alice quadrature (X or P)
                    quadrature_choice: 0=X, 1=P
                """
                N_test, T_payload = phi_est.shape
                hom_meas = np.zeros((N_test, T_payload))
                alice_quadrature = np.zeros((N_test, T_payload))
                quadrature_choice = rng.integers(0, 2, size=(N_test, T_payload))

                for i in range(N_test):
                    for t in range(T_payload):
                        # Reconstruct received complex field
                        r_complex = meas_payload[i, 0, t] + 1j * meas_payload[i, 1, t]

                        # Phase correction
                        r_corrected = r_complex * np.exp(-1j * phi_est[i, t])

                        # Homodyne quadrature measurement
                        if quadrature_choice[i, t] == 0:  # X quadrature
                            measured = np.real(r_corrected)
                            alice_quadrature[i, t] = np.real(alice_payload[i, t])
                        else:  # P quadrature
                            measured = np.imag(r_corrected)
                            alice_quadrature[i, t] = np.imag(alice_payload[i, t])

                        # Add measurement noise
                        measured += rng.normal(0.0, np.sqrt(meas_var))
                        hom_meas[i, t] = measured

                return hom_meas, alice_quadrature, quadrature_choice


            # ==========================
            # Empirical Gaussian mutual information
            # ==========================
            def empirical_mutual_info(a_samples, b_samples):
                var_a = np.var(a_samples, ddof=1)
                var_b = np.var(b_samples, ddof=1)
                cov_ab = np.cov(a_samples, b_samples, ddof=1)[0, 1]

                if var_b <= 1e-12:
                    return 0.0

                var_a_given_b = max(var_a - cov_ab ** 2 / var_b, 1e-12)
                return max(0.5 * math.log2(var_a / var_a_given_b), 0.0)


            # ==========================
            # Key rate calculation using exact excess noise
            # ==========================
            def compute_key_rate_exact(hom_meas, alice_quadrature, quadrature_choice,
                                       Vmod, meas_var, A_chan, beta=0.95, V_shot=1.0):
                mask_X = quadrature_choice == 0
                mask_P = quadrature_choice == 1
                count_X, count_P = mask_X.sum(), mask_P.sum()

                I_X = empirical_mutual_info(alice_quadrature[mask_X], hom_meas[mask_X]) if count_X > 5 else 0.0
                I_P = empirical_mutual_info(alice_quadrature[mask_P], hom_meas[mask_P]) if count_P > 5 else 0.0
                I_AB = (I_X * count_X + I_P * count_P) / max(count_X + count_P, 1)

                # Exact residual noise
                residual = hom_meas - alice_quadrature
                eps_exact = np.var(residual, ddof=1)

                # Eve's information
                T_est = A_chan ** 2
                chi_BE = 0.5 * math.log2(1.0 + (T_est * Vmod) / (V_shot + meas_var + eps_exact))

                # Key rate
                K = max(beta * I_AB - chi_BE, 0.0)
                return I_AB, eps_exact, chi_BE, K


            # ==========================
            # Compute homodyne measurements
            # ==========================
            hom_kn, alice_kn, quad_kn = homodyne_measure(phi_kn_payload, meas_payload, alice_payload,
                                                         measurement_noise_variance, rng)
            hom_rnn, alice_rnn, quad_rnn = homodyne_measure(phi_rnn_payload, meas_payload, alice_payload,
                                                            measurement_noise_variance, rng)
            hom_ukf, alice_ukf, quad_ukf = homodyne_measure(phi_ukf_payload, meas_payload, alice_payload,
                                                            measurement_noise_variance, rng)

            # ==========================
            # Compute key rates
            # ==========================
            I_AB_kn, eps_kn, chi_BE_kn, K_kn = compute_key_rate_exact(hom_kn, alice_kn, quad_kn, Vmod,
                                                                      measurement_noise_variance, A_chan)
            I_AB_rnn, eps_rnn, chi_BE_rnn, K_rnn = compute_key_rate_exact(hom_rnn, alice_rnn, quad_rnn, Vmod,
                                                                          measurement_noise_variance,
                                                                          A_chan)
            I_AB_ukf, eps_ukf, chi_BE_ukf, K_ukf = compute_key_rate_exact(hom_ukf, alice_ukf, quad_ukf, Vmod,
                                                                          measurement_noise_variance,
                                                                          A_chan)

            print(f"KalmanNet: Excess noise = {eps_kn:.6f}, Key rate = {K_kn:.6f}")
            print(f"RNN:       Excess noise = {eps_rnn:.6f}, Key rate = {K_rnn:.6f}")
            print(f"UKF:       Excess noise = {eps_ukf:.6f}, Key rate = {K_ukf:.6f}")

            # ==========================
            # Print one sequence with true & estimated phases + corrected quadratures
            # ==========================
            print(
                "Seq | Time | True phase | KalmanNet est | RNN est | UKF est | Alice X/P | KalmanNet corr | RNN corr | UKF corr"
            )
            for seq_idx in range(20):  # first 10 sequences
                for t in range(T_payload):
                    # True and estimated phases
                    true_phi = phi_payload_true[seq_idx, t]
                    kn_phi = phi_kn_payload[seq_idx, t]
                    rnn_phi = phi_rnn_payload[seq_idx, t]
                    ukf_phi = phi_ukf_payload[seq_idx, t]

                    # Quadrature choice (0=X, 1=P) from KalmanNet
                    quad = quad_kn[seq_idx, t]

                    # Alice transmitted quadrature for this choice
                    alice_q = np.real(alice_payload[seq_idx, t]) if quad == 0 else np.imag(alice_payload[seq_idx, t])

                    # Measured I and Q from the channel
                    meas_I = meas_payload[seq_idx, 0, t]
                    meas_Q = meas_payload[seq_idx, 1, t]

                    # Phase-corrected homodyne measurements
                    kn_corr = meas_I * np.cos(kn_phi) + meas_Q * np.sin(kn_phi) if quad == 0 else meas_Q * np.cos(
                        kn_phi) - meas_I * np.sin(kn_phi)
                    rnn_corr = meas_I * np.cos(rnn_phi) + meas_Q * np.sin(rnn_phi) if quad == 0 else meas_Q * np.cos(
                        rnn_phi) - meas_I * np.sin(rnn_phi)
                    ukf_corr = meas_I * np.cos(ukf_phi) + meas_Q * np.sin(ukf_phi) if quad == 0 else meas_Q * np.cos(
                        ukf_phi) - meas_I * np.sin(ukf_phi)

                    # Print all values in a formatted row
                    print(
                        f"{seq_idx:3d} | {t:2d}   | {true_phi:+.4f}     | {kn_phi:+.4f}       | {rnn_phi:+.4f}  | {ukf_phi:+.4f}  | "
                        f"{alice_q:+.4f}   | {kn_corr:+.4f}       | {rnn_corr:+.4f}  | {ukf_corr:+.4f}")

            seq_idx = 0
            print(
                "Time | True phase | KalmanNet est | RNN est | UKF est | Alice X/P | KalmanNet corr | RNN corr | UKF corr")
            for t in range(T_payload):
                true_phi = phi_payload_true[seq_idx, t]
                kn_phi = phi_kn_payload[seq_idx, t]
                rnn_phi = phi_rnn_payload[seq_idx, t]
                ukf_phi = phi_ukf_payload[seq_idx, t]

                quad = quad_kn[seq_idx, t]
                alice_q = np.real(alice_payload[seq_idx, t]) if quad == 0 else np.imag(alice_payload[seq_idx, t])

                meas_I = meas_payload[seq_idx, 0, t]
                meas_Q = meas_payload[seq_idx, 1, t]

                kn_corr = meas_I * np.cos(kn_phi) + meas_Q * np.sin(kn_phi) if quad == 0 else meas_Q * np.cos(
                    kn_phi) - meas_I * np.sin(kn_phi)
                rnn_corr = meas_I * np.cos(rnn_phi) + meas_Q * np.sin(rnn_phi) if quad == 0 else meas_Q * np.cos(
                    rnn_phi) - meas_I * np.sin(rnn_phi)
                ukf_corr = meas_I * np.cos(ukf_phi) + meas_Q * np.sin(ukf_phi) if quad == 0 else meas_Q * np.cos(
                    ukf_phi) - meas_I * np.sin(ukf_phi)

                print(f"{t:2d}   | {true_phi:+.4f}     | {kn_phi:+.4f}       | {rnn_phi:+.4f}  | {ukf_phi:+.4f}  | "
                      f"{alice_q:+.4f}   | {kn_corr:+.4f}       | {rnn_corr:+.4f}  | {ukf_corr:+.4f}")

        # Append to CSV (one row per combination)
        # Store all relevant values in a dictionary
        row = {
            "PhaseVar": phase_noise_variance,
            "MeasVar": measurement_noise_variance,
            "K_KNet": K_kn,
            "K_RNN": K_rnn,
            "K_UKF": K_ukf,
            "ExcessNoise_KNet": eps_kn,
            "ExcessNoise_RNN": eps_rnn,
            "ExcessNoise_UKF": eps_ukf,
            "MSE_KalmanNet": mse_knet_mean,
            "CI_KalmanNet": mse_knet_ci,
            "MSE_RNN": mse_rnn_mean,
            "CI_RNN": mse_rnn_ci,
            "MSE_UKF": mse_ukf_mean,
            "CI_UKF": mse_ukf_ci,
            "MSE_KalmanNet_Last": mse_knet_last,
            "MSE_RNN_Last": mse_rnn_last,
            "MSE_UKF_Last": mse_ukf_last
        }

        # Append to in-memory results
        results.append(row)

        # Optional: incremental save per iteration
        df_temp = pd.DataFrame([row])
        df_temp.to_csv(csv_filename, mode="a", index=False, header=not os.path.exists(csv_filename))


# ============================================================
# Summary results and plotting (unchanged plotting function)
# ============================================================
def plot_key_rate_vs_phase_var(results):
    """
    Plot Key Rate vs Phase Noise Variance for KalmanNet, RNN, and UKF.

    Args:
        results: list of tuples [(phase_var, meas_var, K_kn, K_rnn, K_ukf), ...]
    """
    # If results contain multiple measurement noise values, plot only the last sweep (keeps original behavior)
    # Extract entries with same meas var as last entry
    if len(results) == 0:
        print("No results to plot.")
        return

    last_meas_var = results[-1][1]
    filtered = [r for r in results if r[1] == last_meas_var]
    phase_vars = [r[0] for r in filtered]
    K_kn = [r[2] for r in filtered]
    K_rnn = [r[3] for r in filtered]
    K_ukf = [r[4] for r in filtered]

    plt.figure(figsize=(8, 5))
    plt.plot(phase_vars, K_kn, marker='o', label='KalmanNet')
    plt.plot(phase_vars, K_rnn, marker='s', label='RNN')
    plt.plot(phase_vars, K_ukf, marker='^', label='UKF')

    plt.xlabel('Phase Noise Variance')
    plt.ylabel('Key Rate')
    plt.title(f'Key Rate vs Phase Noise Variance (MeasVar={last_meas_var})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage after computing `results`:
# === Summary print ===
print("\n=== Key Rate and Phase MSE Summary ===")
for row in results:
    print(f"PhaseVar={row['PhaseVar']:.3f}, MeasVar={row['MeasVar']:.3f} -> "
          f"K_KNet={row['K_KNet']:.6f}, K_RNN={row['K_RNN']:.6f}, K_UKF={row['K_UKF']:.6f} | "
          f"MSE_KNet={row['MSE_KalmanNet']:.6f} ± {row['CI_KalmanNet']:.6f}, "
          f"MSE_RNN={row['MSE_RNN']:.6f} ± {row['CI_RNN']:.6f}, "
          f"MSE_UKF={row['MSE_UKF']:.6f} ± {row['CI_UKF']:.6f}")

# === Save full results as CSV ===
df_all = pd.DataFrame(results)
df_all.to_csv(csv_filename, index=False)  # overwrite with full table

# === Plot key rate vs phase noise variance for the last measurement variance ===
def plot_key_rate_vs_phase_var(results):
    if len(results) == 0:
        print("No results to plot.")
        return

    last_meas_var = results[-1]['MeasVar']
    filtered = [r for r in results if r['MeasVar'] == last_meas_var]
    phase_vars = [r['PhaseVar'] for r in filtered]
    K_kn = [r['K_KNet'] for r in filtered]
    K_rnn = [r['K_RNN'] for r in filtered]
    K_ukf = [r['K_UKF'] for r in filtered]

    plt.figure(figsize=(8, 5))
    plt.plot(phase_vars, K_kn, marker='o', label='KalmanNet')
    plt.plot(phase_vars, K_rnn, marker='s', label='RNN')
    plt.plot(phase_vars, K_ukf, marker='^', label='UKF')

    plt.xlabel('Phase Noise Variance')
    plt.ylabel('Key Rate')
    plt.title(f'Key Rate vs Phase Noise Variance (MeasVar={last_meas_var})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_key_rate_vs_phase_var(results)
print(f"\nSaved key rates and MSEs to {csv_filename}")

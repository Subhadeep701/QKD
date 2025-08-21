import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints

# ------------------ Helper functions (GG02 style) -----------------------------

def prepare_states(n, mu):
    """Alice Gaussian-modulates coherent states: q,p ~ N(0, mu-1)."""
    sigma2 = max(mu - 1.0, 0.0)
    q = np.random.normal(0, np.sqrt(sigma2), n).astype(np.float64)
    p = np.random.normal(0, np.sqrt(sigma2), n).astype(np.float64)
    return q, p

def transmit_states(n, q, p, T, eta, s):
    """Thermal-loss (linear gain) + additive Gaussian noise ~ N(0, s)."""
    g = np.sqrt(T * eta)
    zq = np.random.normal(0, np.sqrt(s), n)
    zp = np.random.normal(0, np.sqrt(s), n)
    qB = g * q + zq
    pB = g * p + zp
    return qB, pB

def measure_states(n, q, p):
    """Bob randomly measures X (0) or P (1). Returns (basis, measured_values)."""
    basis = np.random.randint(0, 2, size=n).astype(np.uint8)
    vals = np.where(basis == 0, q, p).astype(np.float64)
    return basis, vals

def key_sifting(n, q, p, basis):
    """Alice keeps the quadrature that Bob measured (same basis)."""
    x = np.where(basis == 0, q, p).astype(np.float64)
    return x

def phase_random_walk(n, std):
    phi = np.zeros(n, dtype=np.float64)
    for k in range(1, n):
        phi[k] = phi[k-1] + np.random.normal(0.0, std)
    return phi

def rotate(q, p, phi):
    c = np.cos(phi); s = np.sin(phi)
    q_r = q * c - p * s
    p_r = q * s + p * c
    return q_r, p_r

def build_pilots(n, pilot_period=6, A_pilot=2.0, use_two_axis=True):
    is_pilot = np.zeros(n, dtype=bool)
    pilot_idx = []
    i = 0
    while i < n:
        is_pilot[i] = True
        pilot_idx.append(i)
        if use_two_axis and i+1 < n:
            is_pilot[i+1] = True
            pilot_idx.append(i+1)
            i += pilot_period
        else:
            i += pilot_period
    xA = np.zeros(n)
    pA = np.zeros(n)
    if use_two_axis:
        toggle = 0
        for k in pilot_idx:
            if toggle == 0:
                xA[k], pA[k] = A_pilot, 0.0
                toggle = 1
            else:
                xA[k], pA[k] = 0.0, A_pilot
                toggle = 0
    else:
        xA[is_pilot] = A_pilot
        pA[is_pilot] = 0.0
    return is_pilot, xA, pA

def homodyne(phi, a, xAk, pAk, basis, g):
    """Measurement model given state (phi,a), Alice (xA,pA), basis and gain g."""
    if basis == 0:  # X
        return g * a * (xAk * np.cos(phi) + pAk * np.sin(phi))
    else:           # P
        return g * a * (-xAk * np.sin(phi) + pAk * np.cos(phi))

# ---------- simple metrics ---------------------------------------------------

def rmse(a,b):
    return np.sqrt(np.mean((a-b)**2))

def gaussian_mi_from_samples(x, y):
    """Estimate mutual information I(X;Y) assuming approximately linear Gaussian relation.
       Uses linear regression y = a x + e, then MI = 0.5 log2( Var(y)/Var(e) )."""
    x = x - np.mean(x); y = y - np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = np.cov(x, y, ddof=1)[0,1]
    a = cov / vx
    resid = y - a * x
    v_res = np.var(resid, ddof=1)
    v_res = max(v_res, 1e-12)
    return 0.5 * np.log2(vy / v_res)

# ------------------ Simulation & UKF config ----------------------------------

np.random.seed(0)

# simulation size
N = 4000                     # number of symbols
pilot_period = 8             # spacing of pilot pairs
use_two_axis = True
A_pilot = 1.5

# channel
T = 0.25
eta = 0.80
g_amp = np.sqrt(T * eta)

# phase + amplitude processes
Q_phi = 5e-4                 # phase random-walk variance per step (tune larger for harder task)
Q_amp = 2e-4                 # amplitude jitter variance per step

# detector noise variance (shot + electronics)
s_z = 0.6

# modulation
mu = 6.0                     # so Var(q) = Var(p) = mu-1
beta_rr = 0.95               # reconciliation efficiency (proxy)

# --------- Build data block -------------------------------------------------

# 1) Alice prepares data (and pilots later override)
qA, pA = prepare_states(N, mu)

# 2) Insert pilots
is_pilot, xA_pil, pA_pil = build_pilots(N, pilot_period, A_pilot, use_two_axis)
qA[is_pilot] = xA_pil[is_pilot]
pA[is_pilot] = pA_pil[is_pilot]

# 3) True dynamics: phase random walk & amplitude jitter (random walk around 1.0)
phi_true = phase_random_walk(N, np.sqrt(Q_phi))
a_true = np.ones(N, dtype=np.float64)
for k in range(1, N):
    a_true[k] = a_true[k-1] + np.random.normal(0.0, np.sqrt(Q_amp))

# 4) Rotate by true phase before channel
q_rot, p_rot = rotate(qA, pA, phi_true)

# 5) Channel (attenuation + additive noise)
qB, pB = transmit_states(N, q_rot, p_rot, T, eta, s_z)

# 6) Bob measures (random homodyne)
basis, y_meas = measure_states(N, qB, pB)

# 7) Alice sifts to Bob's basis
X_sift = key_sifting(N, qA, pA, basis)

# 8) Ideal (φ=0, a=1) reference for measurement (for diagnostics and MI)
y_ideal = np.array([homodyne(0.0, 1.0, qA[k], pA[k], basis[k], g_amp) for k in range(N)])

# ---------------------------------------------------------------------------
# 9) 2-D UKF (state = [phi, a]) tracking using pilot slots only
# ---------------------------------------------------------------------------

# UKF sigma-point setup for 2-D state
points = MerweScaledSigmaPoints(n=2, alpha=0.4, beta=2.0, kappa=2.0)

def fx_state(x, dt):
    # random-walk prediction (process noise applied via Q)
    return x

def hx_state(x, xAk, pAk, bas, g=g_amp):
    phi = float(x[0]); a = float(x[1])
    return np.array([homodyne(phi, a, xAk, pAk, bas, g)])

ukf = UKF(dim_x=2, dim_z=1, fx=fx_state, hx=None, dt=1.0, points=points)
ukf.x = np.array([0.0, 1.0])            # initial guess
ukf.P = np.diag([0.5, 0.2])            # initial uncertainty
ukf.Q = np.diag([Q_phi, Q_amp])        # process noise covariance
ukf.R = np.array([[s_z]])              # measurement noise variance

phi_est = np.zeros(N)
a_est = np.zeros(N)

for k in range(N):
    ukf.predict()
    if is_pilot[k]:
        # pass a lambda that binds the pilot values and basis
        ukf.update(np.array([y_meas[k]]),
                   hx=lambda x: hx_state(x, qA[k], pA[k], basis[k], g_amp))
    phi_est[k] = ukf.x[0]
    a_est[k]   = ukf.x[1]

# corrected (model-based) measurement stream using UKF estimates
y_corr_model = np.array([homodyne(phi_est[k], a_est[k], qA[k], pA[k], basis[k], g_amp) for k in range(N)])

# ------------------ Metrics & Key-rate proxy (β·I_AB) ------------------------

signal_mask = ~is_pilot  # signal slots used for MI/key (exclude pilots)

# phase & amplitude RMSE
rmse_phi = rmse(phi_true, phi_est)
rmse_amp = rmse(a_true, a_est)

# measurement correction MSE (on signal slots)
mse_before = np.mean((y_meas[signal_mask] - y_ideal[signal_mask])**2)
mse_after  = np.mean((y_corr_model[signal_mask] - y_ideal[signal_mask])**2)

# mutual informations (empirical Gaussian approximation)
I_no_corr  = gaussian_mi_from_samples(X_sift[signal_mask], y_meas[signal_mask])
I_corr     = gaussian_mi_from_samples(X_sift[signal_mask], y_corr_model[signal_mask])

R_no_corr  = beta_rr * I_no_corr
R_corr     = beta_rr * I_corr

# SNRs for diagnostics
def snr_db(signal, residual):
    return 10.0 * np.log10(np.var(signal) / (np.var(residual) + 1e-12))

snr_noisy = snr_db(y_ideal[signal_mask], y_meas[signal_mask] - y_ideal[signal_mask])
snr_corr  = snr_db(y_ideal[signal_mask], y_corr_model[signal_mask] - y_ideal[signal_mask])

# ------------------ Print results -------------------------------------------

print("===== UKF (2D: phase + amplitude) tracking results =====")
print(f"Phase RMSE       : {rmse_phi:.4f} rad")
print(f"Amplitude RMSE   : {rmse_amp:.4f}")
print(f"MSE on signal    : before={mse_before:.4f}, after={mse_after:.4f}")
print(f"SNR (dB)         : noisy={snr_noisy:.2f}, corrected={snr_corr:.2f}")
print()
print("Key-rate proxy (β·I_AB) comparison (signal slots):")
print(f"  no correction : I_AB = {I_no_corr:.4f} bits/use,  β·I_AB = {R_no_corr:.4f}")
print(f"  with correction: I_AB = {I_corr:.4f} bits/use,  β·I_AB = {R_corr:.4f}")
print(f"Δ(β·I_AB) = {R_corr - R_no_corr:+.4f} bits/use")

# ------------------ Diagnostic plots ---------------------------------------

plt.figure(figsize=(10,3.6))
plt.plot(phi_true, label='True φ', alpha=0.9)
plt.plot(phi_est,  '--', label='UKF φ̂', linewidth=1.2)
plt.scatter(np.where(is_pilot)[0], phi_est[is_pilot], s=6, label='pilot updates', color='C3')
plt.title('Phase tracking (true vs UKF estimate)')
plt.legend(); plt.xlabel('Symbol index'); plt.ylabel('Phase [rad]')
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3.6))
plt.plot(a_true, label='True amplitude a', alpha=0.9)
plt.plot(a_est,  '--', label='UKF â', linewidth=1.2)
plt.title('Amplitude jitter tracking (true vs UKF estimate)')
plt.legend(); plt.xlabel('Symbol index'); plt.ylabel('Amplitude'); plt.tight_layout(); plt.show()

# short-window measurement comparison
n_show = 200
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax1.plot(y_meas[:n_show], label='Measured (noisy)', alpha=0.7)
ax1.plot(y_ideal[:n_show], label='Ideal (φ=0,a=1)', lw=2)
ax1.set_title('Noisy vs Ideal (short window)'); ax1.legend()
ax2.plot(y_ideal[:n_show], label='Ideal', lw=2)
ax2.plot(y_corr_model[:n_show], '--', label='Corrected (model-based, φ̂,â)', lw=1.6)
ax2.set_title('Ideal vs Corrected (short window)'); ax2.legend()
plt.xlabel('Symbol index'); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3.6))
plt.plot(y_meas[signal_mask] - y_ideal[signal_mask], label='Residual before corr', alpha=0.6)
plt.plot(y_corr_model[signal_mask] - y_ideal[signal_mask], '--', label='Residual after corr', alpha=0.9)
plt.title('Residuals on signal slots'); plt.legend(); plt.tight_layout(); plt.show()

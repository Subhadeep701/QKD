import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from filterpy.kalman import ExtendedKalmanFilter as EKF

# ---------------- Base simulation parameters ----------------
n = 20000
m = 3
t = 0.9
h = 0.95
s = 0.001
ar_coeff = 0.99
pilot_interval = 10

beta = 0.95           # reconciliation efficiency
use_homodyne = True    # homodyne RR model for I_AB and chi_BE

# ------------------- Helpers -------------------
def est_stats(x, y):
    """Fit y ≈ g x + n and return g, Var(x), Var(n), SNR."""
    Vx = np.var(x, ddof=1)
    g  = np.cov(x, y, ddof=1)[0, 1] / Vx
    n  = y - g * x
    sig2_n = np.var(n, ddof=1)
    snr = (g**2 * Vx) / sig2_n
    return g, Vx, sig2_n, snr

def holevo_rr_homodyne_from_residuals(q_a, p_a, q_corr, p_corr, t, h, s, beta=0.95):
    """
    Estimates I_AB from SNR; estimates excess noise xi from output residuals (trusted detection);
    computes Holevo χ_BE (reverse recon, homodyne) and asymptotic key rate K = β I_AB - χ_BE.
    """
    # Fit each quadrature
    gq, Vq, sig2_n_q, SNR_q = est_stats(q_a, q_corr)
    gp, Vp, sig2_n_p, SNR_p = est_stats(p_a, p_corr)

    V_A   = 0.5*(Vq + Vp)          # Alice modulation var per quadrature
    SNR_B = 0.5*(SNR_q + SNR_p)    # Bob SNR (averaged)
    g_eff = 0.5*(gq + gp)

    # Mutual information (homodyne)
    I_AB = 0.5*np.log2(1 + SNR_B) if use_homodyne else np.log2(1 + SNR_B)

    # Trusted detection model parameters
    T = t
    eta = h
    v_el_out = s

    # Residual output noise (averaged over quadratures) on Bob's output scale
    sig2_n_out = 0.5*(sig2_n_q + sig2_n_p)

    # Detection noise at Bob output (vacuum due to inefficiency + electronics), SNU
    chi_det_out = (1 - eta) + v_el_out

    # Excess noise referred to channel input (SNU)
    denom = (eta * T)
    xi = (sig2_n_out - denom*1.0 - chi_det_out) / (denom if denom > 0 else 1.0)
    xi = max(xi, 0.0)

    # Holevo χ_BE for homodyne RR (standard Gaussian formula)
    def G(x):
        return 0.0 if x <= 0 else (x+1)*np.log2(x+1) - x*np.log2(x)

    V = V_A + 1.0                     # total var at Alice (modulation + shot noise)
    chi_line = (1 - T)/T + xi

    # Bob's measured quadrature variance at output (SNU)
    b = T*(V + chi_line) + (v_el_out + 1 - eta)/eta
    a = V
    c = np.sqrt(T*(V**2 - 1))

    # Symplectic eigenvalues of pre-measurement AB state
    def symplectic_eigs(a, b, c):
        Delta = a**2 + b**2 - 2*(c**2)
        detAB = (a*b - c**2)**2
        disc = max(Delta**2 - 4*detAB, 0.0)
        l1 = np.sqrt(max((Delta + np.sqrt(disc))/2, 1.0))
        l2 = np.sqrt(max((Delta - np.sqrt(disc))/2, 1.0))
        return l1, l2

    lam1, lam2 = symplectic_eigs(a, b, c)

    # Conditional symplectic eigenvalue after Bob homodynes
    V_Bx = b
    V_cond = a - (c**2)/V_Bx
    lam3 = np.sqrt(max(V_cond, 1.0))

    chi_BE = G((lam1 - 1)/2) + G((lam2 - 1)/2) - G((lam3 - 1)/2)

    K_asym = max(beta * I_AB - chi_BE, 0.0)

    return {
        "I_AB": I_AB,
        "chi_BE": chi_BE,
        "K_asym": K_asym,
        "xi": xi,
        "SNR_B": SNR_B,
        "V_A": V_A,
        "g_eff": g_eff
    }

# ------------------- Phase variance sweep -------------------
phase_vars = np.logspace(np.log10(5e-2), np.log10(1e-3), 20)

# storage
pv_list = []
ukf_K_list, ekf_K_list = [], []
ukf_SNR_list, ekf_SNR_list = [], []
ukf_IAB_list, ekf_IAB_list = [], []   # store mutual info
ukf_lines, ekf_lines = [], []

for idx, phase_var in enumerate(phase_vars):
    np.random.seed(12345)  # fix RNG

    # Alice's states
    q = np.random.normal(0, np.sqrt(m - 1), n)
    p = np.random.normal(0, np.sqrt(m - 1), n)

    # Phase noise AR(1)
    phi = np.empty(n)
    phi[0] = np.random.normal(0, np.sqrt(phase_var))
    for i in range(1, n):
        phi[i] = ar_coeff * phi[i - 1] + np.random.normal(0, np.sqrt(phase_var))

    # Channel output (Bob)
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

    # --- UKF ---
    def fx(phi_state, dt):
        return ar_coeff * phi_state

    points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=0.0)
    ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=lambda x: x, dt=1.0, points=points)
    ukf.x = np.array([0.0])
    ukf.P = np.array([[1e-2]])
    ukf.Q = np.array([[phase_var]])
    ukf.R = np.array([[s]])

    phi_est_ukf = np.zeros(n)
    for k in range(n):
        ukf.predict()
        if k % pilot_interval == 0:
            if b[k] == 0:
                def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
                    phi_val = float(phi_state[0])
                    return np.array([g * (qk * np.cos(phi_val) - pk * np.sin(phi_val))])
            else:
                def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
                    phi_val = float(phi_state[0])
                    return np.array([g * (qk * np.sin(phi_val) + pk * np.cos(phi_val))])
            ukf.update(np.array([vals_q[k]]), hx=hx_local)
        phi_est_ukf[k] = float(ukf.x[0])

    # --- EKF ---
    ekf = EKF(dim_x=1, dim_z=1)
    ekf.x = np.array([0.0])
    ekf.P = np.array([[1e-2]])
    ekf.Q = np.array([[phase_var]])
    ekf.R = np.array([[s]])

    phi_est_ekf = np.zeros(n)

    def f_fx(x, dt):
        return np.array([ar_coeff * x[0]])

    def F_jacobian(x, dt):
        return np.array([[ar_coeff]])

    for k in range(n):
        ekf.F = F_jacobian(ekf.x, dt=1.0)
        ekf.x = f_fx(ekf.x, dt=1.0)
        ekf.predict()
        if k % pilot_interval == 0:
            if b[k] == 0:
                def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
                    return np.array([g * (qk * np.cos(phi_state[0]) - pk * np.sin(phi_state[0]))])
                def H_jac(phi_state, qk=q[k], pk=p[k], g=gain):
                    return np.array([[-g * (qk * np.sin(phi_state[0]) + pk * np.cos(phi_state[0]))]])
            else:
                def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
                    return np.array([g * (qk * np.sin(phi_state[0]) + pk * np.cos(phi_state[0]))])
                def H_jac(phi_state, qk=q[k], pk=p[k], g=gain):
                    return np.array([[g * (qk * np.cos(phi_state[0]) - pk * np.sin(phi_state[0]))]])
            ekf.update(np.array([vals_q[k]]), HJacobian=H_jac, Hx=hx_local)
        phi_est_ekf[k] = float(ekf.x[0])

    # Measurement correction
    def correct_measurements(phi_est_arr):
        q_corr = np.zeros(n)
        p_corr = np.zeros(n)
        for i in range(n):
            phi_hat = phi_est_arr[i]
            q_corr[i] = (qB[i] * np.cos(phi_hat) + pB[i] * np.sin(phi_hat)) / gain
            p_corr[i] = (-qB[i] * np.sin(phi_hat) + pB[i] * np.cos(phi_hat)) / gain
        return q_corr, p_corr

    q_corr_ukf, p_corr_ukf = correct_measurements(phi_est_ukf)
    q_corr_ekf, p_corr_ekf = correct_measurements(phi_est_ekf)

    # Compute info-theoretic metrics
    res_ukf = holevo_rr_homodyne_from_residuals(q, p, q_corr_ukf, p_corr_ukf, t, h, s, beta)
    res_ekf = holevo_rr_homodyne_from_residuals(q, p, q_corr_ekf, p_corr_ekf, t, h, s, beta)

    # store values
    pv_list.append(phase_var)
    ukf_K_list.append(res_ukf['K_asym'])
    ekf_K_list.append(res_ekf['K_asym'])
    ukf_SNR_list.append(res_ukf['SNR_B'])
    ekf_SNR_list.append(res_ekf['SNR_B'])
    ukf_IAB_list.append(res_ukf['I_AB'])
    ekf_IAB_list.append(res_ekf['I_AB'])

    # formatted lines
    ukf_lines.append(
        f"UKF:  I_AB = {res_ukf['I_AB']:.4f} bits/use | χ_BE = {res_ukf['chi_BE']:.4f} bits/use | "
        f"K = {res_ukf['K_asym']:.4f} | SNR_B={res_ukf['SNR_B']:.2f} | xi={res_ukf['xi']:.5f} | "
        f"V_A={res_ukf['V_A']:.2f} | g_eff≈{res_ukf['g_eff']:.3f}"
    )
    ekf_lines.append(
        f"EKF:  I_AB = {res_ekf['I_AB']:.4f} bits/use | χ_BE = {res_ekf['chi_BE']:.4f} bits/use | "
        f"K = {res_ekf['K_asym']:.4f} | SNR_B={res_ekf['SNR_B']:.2f} | xi={res_ekf['xi']:.5f} | "
        f"V_A={res_ekf['V_A']:.2f} | g_eff≈{res_ekf['g_eff']:.3f}"
    )

# ------------------- Print all stored lines -------------------
print("\n=== Sweep results over phase variance ===")
for pv, uline, eline in zip(pv_list, ukf_lines, ekf_lines):
    print(f"\nphase_var = {pv:.6e}")
    print(uline)
    print(eline)

# ------------------- Plot: Asymptotic Key Rate vs Phase Variance -------------------
plt.figure(figsize=(7.5,4))
plt.semilogx(pv_list, ukf_K_list, 'o-', label='UKF')
plt.semilogx(pv_list, ekf_K_list, 's-', label='EKF')
plt.xlabel("Phase variance (rad²)")
plt.ylabel("Asymptotic key rate K (bits/use)")
plt.grid(True, which='both')
plt.title("Asymptotic Key Rate vs Phase Noise Variance (UKF vs EKF Tracking in CV-QKD)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------- Plot: Mutual Information vs Phase Variance -------------------
plt.figure(figsize=(7.5,4))
plt.semilogx(pv_list, ukf_IAB_list, 'o-', label='UKF')
plt.semilogx(pv_list, ekf_IAB_list, 's-', label='EKF')
plt.xlabel("Phase variance (rad²)")
plt.ylabel("Mutual Information I_AB (bits/use)")
plt.grid(True, which='both')
plt.title("Mutual Information vs Phase Noise Variance (UKF vs EKF Tracking in CV-QKD)")
plt.legend()
plt.tight_layout()
plt.show()

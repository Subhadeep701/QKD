import numpy as np
from numba import njit
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints

# ---------------- Simulation parameters ----------------
n = 10000              # number of coherent states
m = 20                 # modulation variance
t = 0.9                # channel transmittance
h = 0.85               # setup efficiency
s = 0.01               # Gaussian noise variance (per quadrature)
phase_var = 1e-1       # AR(1) innovation variance
ar_coeff = 0.99        # AR(1) coefficient

# ------------------- MAIN LOOP -------------------------

# 1. Alice prepares states
q = np.random.normal(0, np.sqrt(m - 1), n)
p = np.random.normal(0, np.sqrt(m - 1), n)

# 2. Channel transmission with phase noise (thermal-loss + AR(1) phase drift)
qB = np.empty(n, dtype=np.float64)
pB = np.empty(n, dtype=np.float64)
zq = np.random.normal(0, np.sqrt(s), n)
zp = np.random.normal(0, np.sqrt(s), n)

phi = np.empty(n, dtype=np.float64)
phi[0] = np.random.normal(0, np.sqrt(phase_var))
for i in range(1, n):
    phi[i] = ar_coeff * phi[i - 1] + np.random.normal(0, np.sqrt(phase_var))

gain = np.sqrt(t * h)

for i in range(n):
    q_rot = q[i] * np.cos(phi[i]) - p[i] * np.sin(phi[i])
    p_rot = q[i] * np.sin(phi[i]) + p[i] * np.cos(phi[i])
    qB[i] = gain * q_rot + zq[i]
    pB[i] = gain * p_rot + zp[i]

# 3. Bob measures states (random homodyne basis selection)
meas_q = np.empty(n, dtype=np.uint8)
vals_q = np.empty(n, dtype=np.float32)
b = np.random.randint(low=0, high=2, size=n)
for i in range(n):
    if b[i] == 0:
        meas_q[i] = 0
        vals_q[i] = qB[i]
    else:
        meas_q[i] = 1
        vals_q[i] = pB[i]

# 4. Key sifting (Alice keeps the quadratures Bob measured)
x = np.empty(n, dtype=np.float32)
for i in range(n):
    if b[i] == 0:
        x[i] = q[i]
    else:
        x[i] = p[i]

# ------------------- UKF Phase Tracking & Correction -------------------------

# State transition (AR(1))
def fx(phi_state, dt):
    # phi_{k+1} = a * phi_k + w_k
    return ar_coeff * phi_state

# We'll override hx at each step with a closure that knows q[i], p[i], gain, and basis b[i]
points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=0.0)
ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=lambda x: x, dt=1.0, points=points)

# Initial UKF settings
ukf.x = np.array([0.0])        # initial phase estimate
ukf.P = np.array([[1e-2]])     # initial covariance
ukf.Q = np.array([[phase_var]])# process noise variance (matches channel innovation)
ukf.R = np.array([[s]])        # measurement noise variance (matches added Gaussian noise)

phi_est = np.zeros(n, dtype=np.float64)

for k in range(n):
    ukf.predict()

    # Measurement model for this k (based on which quadrature Bob measured)
    if b[k] == 0:
        # q-basis: z_k = gain * (q*cos(phi) - p*sin(phi)) + noise
        def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
            phi_val = float(phi_state[0]) if np.ndim(phi_state) > 0 else float(phi_state)
            return np.array([g * (qk * np.cos(phi_val) - pk * np.sin(phi_val))])
    else:
        # p-basis: z_k = gain * (q*sin(phi) + p*cos(phi)) + noise
        def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
            phi_val = float(phi_state[0]) if np.ndim(phi_state) > 0 else float(phi_state)
            return np.array([g * (qk * np.sin(phi_val) + pk * np.cos(phi_val))])

    ukf.update(np.array([vals_q[k]]), hx=hx_local)
    phi_est[k] = float(ukf.x[0])

# De-rotate Bob's received signals using estimated phase
cos_m = np.cos(-phi_est)
sin_m = np.sin(-phi_est)
qB_corr = qB * cos_m - pB * sin_m
pB_corr = qB * sin_m + pB * cos_m

# Corrected measured (still in Bob scale). For comparison with Alice, de-gain:
y_raw = vals_q.astype(np.float64)           # raw measured (Bob scale)
y_corr = np.where(b == 0, qB_corr, pB_corr) # corrected (Bob scale)
y_raw_alice_scale  = y_raw  / gain          # scale back to Alice's units
y_corr_alice_scale = y_corr / gain          # scale back to Alice's units

# ------------------- Results -------------------------
print("Alice sifted key (first 10):        ", x[10:20])
print("Bob measured (raw, first 10):       ", y_raw[10:20])
print("Bob corrected (UKF, first 10):      ", y_corr[10:20])
print("Raw (Alice scale, first 10):        ", y_raw_alice_scale[:10])
print("Corrected (Alice scale, first 10):  ", y_corr_alice_scale[:10])

# Quick quality metrics
mse_raw  = np.mean((y_raw_alice_scale  - x)**2)
mse_corr = np.mean((y_corr_alice_scale - x)**2)
print(f"\nMSE vs Alice (raw):  {mse_raw:.6e}")
print(f"MSE vs Alice (UKF):  {mse_corr:.6e}")

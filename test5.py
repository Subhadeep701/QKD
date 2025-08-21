import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from filterpy.kalman import ExtendedKalmanFilter as EKF

# ---------------- Simulation parameters ----------------
n = 1000
m = 20
t = 0.9
h = 0.85
s = 0.05
phase_var = 5*1e-3

ar_coeff = 0.99
pilot_interval = 20

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

# ------------------- UKF -------------------
def fx(phi_state, dt):
    return ar_coeff * phi_state

points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=0.0)
ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=lambda x: x, dt=1.0, points=points)

ukf.x = np.array([0.0])
ukf.P = np.array([[1e-2]])
ukf.Q = np.array([[phase_var]])
ukf.R = np.array([[s]])

phi_est = np.zeros(n)

for k in range(n):
    ukf.predict()

    if k % pilot_interval == 0:  # <-- only update at pilots
        if b[k] == 0:
            def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
                phi_val = float(phi_state[0])
                return np.array([g * (qk * np.cos(phi_val) - pk * np.sin(phi_val))])
        else:
            def hx_local(phi_state, qk=q[k], pk=p[k], g=gain):
                phi_val = float(phi_state[0])
                return np.array([g * (qk * np.sin(phi_val) + pk * np.cos(phi_val))])
        ukf.update(np.array([vals_q[k]]), hx=hx_local)

    phi_est[k] = float(ukf.x[0])

# ------------------- EKF -------------------
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
    # ---- Predict ----
    ekf.F = F_jacobian(ekf.x, dt=1.0)   # state Jacobian
    ekf.x = f_fx(ekf.x, dt=1.0)         # propagate state
    ekf.predict()                       # propagate covariance

    # ---- Update only at pilots ----
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

        ekf.update(
            np.array([vals_q[k]]),
            HJacobian=H_jac,
            Hx=hx_local
        )

    phi_est_ekf[k] = float(ekf.x[0])

# ------------------- Iterated EKF (EIKF) -------------------
def iterated_update(ekf, z, HJacobian, Hx, R=None, N=10, tol=1e-6):
    """
    Iterated EKF update with Joseph form covariance update.

    ekf        : filterpy ExtendedKalmanFilter-like object (uses ekf.x and ekf.P)
    z          : measurement (array-like)
    HJacobian  : function H(x) -> Jacobian matrix evaluated at x
    Hx         : measurement function h(x) -> predicted measurement vector
    R          : measurement noise covariance (if None, ekf.R is used)
    N          : max iterations
    tol        : convergence tolerance for iteration
    """
    if R is None:
        R = ekf.R

    # keep the prior (predicted) covariance and state (these remain fixed during iterations)
    P_pred = ekf.P.copy()
    x_pred_state = ekf.x.copy()   # x_{k|k-1} (used in innovation linearization)

    x_iter = x_pred_state.copy()  # current iterate (initialized to prior)
    K = None
    H = None

    for i in range(N):
        H = HJacobian(x_iter)                 # linearize about current iterate
        # innovation (Gauss-Newton style)
        y = z - Hx(x_iter) + H @ (x_iter - x_pred_state)

        S = H @ P_pred @ H.T + R
        # Solve for K more stably than using inv where possible:
        # K = P_pred @ H.T @ np.linalg.inv(S)
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_new = x_pred_state + K @ y

        if np.linalg.norm(x_new - x_iter) < tol:
            x_iter = x_new
            break

        x_iter = x_new

    # final assignment of state
    ekf.x = x_iter

    # Joseph form covariance update (robust)
    I = np.eye(len(ekf.x))
    ekf.P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T


eikf = EKF(dim_x=1, dim_z=1)
eikf.x = np.array([0.0])
eikf.P = np.array([[1e-2]])
eikf.Q = np.array([[phase_var]])
eikf.R = np.array([[s]])

phi_est_eikf = np.zeros(n)

for k in range(n):
    # ---- Predict ----
    eikf.F = F_jacobian(eikf.x, dt=1.0)
    eikf.x = f_fx(eikf.x, dt=1.0)
    eikf.predict()

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

        iterated_update(
            eikf,
            np.array([vals_q[k]]),
            HJacobian=H_jac,
            Hx=hx_local,
            N=10
        )

    phi_est_eikf[k] = float(eikf.x[0])

# ------------------- Error Metrics -------------------
mse_ukf = np.mean((phi - phi_est)**2)
mae_ukf = np.mean(np.abs(phi - phi_est))

mse_ekf = np.mean((phi - phi_est_ekf)**2)
mae_ekf = np.mean(np.abs(phi - phi_est_ekf))

mse_eikf = np.mean((phi - phi_est_eikf)**2)
mae_eikf = np.mean(np.abs(phi - phi_est_eikf))

print("Error Metrics:")
print(f"UKF  -> MSE: {mse_ukf:.6f}, MAE: {mae_ukf:.6f}")
print(f"EKF  -> MSE: {mse_ekf:.6f}, MAE: {mae_ekf:.6f}")
print(f"EIKF -> MSE: {mse_eikf:.6f}, MAE: {mae_eikf:.6f}")


# ------------------- Comparison Plots -------------------
plt.figure(figsize=(12,5))
plt.plot(phi, label="True phase", linewidth=2)
plt.plot(phi_est, label="UKF estimate", alpha=0.7)
plt.plot(phi_est_ekf, label="EKF estimate", alpha=0.7)
plt.plot(phi_est_eikf, label="EIKF estimate", alpha=0.7)
plt.xlabel("Symbol index")
plt.ylabel("Phase (rad)")
plt.title("Pilot-aided Phase Tracking: UKF vs EKF vs EIKF")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(phi - phi_est, label="UKF error")
plt.plot(phi - phi_est_ekf, label="EKF error", alpha=0.7)
plt.plot(phi - phi_est_eikf, label="EIKF error", alpha=0.7)
plt.xlabel("Symbol index")
plt.ylabel("Error (rad)")
plt.title("Phase Estimation Error: UKF vs EKF vs EIKF")
plt.legend()
plt.grid()
plt.show()

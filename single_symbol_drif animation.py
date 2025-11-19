import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================================
#  CV-QKD Symbol Visualization: Noise & Phase Drift
# --------------------------------------------------------------
# This script visualizes the evolution of a *single symbol* in
# continuous-variable QKD (CV-QKD). It compares:
#   1. The original (ideal) symbol,
#   2. The same symbol with additive Gaussian noise but no drift,
#   3. The symbol under both noise and random phase drift.
#
# Additionally, histograms of the X and P quadrature marginals
# are overlaid as line plots to show how distributions spread.
#
# Output: animated .mp4 for presentation/PowerPoint use.
# ==============================================================


# ---------------- Simulation Parameters ----------------
symbol_point = (1.5, 1)  # Alice's original symbol (q, p)
radius_orig = 0.1  # "vacuum noise" / ideal uncertainty
radius_noise = 0.15  # noise-affected uncertainty (no drift)
radius_drift = 0.15  # noise + drift uncertainty radius

N = 40  # number of animation frames
theta_std = np.deg2rad(3)  # standard deviation of phase increments (radians)
n_measurements = 2000  # Monte Carlo samples per frame (for marginals)

# ---------------- Phase Drift Process ----------------
# Phase noise is modeled as a random walk (Brownian motion in phase).
theta_increments = np.random.normal(0, theta_std, N)  # small increments
thetas = np.cumsum(theta_increments)  # cumulative phase drift

# ---------------- Base Figure ----------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='k', linewidth=1)  # X-axis
ax.axvline(0, color='k', linewidth=1)  # P-axis

# --- Vacuum circle (shot noise unit) around origin ---
vacuum_circle = plt.Circle((0, 0), radius=radius_orig, facecolor='none',
                           edgecolor='gray', linestyle='--', linewidth=2,
                           label="Vacuum (Shot noise)")
ax.add_patch(vacuum_circle)

# --- Symbol circles (Alice's transmitted symbol representations) ---
# Original symbol (ideal reference, only shot noise)
c_orig = plt.Circle(symbol_point, radius=radius_orig, facecolor='none',
                    edgecolor='red', linestyle='--', linewidth=2,
                    label="Original symbol")
ax.add_patch(c_orig)

# Symbol with noise (no drift)
c_noise = plt.Circle(symbol_point, radius=radius_noise, facecolor='none',
                     edgecolor='green', linestyle='--', linewidth=2,
                     label="Noisy (no drift)")
ax.add_patch(c_noise)

# Symbol with noise + drift (this one will move in animation)
c_drift = plt.Circle(symbol_point, radius=radius_drift, facecolor='none',
                     edgecolor='blue', linestyle='--', linewidth=2,
                     label="Noisy (with drift)")
ax.add_patch(c_drift)

# --- Figure Formatting ---
ax.set_xlim(-0.2, 2.5)
ax.set_ylim(-0.2, 2)
ax.set_aspect('equal', 'box')
ax.set_title("Single Symbol Evolution in CV-QKD")
ax.set_xlabel("X quadrature")
ax.set_ylabel("P quadrature")
ax.legend(loc="upper right")

# ---------------- Marginal Projections ----------------
# To show how noise/drift affects probability distributions,
# we project samples on the X and P quadratures and overlay
# normalized histograms as curves.

# Storage for marginal line objects (so we can delete/redraw them each frame)
marginal_lines = []


def add_projection(samples, color, linestyle="-", linewidth=2):
    """
    Adds marginal distribution curves (X, P histograms) to the plot.

    samples : np.ndarray [n_samples, 2]
        Monte Carlo samples of (q, p).
    color : str
        Color of the marginal lines.
    linestyle : str
        Line style for plotting.
    linewidth : float
        Line thickness.
    """
    x_proj, p_proj = samples[:, 0], samples[:, 1]

    # axis spans for scaling histograms
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    x_span = x_max - x_min

    lines = []

    # --- X-axis marginal (horizontal projection) ---
    counts, bins = np.histogram(x_proj, bins=60, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    scale_y = 0.05 * y_span  # scaling factor so histogram fits in plot
    lineX, = ax.plot(centers, counts * scale_y, color=color,
                     linestyle=linestyle, linewidth=linewidth, zorder=3)
    lines.append(lineX)

    # --- P-axis marginal (vertical projection) ---
    counts, bins = np.histogram(p_proj, bins=60, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    scale_x = 0.05 * x_span
    lineP, = ax.plot(counts * scale_x, centers, color=color,
                     linestyle=linestyle, linewidth=linewidth, zorder=3)
    lines.append(lineP)

    return lines


# ---------------- Animation Update ----------------
def update(frame):
    """
    Update function for animation.
    Each frame rotates the symbol by current drift, regenerates noisy samples,
    and redraws marginal distributions.
    """
    global marginal_lines
    # Remove old marginal curves before drawing new ones
    for l in marginal_lines:
        try:
            l.remove()
        except:
            pass
    marginal_lines = []

    # --- Drifted symbol coordinates ---
    x0, y0 = symbol_point
    theta = thetas[frame]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = cos_t * x0 - sin_t * y0
    y_rot = sin_t * x0 + cos_t * y0
    c_drift.center = (x_rot, y_rot)  # update circle center

    # --- Monte Carlo sampling for uncertainty clouds ---
    n_samples = n_measurements
    samples_orig = np.random.normal([x0, y0], radius_orig, size=(n_samples, 2))
    samples_noise = np.random.normal([x0, y0], radius_noise, size=(n_samples, 2))
    samples_drift = np.random.normal([x_rot, y_rot], radius_drift, size=(n_samples, 2))

    # --- Add projections (marginal histograms) ---
    marginal_lines += add_projection(samples_orig, "red", linestyle="-")
    marginal_lines += add_projection(samples_noise, "green", linestyle="--")
    marginal_lines += add_projection(samples_drift, "blue", linestyle=":")

    return [c_drift] + marginal_lines


# ---------------- Animate & Save ----------------
ani = FuncAnimation(fig, update, frames=N, interval=50, blit=True)

plt.show()

# Export animation for presentations (PowerPoint, etc.)
ani.save("cvqkd_single_symbol_orig_noise_drift_lines.mp4", writer="ffmpeg", fps=15)

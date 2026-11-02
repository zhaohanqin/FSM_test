#!/usr/bin/env python3
# FSM four-step vs shared white-reference + two 90-degree phase patterns.
# Simulates one camera pixel receiving one direct and two indirect
# projector-to-camera paths; the camera is linear and static unless stated.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
N = 128
freq = np.arange(N)
projector_columns = np.array([23, 67, 101])
path_weights = np.array([0.65, 0.24, 0.11])  # direct, indirect #1, indirect #2
ambient = 0.08
theta = 2 * np.pi * freq[:, None] * projector_columns[None, :] / N
white = ambient + path_weights.sum()
dark = ambient
background = (white + dark) / 2.0

def record_pattern(shift, gamma=1.0):
    # Projected code range is [0, 1]. Gamma != 1 models projector nonlinearity.
    projected = (1.0 + np.cos(theta + shift)) / 2.0
    return ambient + (projected**gamma) @ path_weights

def four_step(i0, i1, i2, i3):
    # This uses phase shifts 0, pi/2, pi, 3*pi/2.
    return (i0 - i2) + 1j * (i3 - i1)

def two_plus_one(i0, i1, reference):
    # Works for two 90-degree-separated images and a correct DC reference.
    return 2.0 * ((i0 - reference) + 1j * (reference - i1))

def spectrum(c):
    # FFT along the projected-frequency index (not across camera pixels).
    return np.fft.fft(c, axis=-1) / N

def plot_spectrum(curves, filename, title, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 4.1))
    for label, signal, style in curves:
        ax.plot(np.arange(N), np.abs(spectrum(signal)), linestyle=style,
                linewidth=1.8, label=label)
    ax.set_xlabel("Projector column recovered by FFT")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.set_xlim(0, N - 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)

# Case A: ideal linear transport, with two interreflection paths.
i0, i1, i2, i3 = [record_pattern(k*np.pi/2) for k in range(4)]
c_four = four_step(i0, i1, i2, i3)
c_21 = two_plus_one(i0, i1, background)
# WRONG: I1 and I3 differ by 180 degrees and cannot recover two quadratures.
c_wrong = 2*((i3-background) + 1j*(background-i1))
# WRONG unless there is no ambient: simply dividing a full-white exposure by 2.
c_white_half = two_plus_one(i0, i1, white/2)

plot_spectrum([
    ("Four-step", c_four, "-"),
    ("White + 2 quadrature patterns", c_21, "--")
], "fsm_ideal_fft.png", "Linear transport with interreflections: FFT spectra")

plot_spectrum([
    ("Four-step (reference)", c_four, "-"),
    ("Wrong pair: I1 and I3", c_wrong, "--"),
    ("Incorrect DC: white/2", c_white_half, ":")
], "fsm_wrong_fft.png", "Incorrect phase pair or white-image scaling")

# Case B: gamma nonlinearity breaks equivalence.
j0, j1, j2, j3 = [record_pattern(k*np.pi/2, gamma=2.0) for k in range(4)]
g_four = four_step(j0, j1, j2, j3)
g_21 = two_plus_one(j0, j1, background)
plot_spectrum([
    ("Four-step, gamma=2", g_four, "-"),
    ("White + 2 patterns, gamma=2", g_21, "--")
], "fsm_gamma_fft.png", "Gamma nonlinear projector: FFT artifacts")

# Case C: Gaussian camera noise and a white/dark measurement shared across frequencies.
rng = np.random.default_rng(20261029)
trials = 2000
sigma = 0.01
noise = rng.normal(0, sigma, size=(trials, 4, N))
white_noise = rng.normal(0, sigma, size=(trials, 1))
dark_noise = rng.normal(0, sigma, size=(trials, 1))
i0n, i1n, i2n, i3n = [i[None, :] + noise[:, k, :] for k, i
                     in enumerate((i0, i1, i2, i3))]
a_n = (white + white_noise + dark + dark_noise)/2
cf_n = four_step(i0n, i1n, i2n, i3n)
c21_n = two_plus_one(i0n, i1n, a_n)
sf_n, s21_n = spectrum(cf_n), spectrum(c21_n)
true_spectrum = spectrum(c_four)
rms4 = np.sqrt(np.mean(np.abs(sf_n[:, 1:] - true_spectrum[None, 1:])**2))
rms21 = np.sqrt(np.mean(np.abs(s21_n[:, 1:] - true_spectrum[None, 1:])**2))
rms21_zero = np.sqrt(np.mean(np.abs(s21_n[:, 0] - true_spectrum[0])**2))

# Numerical test assertions.
assert np.max(np.abs(c_four-c_21)) < 1e-12
assert np.max(np.abs(spectrum(c_four)-spectrum(c_21))) < 1e-12
assert np.allclose(np.abs(true_spectrum[projector_columns]), path_weights, atol=1e-12)
assert np.abs(spectrum(c_white_half)[0]) > 0.10
assert rms21 > rms4

print(f"Number of sampled frequencies: {N}")
print(f"Ground-truth path positions: {projector_columns.tolist()}")
print(f"Ground-truth path weights: {path_weights.tolist()}")
print(f"Four-step max complex error vs 2+1 (ideal): {np.max(np.abs(c_four-c_21)):.3e}")
print(f"FFT max absolute difference (ideal): {np.max(np.abs(spectrum(c_four)-spectrum(c_21))):.3e}")
print(f"FFT peaks at true columns: {np.abs(true_spectrum[projector_columns]).round(6).tolist()}")
print(f"Wrong I1/I3 complex relative error: {np.linalg.norm(c_wrong-c_four)/np.linalg.norm(c_four):.3f}")
print(f"Using white/2 without dark: false FFT peak at column 0 = {np.abs(spectrum(c_white_half)[0]):.6f}")
print(f"Gamma=2 relative complex difference: {np.linalg.norm(g_21-g_four)/np.linalg.norm(g_four):.3f}")
print(f"Gamma=2 false FFT peak at column 0 = {np.abs(spectrum(g_21)[0]):.6f}")
print(f"Noise sigma: {sigma}; Monte Carlo trials: {trials}")
print(f"FFT RMS error excluding column 0, four-step: {rms4:.6f}")
print(f"FFT RMS error excluding column 0, 2+1:       {rms21:.6f}")
print(f"RMS ratio 2+1 / four-step: {rms21/rms4:.3f}")
print(f"2+1 RMS error at column 0 (shared DC noise): {rms21_zero:.6f}")
print(f"Projection count: four-step={4*N}; shared white + two per frequency={2*N+1}; "
      f"relative reduction={(1-(2*N+1)/(4*N))*100:.2f}%")

"""
Shared waveform calculations for plot_waveforms.py and compare_waveforms.py.

Waveform files are N x 3 CSVs (x,y,z per row), as used by Simulation.py, and
already include the effect of the 180° pulse.
"""

import os

import numpy as np

GAMMA = 2.6752218744e8  # Proton gyromagnetic ratio (rad/s/T)
AXES = ["x", "y", "z"]


def analyze_waveform(file, raster_time_ms=0.02, gradient_scale=1e-3):
    """
    Load a waveform file and compute its properties. Returns a dict with:
        name              File name without extension
        data              Raw file values (N x 3, file units)
        t                 Time (ms)
        g                 Gradient (T/m)
        q                 Dephasing q(t) = gamma * integral of g dt (rad/m)
        b                 b-value at the file amplitude (s/mm^2)
        B                 b-tensor (s/mm^2)
        eigenvalues       b-tensor eigenvalues, largest first (s/mm^2)
        refocusing        |q(end)| / max |q| (0 = refocused)
        freqs             Frequencies of the spectrum (Hz)
        power             Dephasing spectrum |Q(f)|^2 per axis (not normalized)
        centroids         Centroid frequency per axis (Hz), None for axes without gradient
        centroid_combined Centroid frequency of the total power of the three axes (Hz)
    """
    data = np.loadtxt(file, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{file}: expected an N x 3 CSV (x,y,z columns), got shape {data.shape}.")

    dt = raster_time_ms * 1e-3                # s
    g = data * gradient_scale                 # T/m
    t = np.arange(len(g)) * raster_time_ms    # ms
    q = GAMMA * np.cumsum(g, axis=0) * dt     # rad/m

    # b-tensor in s/mm^2
    B = (q.T @ q) * dt * 1e-6
    b = np.trace(B)
    eigenvalues = np.sort(np.linalg.eigvalsh(B))[::-1]

    q_norm = np.linalg.norm(q, axis=1)
    refocusing = q_norm[-1] / q_norm.max()

    # Encoding spectrum, zero-padded for a finer frequency resolution
    n_fft = max(2**16, len(q))
    freqs = np.fft.rfftfreq(n_fft, dt)
    power = np.abs(np.fft.rfft(q, n=n_fft, axis=0) * dt) ** 2

    # Centroid (power-weighted mean) frequencies of the dephasing spectrum
    # |Q(f)|^2, over the full spectrum. Per axis, and combined from the total
    # power of the three axes, which does not depend on the waveform
    # orientation. NOTE: the dephasing spectrum is used here; the gradient
    # spectrum |G(f)|^2 = (2 pi f)^2 |Q(f)|^2 would give higher centroids.
    axis_power = power.sum(axis=0)
    has_power = axis_power > 1e-12 * axis_power.max()
    centroids = [freqs @ power[:, i] / axis_power[i] if has_power[i] else None for i in range(3)]
    centroid_combined = freqs @ power.sum(axis=1) / power.sum()

    return {
        "name": os.path.splitext(os.path.basename(file))[0],
        "data": data,
        "t": t,
        "g": g,
        "q": q,
        "b": b,
        "B": B,
        "eigenvalues": eigenvalues,
        "refocusing": refocusing,
        "freqs": freqs,
        "power": power,
        "centroids": centroids,
        "centroid_combined": centroid_combined,
    }

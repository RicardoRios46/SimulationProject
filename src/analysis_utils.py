"""
Shared functions for the signal analysis scripts (fit_powder_average.py,
fit_tensor.py, plot_trajectories.py).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from waveform_utils import analyze_waveform


def output_dir(signal_file):
    """Return the signal file name without extension and its output folder
    graphOutputs/<name>/ (created if needed), e.g. outputs/run/run_1.csv ->
    ("run_1", "graphOutputs/run_1")."""
    name = Path(signal_file).stem
    output = f"graphOutputs/{name}"
    os.makedirs(output, exist_ok=True)
    return name, output


def load_signals(signal_file):
    """Read a signal CSV written by Simulation.py."""
    return pd.read_csv(signal_file, comment="#")


def powder_average(df):
    """Mean signal over the rotations, indexed by (waveform_idx, bval)."""
    return df.groupby(['waveform_idx', 'bval'])['signal'].mean()


def waveform_info(df, signal_file):
    """Properties of each waveform in a signal file, keyed by waveform_idx.

    The waveform files are read from waveforms/ (as in Simulation.py), with the
    raster time and gradient scale of the run's metadata file (defaults 0.02 ms
    and 1e-3 if there is no metadata file). For each waveform:
        label      File name without extension
        encoding   "LTE" (one non-zero b-tensor eigenvalue), "STE" (three equal
                   eigenvalues) or "other", from the b-tensor shape
        frequency  Centroid frequency of the dephasing spectrum |Q(f)|^2 (Hz)
        direction  Main axis of the b-tensor (unit vector, before rotation)
    """
    meta_file = f"{os.path.splitext(signal_file)[0]}_metadata.json"
    waveform_config = {}
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            waveform_config = json.load(f).get("waveform", {})
    raster_time = waveform_config.get("raster_time_ms", 0.02)
    gradient_scale = waveform_config.get("gradient_scale", 1e-3)

    info = {}
    for wf, file in df.groupby('waveform_idx')['file'].first().items():
        w = analyze_waveform(f"waveforms/{file}", raster_time, gradient_scale)
        shape = w["eigenvalues"] / w["b"]      # largest first
        if shape[1] < 0.01:
            encoding = "LTE"
        elif shape[2] > 0.3:
            encoding = "STE"
        else:
            encoding = "other"
        info[wf] = {
            "label": w["name"],
            "encoding": encoding,
            "frequency": w["centroid_combined"],
            "direction": np.linalg.eigh(w["B"])[1][:, -1],
        }
    return info


def fit_cumulant(b, signal, order=2, fix_intercept=False, b_max=10):
    """Fit the cumulant expansion of log(signal), with b in ms/µm² (only b <= b_max):
        order 2:  log(signal) = C + B b + A b^2
        order 3:  log(signal) = C + B b + A b^2 + E b^3
    With fix_intercept, C = 0 (signal = 1 at b = 0).

    Following log S = -b D + (b^2/2) V - (b^3/6) k3 + ..., returns a dict with
    the coefficients A, B, C, E (E = 0 for order 2) and
        D = -B (µm²/ms), V = 2A (µm⁴/ms²), K = 3V/D² = 6A/D²,
        k3 = -6E (µm⁶/ms³), skewness = k3 / V^(3/2) (NaN if V <= 0 or order 2).
    """
    mask = b <= b_max
    b, y = b[mask], np.log(signal[mask])
    if fix_intercept:
        powers = np.arange(order, 0, -1)
        coeffs = np.linalg.lstsq(b[:, None] ** powers, y, rcond=None)[0]
        coeffs = np.append(coeffs, 0.0)
    else:
        coeffs = np.polyfit(b, y, order)
    if order == 2:
        coeffs = np.insert(coeffs, 0, 0.0)
    E, A, B, C = coeffs
    if abs(A) < 1e-9:
        A = 0
    return cumulant_parameters(C, B, A, E, order)


def cumulant_parameters(C, B, A, E, order):
    """Coefficients and derived parameters of a cumulant fit (see fit_cumulant)."""
    D = -B
    V = A * 2
    k3 = -6 * E
    skewness = k3 / V**1.5 if order == 3 and V > 0 else np.nan
    return {"A": A, "B": B, "C": C, "E": E, "D": D, "K": (6 * A) / (D**2), "V": V,
            "k3": k3, "skewness": skewness}


#Frequency dependence models: value = slope * g(f) + intercept
FREQUENCY_MODELS = {
    "Square Root": np.sqrt,
    "Linear": lambda f: f,
    "Squared": lambda f: f**2,
}
FREQUENCY_MODEL_COLORS = {"Squared": "#004949", "Linear": "#FF6B6B", "Square Root": "#009999"}


def fit_frequency_models(freq, values):
    """Least-squares fit of each model in FREQUENCY_MODELS.

    Returns (fits, best): fits maps the model name to (slope, intercept,
    sum of squared errors), best is the name of the model with the lowest error.
    """
    fits = {}
    for name, g in FREQUENCY_MODELS.items():
        slope, intercept = np.polyfit(g(freq), values, 1)
        error = np.sum((values - (slope * g(freq) + intercept)) ** 2)
        fits[name] = (slope, intercept, error)
    best = min(fits, key=lambda name: fits[name][2])
    return fits, best


def plot_frequency_models(ax, fits, best, x, label_suffix=""):
    """Draw the fitted frequency models over x (best fit solid, others dotted)."""
    for name in ["Squared", "Linear", "Square Root"]:
        slope, intercept, _ = fits[name]
        y = slope * FREQUENCY_MODELS[name](x) + intercept
        ax.plot(x, y, color=FREQUENCY_MODEL_COLORS[name],
                linestyle='-' if name == best else ':', linewidth=2,
                label=f'{name}{label_suffix} {"(Best)" if name == best else ""}')

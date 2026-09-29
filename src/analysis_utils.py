"""
Shared functions for the signal analysis scripts (plot_signal.py,
fit_signal.py, plot_trajectories.py).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


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


def waveform_label(df, wf):
    """Waveform file name without extension for a waveform_idx."""
    raw_path = df[df['waveform_idx'] == wf]['file'].iloc[0]
    return os.path.basename(raw_path).replace(".csv", "")


def fit_cumulant(b, signal, b_max=10):
    """Fit log(signal) = A b^2 + B b + C, with b in ms/µm² (only b <= b_max).

    Returns a dict with the coefficients A, B, C and
    D = -B (µm²/ms), K = 6A/D² and V = 2A (µm⁴/ms²).
    """
    mask = b <= b_max
    A, B, C = np.polyfit(b[mask], np.log(signal[mask]), 2)
    if abs(A) < 1e-9:
        A = 0
    D = -B
    return {"A": A, "B": B, "C": C, "D": D, "K": (6 * A) / (D**2), "V": A * 2}


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

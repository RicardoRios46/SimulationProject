"""
Signal fitting and DKI analysis
Fits the powder-averaged signal (mean over rotations) of each waveform for
the diffusivity D, kurtosis K and variance V, fits the frequency dependence of
D, K and V (linear, square root and squared models), and fits DKI (DIPY) for
FA, MD, AD and RD.

Current assumption: exactly five waveforms, in this order: LTE 0 Hz,
LTE 50 Hz, LTE 100 Hz, STE isotropic, STE anisotropic.

Outputs in graphOutputs/<signal>/:
    poweder_average_signal.csv      Powder-averaged signal per waveform and b-value
    signal_fit_<signal>.svg         Signal decay with the fits
    Diffusivity_/Kurtosis_/Variance_<signal>.svg   Frequency dependence of D, K, V
    MD_AD_RD.png                    Frequency dependence of the DKI metrics
    results.csv                     FA, MD, AD, RD, D, kurtosis and variance per waveform

Usage (from the project root):
    pixi run -e dipy-env python src/fit_signal.py outputs/<config>/<signal>.csv
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dki

from analysis_utils import (output_dir, load_signals, powder_average, waveform_label,
                            fit_cumulant, fit_frequency_models, plot_frequency_models)

parser = argparse.ArgumentParser(description="Fit the signal (D, K, V) and DKI.")
parser.add_argument("signals", help="Signal CSV written by Simulation.py")
args = parser.parse_args()

name, output = output_dir(args.signals)
df = load_signals(args.signals)

df_averaged = powder_average(df)
df_averaged.reset_index().to_csv(f"{output}/poweder_average_signal.csv", index=False)

#Fit the signal decay of each waveform
fits = []
x_fit = np.linspace(0, max(df['bval']) / 1000, 100)
fig, ax = plt.subplots(figsize=(9, 5))

for wf in sorted(df['waveform_idx'].unique()):
    wf_data = df_averaged.loc[wf]
    b_arr = np.array(wf_data.index) / 1000
    signals = wf_data.values

    fit = fit_cumulant(b_arr, signals)
    fits.append(fit)

    label_name = waveform_label(df, wf)
    y_fit = np.exp(fit["A"] * (x_fit**2) + fit["B"] * x_fit + fit["C"])

    ax.scatter(b_arr, signals, marker='o', label=rf"$\bf{{{label_name}\ Data}}$")
    ax.plot(x_fit, y_fit, linestyle='--',
            label=f"D: {fit['D']:.4f} µm²/ms\nK: {fit['K']:.4f}\nV: {fit['V']:.4f} µm⁴/ms²")

ax.set_yscale('log')
ax.set_xlabel("b-value (ms/µm²)")
ax.set_ylabel("Normalized Signal ($S/S_0$)")
ax.set_title("Signal Decay")
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

plt.subplots_adjust(right=0.7)
plt.savefig(f"{output}/signal_fit_{name}.svg", dpi=300, bbox_inches='tight')
plt.close(fig)

d = [fit["D"] for fit in fits]
kurt = [fit["K"] for fit in fits]
variance = [fit["V"] for fit in fits]

#Frequency dependence of D, K and V for the three LTE waveforms
freq = np.array([0, 50, 100])
x1 = np.linspace(0, 200, 100)

paramList = [("Diffusivity", np.array(d[:3]), "Diffusivity (µm²/ms)"),
             ("Kurtosis", np.array(kurt[:3]), "Kurtosis"),
             ("Variance", np.array(variance[:3]), "Variance (µm⁴/ms²)")]

for param_name, param, ylabel in paramList:
    freq_fits, best = fit_frequency_models(freq, param)
    print(f"{param_name} Best Fit: {best}")

    fig, ax = plt.subplots()
    ax.scatter(freq, param, color='red', marker='o', s=10, zorder=5, label=f'{param_name} Data Points')
    plot_frequency_models(ax, freq_fits, best, x1, label_suffix=" Fit")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"Frequency Dependence of {param_name}")
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{output}/{param_name}_{name}.svg", dpi=300)
    plt.close(fig)

#DKI fit for the first three (LTE) waveforms
R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]

md = []
ad = []
rd = []
fa = []

for count, (waveform_file, df_wave) in enumerate(df.groupby("file", sort=False)):
    if count > 2:
        break
    print(f"\nWaveform: {waveform_file}")

    bvals = df_wave["bval"].to_numpy()
    signals = df_wave["signal"].to_numpy()

    #Extract gradient directions
    rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
    bvecs = rot_matrices[:, 0, :]

    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    fit = dki.DiffusionKurtosisModel(gtab).fit(signals)

    fa.append(fit.fa)
    print(f"Fractional Anisotropy (FA): {fit.fa:.3f}")
    md.append(fit.md * 1000)
    print(f"Mean Diffusivity (MD):     {fit.md*1000:.4f} um^2/ms")
    ad.append(fit.ad * 1000)
    print(f"Axial Diffusivity (AD):    {fit.ad*1000:.4f} um^2/ms")
    rd.append(fit.rd * 1000)
    print(f"Radial Diffusivity (RD):   {fit.rd*1000:.4f} um^2/ms")

#Plot DKI metrics vs frequency
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

metrics = [
    ("Mean Diffusivity (MD)", np.array(md), axes[0]),
    ("Axial Diffusivity (AD)", np.array(ad), axes[1]),
    ("Radial Diffusivity (RD)", np.array(rd), axes[2]),
]
all_values = md + ad + rd

for metric_name, diff, ax in metrics:
    freq_fits, best = fit_frequency_models(freq, diff)

    ax.scatter(freq, diff, color="red", marker="o", s=30, zorder=5, label="Data Points")
    plot_frequency_models(ax, freq_fits, best, x1)

    #Display best-fit equation and values
    slope, intercept, _ = freq_fits[best]
    power = {"Linear": "x", "Square Root": "x^1/2", "Squared": "x^2"}[best]
    eq_text = (
        f"Best Fit: {best}\n"
        f"y = {slope:.3f}{power} {intercept:+.3f}\n\n"
        f"0 Hz   = {diff[0]:.3f}\n"
        f"50 Hz  = {diff[1]:.3f}\n"
        f"100 Hz = {diff[2]:.3f}"
    )
    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

    ax.set_ylim(min(all_values) * 0.6, max(all_values) * 1.3)
    ax.set_ylabel(f"{metric_name} ($\\mathrm{{µm^2/ms}}$)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"{metric_name} vs Frequency")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(f"{output}/MD_AD_RD.png", dpi=300)
plt.close(fig)

#Export results
waveform = ["LTE-0Hz", "LTE-50Hz", "LTE-100Hz", "STE-Iso", "STE-Aniso"]

results_rows = []
# LTE
for i in range(3):
    results_rows.append({
        "Waveform": waveform[i],
        "FA": fa[i],
        "MD": md[i],
        "AD": ad[i],
        "RD": rd[i],
        "D": d[i],
        "Kurtosis": kurt[i],
        "Variance": variance[i],
    })
#STE
for j in range(3, 5):
    results_rows.append({
        "Waveform": waveform[j],
        "FA": "",
        "MD": "",
        "AD": "",
        "RD": "",
        "D": d[j],
        "Kurtosis": kurt[j],
        "Variance": variance[j],
    })

results_df = pd.DataFrame(
    results_rows,
    columns=["Waveform", "FA", "MD", "AD", "RD", "D", "Kurtosis", "Variance"],
)
results_df.to_csv(f"{output}/results.csv", index=False)
print(f"\nSaved results to {output}/")

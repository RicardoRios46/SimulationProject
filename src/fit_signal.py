"""
Signal fitting and DKI analysis
Fits the powder-averaged signal (mean over rotations) of each waveform for
the diffusivity D, kurtosis K and variance V, fits the frequency dependence of
D, K and V of the LTE waveforms (linear, square root and squared models), and
fits DKI (DIPY) on each LTE waveform for FA, MD, AD and RD.

The waveforms are identified from their files in waveforms/ (see
analysis_utils.waveform_info): the encoding (LTE, STE or other) from the
b-tensor shape, and the frequency as the centroid frequency of the dephasing
spectrum |Q(f)|^2.

Outputs in graphOutputs/<signal>/:
    poweder_average_signal.csv      Powder-averaged signal per waveform and b-value
    signal_fit_<signal>.svg         Signal decay with the fits
    Diffusivity_/Kurtosis_/Variance_<signal>.svg   Frequency dependence of D, K, V (LTE)
    MD_AD_RD.png                    Frequency dependence of the DKI metrics (LTE)
    results.csv                     Encoding, frequency, FA, MD, AD, RD, D, kurtosis
                                    and variance per waveform

Usage (from the project root):
    pixi run -e dipy-env python src/fit_signal.py outputs/<config>/<signal>.csv
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dki

from analysis_utils import (output_dir, load_signals, powder_average, waveform_info,
                            fit_cumulant, fit_frequency_models, plot_frequency_models)

parser = argparse.ArgumentParser(description="Fit the signal (D, K, V) and DKI.")
parser.add_argument("signals", help="Signal CSV written by Simulation.py")
args = parser.parse_args()

name, output = output_dir(args.signals)
df = load_signals(args.signals)
info = waveform_info(df, args.signals)
waveforms = sorted(info)
lte = [wf for wf in waveforms if info[wf]["encoding"] == "LTE"]

print("Waveforms:")
for wf in waveforms:
    print(f"  {wf}: {info[wf]['label']:15s} {info[wf]['encoding']:5s} centroid {info[wf]['frequency']:.1f} Hz")

df_averaged = powder_average(df)
df_averaged.reset_index().to_csv(f"{output}/poweder_average_signal.csv", index=False)

#Fit the signal decay of each waveform
fits = {}
x_fit = np.linspace(0, max(df['bval']) / 1000, 100)
fig, ax = plt.subplots(figsize=(9, 5))

for wf in waveforms:
    wf_data = df_averaged.loc[wf]
    b_arr = np.array(wf_data.index) / 1000
    signals = wf_data.values

    fit = fit_cumulant(b_arr, signals)
    fits[wf] = fit

    label_name = info[wf]["label"]
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

#Frequency dependence of D, K and V for the LTE waveforms
freq = np.array([info[wf]["frequency"] for wf in lte])
x1 = np.linspace(0, max(200, 1.1 * freq.max()) if lte else 200, 100)

#Each model has two parameters, so at least three frequencies are needed to compare them
if len(lte) < 3:
    print("\nLess than three LTE waveforms, skipping the frequency dependence fits")
else:
    paramList = [("Diffusivity", "D", "Diffusivity (µm²/ms)"),
                 ("Kurtosis", "K", "Kurtosis"),
                 ("Variance", "V", "Variance (µm⁴/ms²)")]

    for param_name, key, ylabel in paramList:
        param = np.array([fits[wf][key] for wf in lte])
        freq_fits, best = fit_frequency_models(freq, param)
        print(f"{param_name} Best Fit: {best}")

        fig, ax = plt.subplots()
        ax.scatter(freq, param, color='red', marker='o', s=10, zorder=5, label=f'{param_name} Data Points')
        plot_frequency_models(ax, freq_fits, best, x1, label_suffix=" Fit")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Centroid frequency (Hz)")
        ax.set_title(f"Frequency Dependence of {param_name}")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(f"{output}/{param_name}_{name}.svg", dpi=300)
        plt.close(fig)

#DKI fit for each LTE waveform
R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
dki_results = {}

for wf in lte:
    df_wave = df[df['waveform_idx'] == wf]
    print(f"\nWaveform: {info[wf]['label']}")

    bvals = df_wave["bval"].to_numpy()
    signals = df_wave["signal"].to_numpy()

    #Gradient direction of each rotation: Simulation.py rotates the waveform
    #as g @ R.T, so the LTE direction u becomes R @ u
    rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
    bvecs = rot_matrices @ info[wf]["direction"]

    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    fit = dki.DiffusionKurtosisModel(gtab).fit(signals)

    dki_results[wf] = {"FA": fit.fa, "MD": fit.md * 1000, "AD": fit.ad * 1000, "RD": fit.rd * 1000}
    print(f"Fractional Anisotropy (FA): {fit.fa:.3f}")
    print(f"Mean Diffusivity (MD):     {fit.md*1000:.4f} um^2/ms")
    print(f"Axial Diffusivity (AD):    {fit.ad*1000:.4f} um^2/ms")
    print(f"Radial Diffusivity (RD):   {fit.rd*1000:.4f} um^2/ms")

#Plot DKI metrics vs frequency
if len(lte) >= 3:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    metrics = [
        ("Mean Diffusivity (MD)", "MD", axes[0]),
        ("Axial Diffusivity (AD)", "AD", axes[1]),
        ("Radial Diffusivity (RD)", "RD", axes[2]),
    ]
    all_values = [dki_results[wf][key] for wf in lte for key in ["MD", "AD", "RD"]]

    for metric_name, key, ax in metrics:
        diff = np.array([dki_results[wf][key] for wf in lte])
        freq_fits, best = fit_frequency_models(freq, diff)

        ax.scatter(freq, diff, color="red", marker="o", s=30, zorder=5, label="Data Points")
        plot_frequency_models(ax, freq_fits, best, x1)

        #Display best-fit equation and values
        slope, intercept, _ = freq_fits[best]
        power = {"Linear": "x", "Square Root": "x^1/2", "Squared": "x^2"}[best]
        values_text = "\n".join(f"{info[wf]['label']} ({f:.1f} Hz) = {v:.3f}"
                                for wf, f, v in zip(lte, freq, diff))
        eq_text = (
            f"Best Fit: {best}\n"
            f"y = {slope:.3f}{power} {intercept:+.3f}\n\n"
            f"{values_text}"
        )
        ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

        ax.set_ylim(min(all_values) * 0.6, max(all_values) * 1.3)
        ax.set_ylabel(f"{metric_name} ($\\mathrm{{µm^2/ms}}$)")
        ax.set_xlabel("Centroid frequency (Hz)")
        ax.set_title(f"{metric_name} vs Frequency")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output}/MD_AD_RD.png", dpi=300)
    plt.close(fig)

#Export results, one row per waveform (DKI metrics only for LTE)
results_rows = []
for wf in waveforms:
    results_rows.append({
        "Waveform": info[wf]["label"],
        "Encoding": info[wf]["encoding"],
        "Frequency": info[wf]["frequency"],
        **dki_results.get(wf, {"FA": "", "MD": "", "AD": "", "RD": ""}),
        "D": fits[wf]["D"],
        "Kurtosis": fits[wf]["K"],
        "Variance": fits[wf]["V"],
    })

results_df = pd.DataFrame(
    results_rows,
    columns=["Waveform", "Encoding", "Frequency", "FA", "MD", "AD", "RD", "D", "Kurtosis", "Variance"],
)
results_df.to_csv(f"{output}/results.csv", index=False)
print(f"\nSaved results to {output}/")

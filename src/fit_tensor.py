"""
Tensor (DKI) analysis
Fits DKI (DIPY) on each LTE waveform, using all its rotations and b-values,
for FA, MD, AD and RD, and fits the frequency dependence of MD, AD and RD.

The waveforms are identified from their files in waveforms/ (see
analysis_utils.waveform_info): LTE waveforms from the b-tensor shape, the
frequency as the centroid frequency of the dephasing spectrum |Q(f)|^2, and
the LTE direction as the main axis of the b-tensor. Simulation.py rotates the
waveform as g @ R.T, so the gradient direction of each rotation R is R @ u.

Frequency dependence: with at least three LTE waveforms (frequencies), MD, AD
and RD are fitted against the centroid frequency with linear, square root and
squared models; the lowest least-squares error is reported as the best fit.

Outputs in graphOutputs/<signal>/:
    tensor_fit.csv              Per LTE waveform: frequency, FA, MD, AD, RD
    tensor_frequency_fit.csv    Frequency-dependence models per parameter
    MD_AD_RD.png                Frequency dependence of MD, AD and RD

Usage (from the project root):
    pixi run -e dipy-env python src/fit_tensor.py outputs/<config>/<signal>.csv
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dki

from analysis_utils import (output_dir, load_signals, waveform_info,
                            fit_frequency_models, plot_frequency_models)

parser = argparse.ArgumentParser(description="Fit DKI on the LTE waveforms (FA, MD, AD, RD).")
parser.add_argument("signals", help="Signal CSV written by Simulation.py")
args = parser.parse_args()

name, output = output_dir(args.signals)
df = load_signals(args.signals)
info = waveform_info(df, args.signals)
lte = [wf for wf in sorted(info) if info[wf]["encoding"] == "LTE"]

if not lte:
    raise SystemExit("No LTE waveforms in the signal file, nothing to fit")

#DKI fit for each LTE waveform
R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
dki_results = {}

for wf in lte:
    df_wave = df[df['waveform_idx'] == wf]
    print(f"\nWaveform: {info[wf]['label']} (centroid {info[wf]['frequency']:.1f} Hz)")

    bvals = df_wave["bval"].to_numpy()
    signals = df_wave["signal"].to_numpy()

    #Gradient direction of each rotation: the LTE direction u becomes R @ u
    rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
    bvecs = rot_matrices @ info[wf]["direction"]

    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    fit = dki.DiffusionKurtosisModel(gtab).fit(signals)

    dki_results[wf] = {"FA": fit.fa, "MD": fit.md * 1000, "AD": fit.ad * 1000, "RD": fit.rd * 1000}
    print(f"Fractional Anisotropy (FA): {fit.fa:.3f}")
    print(f"Mean Diffusivity (MD):     {fit.md*1000:.4f} um^2/ms")
    print(f"Axial Diffusivity (AD):    {fit.ad*1000:.4f} um^2/ms")
    print(f"Radial Diffusivity (RD):   {fit.rd*1000:.4f} um^2/ms")

pd.DataFrame([{
    "waveform_idx": wf,
    "Waveform": info[wf]["label"],
    "Encoding": info[wf]["encoding"],
    "Frequency": info[wf]["frequency"],
    **dki_results[wf],
} for wf in lte]).to_csv(f"{output}/tensor_fit.csv", index=False)

#Frequency dependence of MD, AD and RD
#Each model has two parameters, so at least three frequencies are needed to compare them
frequency_rows = []
if len(lte) < 3:
    print("\nLess than three LTE waveforms, skipping the frequency dependence fits")
else:
    freq = np.array([info[wf]["frequency"] for wf in lte])
    x1 = np.linspace(0, max(200, 1.1 * freq.max()), 100)

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
        print(f"{key} Best Fit: {best}")
        for model, (slope, intercept, error) in freq_fits.items():
            frequency_rows.append({"Parameter": key, "Model": model, "Slope": slope,
                                   "Intercept": intercept, "SSE": error, "Best": model == best})

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

pd.DataFrame(frequency_rows, columns=["Parameter", "Model", "Slope", "Intercept", "SSE", "Best"]
             ).to_csv(f"{output}/tensor_frequency_fit.csv", index=False)
print(f"\nSaved results to {output}/")

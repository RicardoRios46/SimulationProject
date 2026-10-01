"""
Powder-average analysis
Averages the signal over the rotations (powder average) for each waveform and
b-value, fits the cumulant expansion of the log-signal for each waveform, and
fits the frequency dependence of D, K and V for each encoding (LTE, STE).

The waveforms are identified from their files in waveforms/ (see
analysis_utils.waveform_info): the encoding (LTE, STE or other) from the
b-tensor shape, and the frequency as the centroid frequency of the dephasing
spectrum |Q(f)|^2.

Cumulant fit of log(signal), b in ms/µm²:
    order 2:            log S = C + B b + A b^2
    order 3 (default):  log S = C + B b + A b^2 + E b^3
    D = -B, V = 2A, K = 3V/D², k3 = -6E, skewness = k3 / V^(3/2)
C is fitted, or fixed at 0 (S = 1 at b = 0) with --fix-intercept.

Frequency dependence: for each encoding with at least three waveforms
(frequencies), D, K and V are fitted against the centroid frequency with
linear, square root and squared models; the lowest least-squares error is
reported as the best fit. Encodings with fewer waveforms are only plotted.

Outputs in graphOutputs/<signal>/ (overwritten by each run, whatever the options):
    powder_average_signal.csv           Powder-averaged signal per waveform and b-value
    powder_average_fit.csv              Per waveform: encoding, frequency, fit settings,
                                        coefficients C, B, A, E, D, kurtosis, variance
                                        (and k3, skewness for order 3)
    powder_average_frequency_fit.csv    Frequency-dependence models per encoding and parameter
    signal_fit_<signal>.svg             Signal decay with the fits (D, K, V and, for order 3, k3)
    Diffusivity_/Kurtosis_/Variance_<signal>.svg   Frequency dependence of D, K, V

Usage (from the project root):
    pixi run -e dipy-env python src/fit_powder_average.py outputs/<config>/<signal>.csv [--order 2] [--fix-intercept]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (output_dir, load_signals, powder_average, waveform_info,
                            fit_cumulant, fit_frequency_models, plot_frequency_models)

parser = argparse.ArgumentParser(description="Powder-average the signal and fit it (D, K, V).")
parser.add_argument("signals", help="Signal CSV written by Simulation.py")
parser.add_argument("--order", type=int, choices=[2, 3], default=3,
                    help="Order of the polynomial in b fitted to log(signal) (default 3)")
parser.add_argument("--fix-intercept", action="store_true",
                    help="Fix the intercept C = 0 (signal = 1 at b = 0) instead of fitting it")
args = parser.parse_args()
intercept = "fixed at 0" if args.fix_intercept else "fitted"
print(f"Cumulant fit: order {args.order}, intercept {intercept}")

name, output = output_dir(args.signals)
df = load_signals(args.signals)
info = waveform_info(df, args.signals)
waveforms = sorted(info)

print("Waveforms:")
for wf in waveforms:
    print(f"  {wf}: {info[wf]['label']:15s} {info[wf]['encoding']:5s} centroid {info[wf]['frequency']:.1f} Hz")

#Powder average
df_averaged = powder_average(df)
averaged_table = df_averaged.reset_index()
averaged_table.insert(1, "Waveform", averaged_table["waveform_idx"].map(lambda wf: info[wf]["label"]))
averaged_table.to_csv(f"{output}/powder_average_signal.csv", index=False)

#Fit the signal decay of each waveform
fits = {}
for wf in waveforms:
    wf_data = df_averaged.loc[wf]
    b_arr = np.array(wf_data.index) / 1000
    fits[wf] = fit_cumulant(b_arr, wf_data.values, order=args.order, fix_intercept=args.fix_intercept)

fit_rows = []
for wf in waveforms:
    fit = fits[wf]
    row = {
        "waveform_idx": wf,
        "Waveform": info[wf]["label"],
        "Encoding": info[wf]["encoding"],
        "Frequency": info[wf]["frequency"],
        "Order": args.order,
        "FixedIntercept": args.fix_intercept,
        **{key: fit[key] for key in ["C", "B", "A", "E"]},
        "D": fit["D"],
        "Kurtosis": fit["K"],
        "Variance": fit["V"],
    }
    if args.order == 3:
        row.update({"k3": fit["k3"], "Skewness": fit["skewness"]})
    fit_rows.append(row)
pd.DataFrame(fit_rows).to_csv(f"{output}/powder_average_fit.csv", index=False)

#Signal decay with the fitted curves (order 3 shows k3; the skewness, which
#blows up when V ~ 0, is only in the CSV)
x_fit = np.linspace(0, max(df['bval']) / 1000, 100)
fig, ax = plt.subplots(figsize=(9, 5))

for wf in waveforms:
    wf_data = df_averaged.loc[wf]
    b_arr = np.array(wf_data.index) / 1000
    label_name = info[wf]["label"]

    fit = fits[wf]
    y_fit = np.exp(np.polyval([fit["E"], fit["A"], fit["B"], fit["C"]], x_fit))
    fit_label = f"D: {fit['D']:.4f} µm²/ms\nK: {fit['K']:.4f}\nV: {fit['V']:.4f} µm⁴/ms²"
    if args.order == 3:
        fit_label += f"\nk3: {fit['k3']:.4f} µm⁶/ms³"
    ax.scatter(b_arr, wf_data.values, marker='o', label=rf"$\bf{{{label_name}\ Data}}$")
    ax.plot(x_fit, y_fit, linestyle='--', label=fit_label)

ax.set_yscale('log')
ax.set_xlabel("b-value (ms/µm²)")
ax.set_ylabel("Normalized Signal ($S/S_0$)")
ax.set_title(f"Signal Decay (order {args.order} fit, intercept {intercept})")
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

plt.subplots_adjust(right=0.7)
plt.savefig(f"{output}/signal_fit_{name}.svg", dpi=300, bbox_inches='tight')
plt.close(fig)

#Frequency dependence of D, K and V, one panel per encoding
encodings = ["LTE", "STE"]
groups = {enc: [wf for wf in waveforms if info[wf]["encoding"] == enc] for enc in encodings}
max_freq = max(info[wf]["frequency"] for wf in waveforms)
x1 = np.linspace(0, max(200, 1.1 * max_freq), 100)

paramList = [("Diffusivity", "D", "Diffusivity (µm²/ms)"),
             ("Kurtosis", "K", "Kurtosis"),
             ("Variance", "V", "Variance (µm⁴/ms²)")]

frequency_rows = []
for param_name, key, ylabel in paramList:
    fig, axes = plt.subplots(1, len(encodings), figsize=(6 * len(encodings), 4.5), sharey=True)

    for enc, ax in zip(encodings, axes):
        group = groups[enc]
        freq = np.array([info[wf]["frequency"] for wf in group])
        param = np.array([fits[wf][key] for wf in group])

        ax.scatter(freq, param, color='red', marker='o', s=10, zorder=5, label=f'{param_name} Data Points')
        for wf, f, p in zip(group, freq, param):
            ax.annotate(info[wf]["label"], (f, p), textcoords="offset points", xytext=(4, 4), fontsize=7)

        #Each model has two parameters, so at least three frequencies are needed to compare them
        if len(group) >= 3:
            freq_fits, best = fit_frequency_models(freq, param)
            print(f"{enc} {param_name} Best Fit: {best}")
            plot_frequency_models(ax, freq_fits, best, x1, label_suffix=" Fit")
            for model, (slope, model_intercept, error) in freq_fits.items():
                frequency_rows.append({"Encoding": enc, "Parameter": param_name, "Model": model,
                                       "Slope": slope, "Intercept": model_intercept, "SSE": error,
                                       "Best": model == best})
        else:
            ax.text(0.5, 0.5, f"{len(group)} {enc} waveform(s):\nat least 3 frequencies\nneeded for the model fits",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8, color="gray")

        ax.set_xlim(x1[0], x1[-1])
        ax.set_xlabel("Centroid frequency (Hz)")
        ax.set_title(f"{param_name}: {enc}")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        if len(group):
            ax.legend(loc='best', fontsize=8)
    axes[0].set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f"{output}/{param_name}_{name}.svg", dpi=300)
    plt.close(fig)

pd.DataFrame(frequency_rows, columns=["Encoding", "Parameter", "Model", "Slope", "Intercept", "SSE", "Best"]
             ).to_csv(f"{output}/powder_average_frequency_fit.csv", index=False)
print(f"\nSaved results to {output}/")

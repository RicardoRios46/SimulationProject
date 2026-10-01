"""
Isotropic and anisotropic variance across runs (V_iso analysis)
Combines simulation runs of the same substrate and walker position (e.g. the
LTE/STE run and the TDE run, which need their own configs) and compares the
cumulant fits of the waveforms at each centroid frequency:

    V_iso(f)   = V of the STE series (the TDE waveforms by default)
    V_aniso(f) = V_LTE(f) - V_iso(f), pairing each LTE waveform with the STE
                 series waveform of the nearest centroid frequency

The other STE waveforms (e.g. STEiso, STEaniso, whose axes have different
frequencies) are shown at their own centroid frequency. The frequency
dependence of V_iso and V_aniso is fitted with the linear, square root and
squared models (at least three frequencies).

The waveforms are identified from their files as in fit_powder_average.py
(encoding from the b-tensor shape, frequency = centroid of |Q(f)|^2), and
fitted with the same cumulant expansion (see analysis_utils.fit_cumulant):
    log S = C + B b + A b^2 (+ E b^3 for order 3), D = -B, V = 2A, K = 3V/D²

Outputs in graphOutputs/viso/<name>/:
    viso_fit.csv              Per waveform: run, encoding, frequency, D, K, V, k3, skewness
    viso_pairs.csv            Per LTE frequency: V_LTE, V_iso, V_aniso (and D, K of both)
    viso_frequency_fit.csv    Frequency models of V_iso and V_aniso
    viso_steaniso_vs_steiso.csv   D, K, V of STEiso and STEaniso and their differences
                              (STEaniso - STEiso), if both waveforms are in the runs
    viso_<name>.svg           D, K and V against the centroid frequency

Usage (from the project root):
    pixi run -e dipy-env python src/fit_viso.py <signal.csv> [<signal.csv> ...] --name <name> [--order 3] [--fix-intercept]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (load_signals, powder_average, waveform_info, fit_cumulant,
                            fit_frequency_models, FREQUENCY_MODELS)

parser = argparse.ArgumentParser(description="Compare V of LTE and STE waveforms across runs (V_iso, V_aniso).")
parser.add_argument("signals", nargs="+", help="Signal CSVs written by Simulation.py (same substrate and position)")
parser.add_argument("--name", required=True, help="Output name, results in graphOutputs/viso/<name>/")
parser.add_argument("--order", type=int, choices=[2, 3], default=3,
                    help="Order of the polynomial in b fitted to log(signal) (default 3)")
parser.add_argument("--fix-intercept", action="store_true",
                    help="Fix the intercept C = 0 (signal = 1 at b = 0) instead of fitting it")
parser.add_argument("--ste-series", default="TDE",
                    help="Name prefix of the STE waveforms that give V_iso(f) (default TDE)")
parser.add_argument("--ste-iso", default="STEiso",
                    help="Waveform compared with --ste-aniso (default STEiso)")
parser.add_argument("--ste-aniso", default="STEaniso",
                    help="Waveform compared with --ste-iso: V_STEaniso - V_STEiso (default STEaniso)")
parser.add_argument("--model-criterion", choices=["sse", "aic"], default="sse",
                    help="Criterion for the best frequency model (default sse; aic ranks the same "
                         "while all models have 2 parameters, see analysis_utils.fit_frequency_models)")
args = parser.parse_args()
intercept = "fixed at 0" if args.fix_intercept else "fitted"
print(f"Cumulant fit: order {args.order}, intercept {intercept}")

output = f"graphOutputs/viso/{args.name}"
os.makedirs(output, exist_ok=True)

#Fit every waveform of every run
rows = []
for signal_file in args.signals:
    df = load_signals(signal_file)
    info = waveform_info(df, signal_file)
    df_averaged = powder_average(df)
    for wf in sorted(info):
        wf_data = df_averaged.loc[wf]
        b_arr = np.array(wf_data.index) / 1000
        fit = fit_cumulant(b_arr, wf_data.values, order=args.order, fix_intercept=args.fix_intercept)
        label = info[wf]["label"]
        if info[wf]["encoding"] == "LTE":
            series = "LTE"
        elif info[wf]["encoding"] == "STE" and label.startswith(args.ste_series):
            series = "V_iso"
        else:
            series = "other"
        rows.append({"Run": Path(signal_file).stem, "Waveform": label, "Encoding": info[wf]["encoding"],
                     "Series": series, "Frequency": info[wf]["frequency"], "D": fit["D"],
                     "Kurtosis": fit["K"], "Variance": fit["V"], "k3": fit["k3"], "Skewness": fit["skewness"]})

fits = pd.DataFrame(rows).sort_values(["Series", "Frequency"])
fits.insert(0, "Order", args.order)
fits.to_csv(f"{output}/viso_fit.csv", index=False)

lte = fits[fits.Series == "LTE"].sort_values("Frequency")
ste = fits[fits.Series == "V_iso"].sort_values("Frequency")
other = fits[fits.Series == "other"]
print("\n" + fits[["Run", "Waveform", "Series", "Frequency", "D", "Kurtosis", "Variance"]].round(4).to_string(index=False))

#Pair each LTE waveform with the STE series waveform of the nearest frequency
pair_rows = []
for _, l in lte.iterrows():
    s = ste.iloc[np.argmin(np.abs(ste.Frequency.to_numpy() - l.Frequency))]
    pair_rows.append({"LTE": l.Waveform, "STE": s.Waveform, "Frequency_LTE": l.Frequency,
                      "Frequency_STE": s.Frequency, "D_LTE": l.D, "D_STE": s.D,
                      "K_LTE": l.Kurtosis, "K_STE": s.Kurtosis,
                      "V_LTE": l.Variance, "V_iso": s.Variance, "V_aniso": l.Variance - s.Variance})
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(f"{output}/viso_pairs.csv", index=False)
print("\n" + pairs[["LTE", "STE", "Frequency_LTE", "Frequency_STE", "V_LTE", "V_iso", "V_aniso"]].round(4).to_string(index=False))

#Effect of the mixed frequencies of STEaniso: STEaniso - STEiso (both isotropic
#b-tensors; STEiso ~47.6 Hz on every axis, STEaniso different per axis)
by_name = fits.set_index("Waveform")
if {args.ste_iso, args.ste_aniso} <= set(by_name.index):
    iso, aniso = by_name.loc[args.ste_iso], by_name.loc[args.ste_aniso]
    comparison = pd.DataFrame([{
        "STEiso": args.ste_iso, "STEaniso": args.ste_aniso,
        "Frequency_STEiso": iso.Frequency, "Frequency_STEaniso": aniso.Frequency,
        **{f"{key}_{name}": row[key] for key in ["D", "Kurtosis", "Variance"]
           for name, row in [("STEiso", iso), ("STEaniso", aniso)]},
        "dD": aniso.D - iso.D, "dK": aniso.Kurtosis - iso.Kurtosis,
        "dV (STEaniso - STEiso)": aniso.Variance - iso.Variance,
    }])
    comparison.to_csv(f"{output}/viso_steaniso_vs_steiso.csv", index=False)
    print(f"\nV_STEaniso - V_STEiso = {aniso.Variance:.4f} - {iso.Variance:.4f} = "
          f"{aniso.Variance - iso.Variance:.4f} (dD {aniso.D - iso.D:.4f}, dK {aniso.Kurtosis - iso.Kurtosis:.4f})")

#Frequency dependence of V_iso and V_aniso (each model has two parameters, so
#at least three frequencies are needed to compare them)
frequency_rows = []
best_fits = {}
for quantity, freq, values in [("V_iso", ste.Frequency.to_numpy(), ste.Variance.to_numpy()),
                               ("V_aniso", pairs.Frequency_STE.to_numpy(), pairs.V_aniso.to_numpy())]:
    if len(values) < 3:
        print(f"\nLess than three frequencies for {quantity}, skipping the frequency models")
        continue
    freq_fits, best = fit_frequency_models(freq, values, args.model_criterion)
    best_fits[quantity] = (freq_fits[best], best)
    print(f"{quantity} Best Fit: {best}")
    for model, model_fit in freq_fits.items():
        frequency_rows.append({"Quantity": quantity, "Model": model, **model_fit, "Best": model == best})
pd.DataFrame(frequency_rows, columns=["Quantity", "Model", "Slope", "Intercept", "SSE", "AIC", "AkaikeWeight", "Best"]
             ).to_csv(f"{output}/viso_frequency_fit.csv", index=False)

#D, K and V against the centroid frequency
x = np.linspace(0, 1.1 * fits.Frequency.max(), 200)   # model curves only near the data
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
panels = [("D", "D (µm²/ms)"), ("Kurtosis", "K"), ("Variance", "V (µm⁴/ms²)")]

for (key, ylabel), ax in zip(panels, axes):
    ax.plot(lte.Frequency, lte[key], "o-", color="tab:blue", label="LTE")
    ax.plot(ste.Frequency, ste[key], "s-", color="tab:red",
            label=f"STE ({args.ste_series})" + (" = V_iso" if key == "Variance" else ""))
    for (_, o), marker in zip(other.iterrows(), ["^", "D", "v", "P"]):
        ax.plot(o.Frequency, o[key], marker, color="black", markerfacecolor="none", markersize=8, label=o.Waveform)
    if key == "Variance":
        ax.plot(pairs.Frequency_STE, pairs.V_aniso, "d-", color="tab:green", label="V_aniso = V_LTE - V_iso")
        for quantity, color in [("V_iso", "tab:red"), ("V_aniso", "tab:green")]:
            if quantity in best_fits:
                model_fit, best = best_fits[quantity]
                slope, model_intercept = model_fit["Slope"], model_fit["Intercept"]
                ax.plot(x, slope * FREQUENCY_MODELS[best](x) + model_intercept, ":", color=color,
                        label=f"{quantity} fit: {best}")
        ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Centroid frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)

fig.suptitle(f"{args.name} (order {args.order} fit, intercept {intercept})")
plt.tight_layout()
plt.savefig(f"{output}/viso_{args.name}.svg", dpi=300)
plt.close(fig)
print(f"\nSaved results to {output}/")

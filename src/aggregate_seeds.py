"""
Seed aggregation of the V_iso analysis
Combines the fit_viso.py results of repeated simulations that differ only in
the random seed (same substrate, position, waveforms and fit settings) and
reports the mean, standard deviation (SD) and standard error of the mean
(SEM = SD / sqrt(n)) over the seeds:

    - D, K, V (and k3 to k5) of every waveform
    - V_LTE, V_iso and V_aniso of every LTE / STE-series pair
    - Paired differences, computed per seed and then averaged:
        STE_I - STE series   STE_I against the STE-series waveform (TDE) of
                             the nearest centroid frequency
        STE_A - STE_I        effect of the mixed frequencies of STE_A
      with t = mean / SEM: |t| above ~2.6 (n = 6, 95%, two-sided) means the
      difference is larger than the Monte Carlo and fitting noise.

STE_I and STE_A are found by name prefix (--ste-iso, --ste-aniso; default
STEiso and STEaniso, so STEiso_pad is matched too); the STE series is the
"V_iso" series of fit_viso.py.

Input: one fit_viso.py output folder per seed, graphOutputs/viso/<name>/,
with viso_fit.csv and viso_pairs.csv. Make them first, e.g. for 6 seeds:

    for rep in 1 2 3 4 5 6; do
        pixi run -e dipy-env python src/fit_viso.py \\
            outputs/viso2_cylgamma_lte_intra_rep$rep/viso2_cylgamma_lte_intra_rep$rep.csv \\
            outputs/viso2_cylgamma_long_intra_rep$rep/viso2_cylgamma_long_intra_rep$rep.csv \\
            --name viso2_cylgamma_intra_rep${rep}_order3
    done

Outputs in graphOutputs/viso/<name>/:
    seeds_fit.csv           Per waveform and parameter: n, mean, SD, SEM
    seeds_pairs.csv         Per LTE / STE-series pair: n, mean, SD, SEM of V_LTE, V_iso, V_aniso
    seeds_differences.csv   Paired differences: n, mean, SD, SEM, t, per-seed values
    seeds_<name>.svg        D, K and V against the centroid frequency, mean +- SD

Usage (from the project root):
    pixi run -e dipy-env python src/aggregate_seeds.py graphOutputs/viso/<run>_rep*_order3 --name <name>
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Mean, SD and SEM of the fit_viso.py results over seeds.")
parser.add_argument("folders", nargs="+", help="fit_viso.py output folders, one per seed")
parser.add_argument("--name", required=True, help="Output name, results in graphOutputs/viso/<name>/")
parser.add_argument("--ste-iso", default="STEiso", help="Name prefix of the STE_I waveform (default STEiso)")
parser.add_argument("--ste-aniso", default="STEaniso", help="Name prefix of the STE_A waveform (default STEaniso)")
args = parser.parse_args()

output = f"graphOutputs/viso/{args.name}"
os.makedirs(output, exist_ok=True)

fits = pd.concat([pd.read_csv(f"{folder}/viso_fit.csv").assign(Seed=folder) for folder in args.folders])
pairs = pd.concat([pd.read_csv(f"{folder}/viso_pairs.csv").assign(Seed=folder) for folder in args.folders])
n_seeds = len(args.folders)

if fits.Order.nunique() > 1:
    raise SystemExit(f"Error: the folders mix fit orders {sorted(fits.Order.unique())}")
order = fits.Order.iloc[0]
print(f"{n_seeds} seeds, order {order} fit")


def stats(group, columns):
    """n, mean, SD and SEM of the columns over the seeds in group."""
    row = {"n": len(group)}
    for col in columns:
        row[f"{col}_mean"] = group[col].mean()
        row[f"{col}_SD"] = group[col].std(ddof=1)
        row[f"{col}_SEM"] = row[f"{col}_SD"] / np.sqrt(len(group))
    return pd.Series(row)


#Per waveform: D, K, V and the higher cumulants
params = [col for col in ["D", "Kurtosis", "Variance", "k3", "k4", "k5"] if col in fits]
fit_stats = (fits.groupby(["Waveform", "Series", "Frequency"])[params]
             .apply(stats, columns=params).reset_index().sort_values(["Series", "Frequency"]))
fit_stats["n"] = fit_stats.n.astype(int)
fit_stats.to_csv(f"{output}/seeds_fit.csv", index=False)

#Per LTE / STE-series pair: V_LTE, V_iso, V_aniso
pair_stats = (pairs.groupby(["LTE", "STE", "Frequency_LTE", "Frequency_STE"])[["V_LTE", "V_iso", "V_aniso"]]
              .apply(stats, columns=["V_LTE", "V_iso", "V_aniso"]).reset_index().sort_values("Frequency_LTE"))
pair_stats["n"] = pair_stats.n.astype(int)
pair_stats.to_csv(f"{output}/seeds_pairs.csv", index=False)


def find(waveforms, prefix):
    """The waveform whose name starts with prefix (None if there is none)."""
    matches = [w for w in waveforms if w.startswith(prefix)]
    if len(matches) > 1:
        raise SystemExit(f"Error: several waveforms start with {prefix!r}: {matches}")
    return matches[0] if matches else None


#Paired differences, per seed
diff_rows = []
for seed, f in fits.groupby("Seed"):
    by_name = f.set_index("Waveform")
    ste_iso = find(by_name.index, args.ste_iso)
    ste_aniso = find(by_name.index, args.ste_aniso)
    series = f[f.Series == "V_iso"]
    if ste_iso is not None and len(series) and ste_iso not in set(series.Waveform):
        nearest = series.iloc[np.argmin(np.abs(series.Frequency.to_numpy() - by_name.Frequency[ste_iso]))]
        for key in ["D", "Variance"]:
            diff_rows.append({"Difference": f"{ste_iso} - {nearest.Waveform}", "Parameter": key, "Seed": seed,
                              "Value": by_name[key][ste_iso] - nearest[key]})
    if ste_iso is not None and ste_aniso is not None:
        for key in ["D", "Variance"]:
            diff_rows.append({"Difference": f"{ste_aniso} - {ste_iso}", "Parameter": key, "Seed": seed,
                              "Value": by_name[key][ste_aniso] - by_name[key][ste_iso]})

diff_stats = []
for (difference, key), group in pd.DataFrame(diff_rows).groupby(["Difference", "Parameter"], sort=False):
    values = group.Value.to_numpy()
    sd = values.std(ddof=1)
    sem = sd / np.sqrt(len(values))
    diff_stats.append({"Difference": difference, "Parameter": key, "n": len(values), "mean": values.mean(),
                       "SD": sd, "SEM": sem, "t": values.mean() / sem,
                       "values": " ".join(f"{v:.5f}" for v in values)})
diff_stats = pd.DataFrame(diff_stats)
diff_stats.to_csv(f"{output}/seeds_differences.csv", index=False)

pd.set_option("display.width", 200)
print("\nV per waveform (mean +- SD over seeds):")
print(fit_stats[["Waveform", "Frequency", "n", "Variance_mean", "Variance_SD", "Variance_SEM"]].round(5).to_string(index=False))
print("\nV_aniso = V_LTE - V_iso:")
print(pair_stats[["LTE", "STE", "n", "V_aniso_mean", "V_aniso_SD"]].round(5).to_string(index=False))
print("\nPaired differences:")
print(diff_stats.drop(columns="values").round(5).to_string(index=False))

#D, K and V against the centroid frequency, mean +- SD
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
panels = [("D", "D (µm²/ms)"), ("Kurtosis", "K"), ("Variance", "V (µm⁴/ms²)")]
for (key, ylabel), ax in zip(panels, axes):
    for series, style, color, label in [("LTE", "o-", "tab:blue", "LTE"), ("V_iso", "s-", "tab:red", "STE series")]:
        s = fit_stats[fit_stats.Series == series]
        ax.errorbar(s.Frequency, s[f"{key}_mean"], yerr=s[f"{key}_SD"], fmt=style, color=color,
                    capsize=3, label=label + (" = V_iso" if key == "Variance" and series == "V_iso" else ""))
    for (_, o), marker in zip(fit_stats[fit_stats.Series == "other"].iterrows(), ["^", "D", "v", "P"]):
        ax.errorbar(o.Frequency, o[f"{key}_mean"], yerr=o[f"{key}_SD"], fmt=marker, color="black",
                    markerfacecolor="none", markersize=8, capsize=3, label=o.Waveform)
    if key == "Variance":
        ax.errorbar(pair_stats.Frequency_STE, pair_stats.V_aniso_mean, yerr=pair_stats.V_aniso_SD, fmt="d-",
                    color="tab:green", capsize=3, label="V_aniso = V_LTE - V_iso")
        ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Centroid frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)

fig.suptitle(f"{args.name}: mean +- SD over {n_seeds} seeds (order {order} fit)")
plt.tight_layout()
plt.savefig(f"{output}/seeds_{args.name}.svg", dpi=300)
plt.close(fig)
print(f"\nSaved results to {output}/")

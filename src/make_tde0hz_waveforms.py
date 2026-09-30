"""
TDE 0 Hz design candidates
The 0 Hz TDE of make_tde_waveforms.py is not an STE: the PGSE half of LTE0hz
does not return q to 0, so the q of each axis stays on while the next axes
play and the b-tensor is not isotropic. For a sequential TDE, each axis must
return q to 0 before the next axis starts (no overlap of q_x, q_y, q_z).

These waveforms are only for simulations, so there is no refocusing pulse
gap to respect (disimpy uses the effective gradient): the time is split into
three equal windows, one per axis, and in each window the gradient is a
positive lobe, an optional plateau of zeros, and a negative lobe, so q is a
triangle (no plateau, bipolar) or a trapezoid (with plateau, PGSE-like) that
returns to 0 at the end of the window. The lobes are trapezoids with the ramp
of the LTE0hz lobe.

All designs have the number of rows of waveforms/TDE50hz.csv (139.2 ms), so
they can be simulated in the same config as TDE50hz and TDE100hz:

    bipolar            triangle q, with the lead and tail zeros of TDE50hz
    pgse               plateau fraction of LTE0hz (gap / (2 lobes + gap)),
                       with the lead and tail zeros
    bipolar_full       triangle q, windows over the whole duration (only one
                       zero row at the start and end, so the gradient starts
                       and ends at 0)
    pgse_full          plateau fraction of LTE0hz, whole duration (one zero row
                       at the start and end)

The frequency is set mostly by the window length (longer = lower); a plateau
lowers it a little (the effect levels off at a plateau fraction of ~0.4).

The user chose pgse_full (8.6 Hz, closest to the 8.4 Hz of LTE0hz) as the
0 Hz TDE: it is also written as waveforms/TDE0hz.csv. The other designs are
kept to revisit them.

Writes waveforms/TDE0hz_designs/TDE0hz_<design>.csv (N x 3, git-ignored) and
waveforms/TDE0hz.csv, and prints a summary. Plot them with plot_waveforms.py.

Usage (from the project root):
    pixi run -e dipy-env python src/make_tde0hz_waveforms.py
    pixi run -e dipy-env python src/plot_waveforms.py waveforms/TDE0hz_designs/*.csv --output-dir graphOutputs/waveforms/TDE0hz_designs --fmax 200
    pixi run -e dipy-env python src/plot_waveforms.py waveforms/TDE0hz.csv
"""

import argparse
import os
import shutil

import numpy as np

from waveform_utils import analyze_waveform

parser = argparse.ArgumentParser(description="Build TDE 0 Hz design candidates (for simulations).")
parser.add_argument("--output-dir", default="waveforms/TDE0hz_designs",
                    help="Folder for the waveform files (default waveforms/TDE0hz_designs)")
parser.add_argument("--raster-time-ms", type=float, default=0.02,
                    help="Time between rows in ms (default 0.02, as in the simulation config)")
args = parser.parse_args()

#Duration, lead and tail zeros from TDE50hz, so the designs match it
tde = np.loadtxt("waveforms/TDE50hz.csv", delimiter=",")
n_rows = len(tde)
active = np.flatnonzero(np.abs(tde).sum(axis=1))
lead, tail = active[0], n_rows - active[-1] - 1

#Lobe ramp, amplitude and plateau fraction from the PGSE of LTE0hz
g = np.loadtxt("waveforms/LTE0hz.csv", delimiter=",")[:, 0]
nonzero = np.nonzero(g)[0]
k = np.argmax(np.diff(nonzero))
lte_lobe = g[nonzero[0]:nonzero[k] + 1]
lte_gap = nonzero[k + 1] - nonzero[k] - 1
amplitude = lte_lobe.max()
ramp = np.argmax(lte_lobe >= 0.999 * amplitude)
pgse_fraction = lte_gap / (2 * len(lte_lobe) + lte_gap)


def lobe(n):
    """Trapezoid lobe of n rows with the LTE0hz ramp and amplitude."""
    values = np.full(n, amplitude)
    r = min(ramp, n // 2)
    values[:r] = amplitude * np.arange(1, r + 1) / r
    values[n - r:] = values[:r][::-1]
    return values


def axis_block(window, plateau_fraction):
    """Positive lobe, plateau of zeros, negative lobe: q returns to 0 in the window."""
    n = int(window * (1 - plateau_fraction) // 2)
    return np.concatenate([lobe(n), np.zeros(window - 2 * n), -lobe(n)])


def build(plateau_fraction, lead, tail):
    """Three equal windows (x, y, z) between the lead and tail zeros (rows left
    over when the rows between them do not divide by 3 are zeros at the end)."""
    window = (n_rows - lead - tail) // 3
    waveform = np.zeros((n_rows, 3))
    for axis in range(3):
        start = lead + axis * window
        waveform[start:start + window, axis] = axis_block(window, plateau_fraction)
    return waveform, window


designs = {
    "bipolar": (0.0, lead, tail),
    "pgse": (pgse_fraction, lead, tail),
    "bipolar_full": (0.0, 1, 1),
    "pgse_full": (pgse_fraction, 1, 1),
}

os.makedirs(args.output_dir, exist_ok=True)
print(f"Rows {n_rows} ({n_rows * args.raster_time_ms:.1f} ms), lead/tail {lead}/{tail} rows (from TDE50hz); "
      f"LTE0hz lobe ramp {ramp} rows, plateau fraction {pgse_fraction:.2f}\n")
print(f"{'design':13s} {'window':>8s} {'plateau':>7s} {'b':>7s}  {'eig/b':17s}  {'centroid x/y/z (Hz)':20s} {'comb':>5s}")

for name, (plateau_fraction, design_lead, design_tail) in designs.items():
    waveform, window = build(plateau_fraction, design_lead, design_tail)
    file = f"{args.output_dir}/TDE0hz_{name}.csv"
    np.savetxt(file, waveform, delimiter=",", fmt="%.15g")

    w = analyze_waveform(file, args.raster_time_ms)
    eig = w["eigenvalues"] / w["b"]
    centroids = " ".join(f"{c:5.1f}" for c in w["centroids"])
    print(f"{name:13s} {window * args.raster_time_ms:5.1f} ms {plateau_fraction:7.2f} {w['b']:7.0f}  "
          f"{eig[0]:.3f} {eig[1]:.3f} {eig[2]:.3f}  {centroids:20s} {w['centroid_combined']:5.1f}")

#The chosen design is the 0 Hz TDE
chosen = "pgse_full"
shutil.copyfile(f"{args.output_dir}/TDE0hz_{chosen}.csv", "waveforms/TDE0hz.csv")

print(f"\nWrote {len(designs)} designs to {args.output_dir}/ (b at the file amplitude, "
      f"the simulation scales them to the b-values of the config)")
print(f"Wrote waveforms/TDE0hz.csv (design {chosen})")

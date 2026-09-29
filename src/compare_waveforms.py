"""
Waveform comparison
Overlays several gradient waveform files (N x 3 CSV, as used by
Simulation.py) to compare their properties, e.g. to check that the x, y and
z axes of an STE match its LTE component files.

Figure: one row per axis (x, y, z) and one column per quantity:
    - Gradient g(t)
    - Dephasing q(t)
    - Encoding spectrum |Q(f)|^2 with the centroid frequency (dashed line)
Each file has its own color; the first file is drawn thick and transparent,
so files that match it show inside its band. Axes without gradient in any file
are left out. Spectra are normalized per curve by default (--shared-scale to
compare encoding power).

A summary table (duration, b-value, b-tensor eigenvalues, refocusing,
centroid frequencies) is printed and saved as a CSV next to the figure.

Usage (from the project root):
    pixi run -e dipy-env python src/compare_waveforms.py waveforms/<file1>.csv waveforms/<file2>.csv [...]

Example:
    python src/compare_waveforms.py waveforms/STEiso.csv waveforms/STEiso_LTE1.csv \
        waveforms/STEiso_LTE2.csv waveforms/STEiso_LTE3.csv --name STEiso_components
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from waveform_utils import AXES, analyze_waveform

parser = argparse.ArgumentParser(description="Compare gradient waveform files.")
parser.add_argument("files", nargs="+", help="Waveform CSV files (N x 3)")
parser.add_argument("--labels", nargs="+", help="Legend labels, one per file (default: file names)")
parser.add_argument("--name", help="Output name (default: compare_<first file>)")
parser.add_argument("--shared-scale", action="store_true",
                    help="Normalize all spectra by the same maximum, to also compare encoding power "
                         "(default: each curve normalized to its own maximum)")
parser.add_argument("--raster-time-ms", type=float, default=0.02,
                    help="Time between rows in ms (default 0.02, as in the simulation config)")
parser.add_argument("--gradient-scale", type=float, default=1e-3,
                    help="Factor converting file values to T/m (default 1e-3, i.e. files in mT/m)")
parser.add_argument("--fmax", type=float, default=500,
                    help="Maximum frequency shown in the spectrum, in Hz (default 500)")
parser.add_argument("--output-dir", default="graphOutputs/waveforms",
                    help="Folder for the figure and table (default graphOutputs/waveforms)")
args = parser.parse_args()

waveforms = [analyze_waveform(f, args.raster_time_ms, args.gradient_scale) for f in args.files]
labels = args.labels or [w["name"] for w in waveforms]
if len(labels) != len(waveforms):
    raise SystemExit(f"Error: got {len(labels)} labels for {len(waveforms)} files")

name = args.name or f"compare_{waveforms[0]['name']}"
os.makedirs(args.output_dir, exist_ok=True)

# Only show the axes that have gradient in at least one file
active_axes = [i for i in range(3) if any(np.any(w["g"][:, i]) for w in waveforms)]

# Spectrum normalization: each curve to its own maximum (compares frequency
# content; identical waveforms overlap), or one maximum shared by all curves
# (also compares encoding power)
shared_max = max(w["power"].max() for w in waveforms)

# Fixed space (in inches) at the top for the title and legend
height = 3.3 * len(active_axes) + 0.7
fig, axs = plt.subplots(len(active_axes), 3, figsize=(16, height),
                        sharex="col", squeeze=False)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
legend_handles = []

for n, (w, label) in enumerate(zip(waveforms, labels)):
    color = colors[n % len(colors)]
    # First file thick and transparent, so files matching it show inside its band
    style = {"color": color, "linewidth": 4, "alpha": 0.4} if n == 0 else {"color": color, "linewidth": 1.2}

    for row, i in enumerate(active_axes):
        if not np.any(w["g"][:, i]):
            continue
        power = w["power"][:, i] / (shared_max if args.shared_scale else w["power"][:, i].max())

        line, = axs[row, 0].plot(w["t"], w["g"][:, i] * 1e3, **style)
        axs[row, 1].plot(w["t"], w["q"][:, i] * 1e-6, **style)
        axs[row, 2].plot(w["freqs"], power, **style)
        axs[row, 2].axvline(w["centroids"][i], color=color, linestyle="--",
                            label=f"{label} centroid: {w['centroids'][i]:.1f} Hz")

    legend_handles.append((line, label))

for row, i in enumerate(active_axes):
    axs[row, 0].set_ylabel(f"{AXES[i]}\ng (mT/m)")
    axs[row, 1].set_ylabel("q (rad/µm)")
    axs[row, 2].set_ylabel("|Q(f)|² (normalized)")
    axs[row, 2].set_xlim(0, args.fmax)
    axs[row, 2].legend(fontsize=7)
    for ax in axs[row]:
        ax.grid(True, linestyle="--", alpha=0.5)

axs[0, 0].set_title("Gradient")
axs[0, 1].set_title("Dephasing")
scale_note = "shared scale" if args.shared_scale else "each curve normalized"
axs[0, 2].set_title(f"Encoding spectrum (dephasing, {scale_note})")
axs[-1, 0].set_xlabel("Time (ms)")
axs[-1, 1].set_xlabel("Time (ms)")
axs[-1, 2].set_xlabel("Frequency (Hz)")

fig.suptitle(name, y=1 - 0.1 / height)
plt.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))
fig.legend([h for h, _ in legend_handles], [l for _, l in legend_handles],
           loc="upper center", bbox_to_anchor=(0.5, 1 - 0.4 / height), ncol=min(len(legend_handles), 6))

output = f"{args.output_dir}/{name}"
plt.savefig(f"{output}.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Summary table
rows = []
for w, label in zip(waveforms, labels):
    b = w["b"]
    rows.append({
        "file": label,
        "duration_ms": len(w["g"]) * args.raster_time_ms,
        "b_value": b,
        "eig1/b": w["eigenvalues"][0] / b,
        "eig2/b": w["eigenvalues"][1] / b,
        "eig3/b": w["eigenvalues"][2] / b,
        "refocusing": w["refocusing"],
        **{f"centroid_{axis}_hz": np.nan if c is None else c for axis, c in zip(AXES, w["centroids"])},
        "centroid_combined_hz": w["centroid_combined"],
    })
table = pd.DataFrame(rows)
table.to_csv(f"{output}.csv", index=False)

print(table.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.4g}"))
print(f"\nSaved {output}.png and {output}.csv")

"""
Signal decay plot
Plots the powder-averaged signal (mean over rotations) against b for each
waveform of a signal file written by Simulation.py. No fitting: see
fit_signal.py for the fits.

Output: graphOutputs/<signal>/signal_<signal>.svg

Usage (from the project root):
    pixi run -e dipy-env python src/plot_signal.py outputs/<config>/<signal>.csv
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import output_dir, load_signals, powder_average, waveform_label

parser = argparse.ArgumentParser(description="Plot the powder-averaged signal decay.")
parser.add_argument("signals", help="Signal CSV written by Simulation.py")
args = parser.parse_args()

name, output = output_dir(args.signals)
df = load_signals(args.signals)
df_averaged = powder_average(df)

fig, ax = plt.subplots(figsize=(9, 5))

for wf in sorted(df['waveform_idx'].unique()):
    wf_data = df_averaged.loc[wf]
    b_arr = np.array(wf_data.index) / 1000
    ax.plot(b_arr, wf_data.values, marker='o', label=waveform_label(df, wf))

ax.set_yscale('log')
ax.set_xlabel("b-value (ms/µm²)")
ax.set_ylabel("Normalized Signal ($S/S_0$)")
ax.set_title("Signal Decay")
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

plt.subplots_adjust(right=0.7)
plt.savefig(f"{output}/signal_{name}.svg", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {output}/signal_{name}.svg")

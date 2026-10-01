"""
Zero-padded waveforms
Pads waveforms with zero rows at the end to the length of a reference
waveform, e.g. the 58.16 ms STE waveforms to the 139.2 ms of the TDE series,
so they can be simulated in the same config (same duration and n_t) and
compared on equal terms.

The gradient is zero in the padding and q(t) is already back to 0, so the
b-tensor and the dephasing spectrum are unchanged and the signal is expected
to be the same apart from the Monte Carlo noise.

Writes waveforms/<name>_pad.csv for each input waveform/<name>.csv.

Usage (from the project root):
    pixi run -e dipy-env python src/pad_waveforms.py [STEiso STEaniso] [--like TDE50hz]

Check the results with plot_waveforms.py and compare_waveforms.py.
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Pad waveforms with zeros at the end to the length of a reference waveform.")
parser.add_argument("names", nargs="*", default=["STEiso", "STEaniso"],
                    help="Waveforms to pad, waveforms/<name>.csv (default STEiso STEaniso)")
parser.add_argument("--like", default="TDE50hz",
                    help="Reference waveform whose number of rows is used (default TDE50hz)")
args = parser.parse_args()

n_rows = len(np.loadtxt(f"waveforms/{args.like}.csv", delimiter=","))

for name in args.names:
    waveform = np.loadtxt(f"waveforms/{name}.csv", delimiter=",")
    if len(waveform) > n_rows:
        raise SystemExit(f"Error: {name} has {len(waveform)} rows, more than {args.like} ({n_rows})")
    padded = np.vstack([waveform, np.zeros((n_rows - len(waveform), 3))])
    np.savetxt(f"waveforms/{name}_pad.csv", padded, delimiter=",", fmt="%.15g")
    print(f"waveforms/{name}_pad.csv: {len(waveform)} -> {n_rows} rows")

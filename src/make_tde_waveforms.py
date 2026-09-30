"""
Triple diffusion encoding (TDE) waveforms
Builds spherical tensor encoding waveforms at the frequencies of the LTE
waveforms by concatenating the LTE blocks on the three axes:

    lead | x first half | y first half | z first half | gap | x second half | y second half | z second half | tail

The first and second halves, the gap for the refocusing pulse and the lead
and tail zeros are taken from waveforms/LTE<f>hz.csv (the second half already
includes the sign flip of the 180° pulse). The b-tensor is isotropic only if
each half returns q to 0 on its own (true for the oscillating 50 and 100 Hz
LTE waveforms). This is checked: if a half does not return q to 0 (the 0 Hz
PGSE waveform), the q of each axis stays on while the next axes play, the
b-tensor is not isotropic, and the waveform is still written, with a warning
and "_broken" in its name, e.g. to show the problem.

Writes, for each frequency f, waveforms/TDE<f>hz.csv (N x 3), or
waveforms/TDE<f>hz_broken.csv if the check fails. Its x, y and z components
are the LTE waveform blocks, so no separate component files are written.

Usage (from the project root):
    pixi run -e dipy-env python src/make_tde_waveforms.py [--frequencies 0 50 100]

Check the results with plot_waveforms.py and compare_waveforms.py.
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Build TDE waveforms from the LTE waveforms.")
parser.add_argument("--frequencies", nargs="+", type=int, default=[0, 50, 100],
                    help="Frequencies (Hz) of the LTE waveforms to use, waveforms/LTE<f>hz.csv (default 0 50 100)")
args = parser.parse_args()

for frequency in args.frequencies:
    lte_file = f"waveforms/LTE{frequency}hz.csv"
    lte = np.loadtxt(lte_file, delimiter=",")
    if np.any(lte[:, 1:]):
        raise SystemExit(f"Error: {lte_file} is expected to have gradient on x only")
    g = lte[:, 0]

    # Split the LTE waveform at the longest run of zeros between its first and
    # last non-zero rows: the gap for the refocusing pulse
    nonzero = np.nonzero(g)[0]
    k = np.argmax(np.diff(nonzero))
    first = g[nonzero[0]:nonzero[k] + 1]
    second = g[nonzero[k + 1]:nonzero[-1] + 1]
    lead = nonzero[0]
    gap = nonzero[k + 1] - nonzero[k] - 1
    tail = len(g) - nonzero[-1] - 1

    # Each half must return q to 0 on its own, otherwise the axes overlap in q
    # and the b-tensor is not isotropic: the waveform is still written, marked broken
    refocused = all(abs(np.cumsum(half)[-1]) <= 1e-6 * np.abs(np.cumsum(half)).max()
                    for half in [first, second])
    name = f"TDE{frequency}hz" if refocused else f"TDE{frequency}hz_broken"
    if not refocused:
        print(f"WARNING: the halves of {lte_file} do not return q to 0 on their own, so the "
              f"axes overlap in q and the b-tensor of this TDE is NOT isotropic (not an STE). "
              f"Writing it as {name} to show the problem.")

    n = len(first)
    tde = np.zeros((lead + 3 * n + gap + 3 * len(second) + tail, 3))
    second_start = lead + 3 * n + gap
    for axis in range(3):
        tde[lead + axis * n: lead + (axis + 1) * n, axis] = first
        tde[second_start + axis * len(second): second_start + (axis + 1) * len(second), axis] = second

    np.savetxt(f"waveforms/{name}.csv", tde, delimiter=",", fmt="%.15g")

    print(f"{name}: from {lte_file}, {len(tde)} rows (LTE {len(g)}); "
          f"halves {n} rows, gap {gap}, lead {lead}, tail {tail}. "
          f"Wrote waveforms/{name}.csv")

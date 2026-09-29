"""
Waveform visualization
Plots gradient waveform files (N x 3 CSV, x,y,z per row, as used by
Simulation.py) to check them before using them in simulations.

For each file, one figure with:
    - Gradient g(t) per axis
    - Dephasing q(t) = gamma * integral of g dt, per axis (should end at 0 if
      the waveform is refocused; files already include the 180° pulse)
    - Encoding power spectrum |Q(f)|^2 per axis, with centroid frequencies
      per axis (dashed) and combined over the three axes (solid)
    - Eigenvalues of the b-tensor B = integral of q q^T dt, normalized by b
      (LTE ~ 1, 0, 0; isotropic STE ~ 1/3, 1/3, 1/3)
and a printed summary per file.

Usage (from the project root):
    pixi run -e dipy-env python src/plot_waveforms.py waveforms/<file>.csv [more files...]

Figures are saved to graphOutputs/waveforms/<file>.png by default.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from waveform_utils import AXES, analyze_waveform

parser = argparse.ArgumentParser(description="Plot gradient waveform files.")
parser.add_argument("files", nargs="+", help="Waveform CSV files (N x 3)")
parser.add_argument("--raster-time-ms", type=float, default=0.02,
                    help="Time between rows in ms (default 0.02, as in the simulation config)")
parser.add_argument("--gradient-scale", type=float, default=1e-3,
                    help="Factor converting file values to T/m (default 1e-3, i.e. files in mT/m)")
parser.add_argument("--fmax", type=float, default=500,
                    help="Maximum frequency shown in the spectrum, in Hz (default 500)")
parser.add_argument("--output-dir", default="graphOutputs/waveforms",
                    help="Folder for the figures (default graphOutputs/waveforms)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

for file in args.files:
    w = analyze_waveform(file, args.raster_time_ms, args.gradient_scale)
    name, t, g, q, b = w["name"], w["t"], w["g"], w["q"], w["b"]
    freqs, centroids, centroid_combined = w["freqs"], w["centroids"], w["centroid_combined"]
    power = w["power"] / w["power"].max()

    print(f"\n{name}")
    print(f"  Duration:            {len(g) * args.raster_time_ms:.2f} ms ({len(g)} rows)")
    print(f"  Max |g| per axis:    {', '.join(f'{v:.1f}' for v in np.abs(w['data']).max(axis=0))} (file units)")
    print(f"  b-value:             {b:.1f} s/mm^2 (at the file amplitude)")
    print(f"  B eigenvalues / b:   {', '.join(f'{v:.3f}' for v in w['eigenvalues'] / b)}")
    print(f"  |q(end)| / max |q|:  {w['refocusing']:.2e} (0 = refocused)")
    print(f"  Centroid frequency:  "
          + ", ".join(f"{axis} {c:.1f}" for axis, c in zip(AXES, centroids) if c is not None)
          + f", combined {centroid_combined:.1f} Hz")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    for i, axis in enumerate(AXES):
        axs[0, 0].plot(t, g[:, i] * 1e3, label=axis)
        axs[0, 1].plot(t, q[:, i] * 1e-6, label=axis)
        line, = axs[1, 0].plot(freqs, power[:, i], label=axis)
        if centroids[i] is not None:
            axs[1, 0].axvline(centroids[i], color=line.get_color(), linestyle="--",
                              label=f"{axis} centroid: {centroids[i]:.1f} Hz")

    axs[1, 0].axvline(centroid_combined, color="black",
                      label=f"combined centroid: {centroid_combined:.1f} Hz")

    axs[0, 0].set(title="Gradient", xlabel="Time (ms)", ylabel="g (mT/m)")
    axs[0, 1].set(title="Dephasing", xlabel="Time (ms)", ylabel="q (rad/µm)")
    axs[1, 0].set(title="Encoding spectrum (dephasing)", xlabel="Frequency (Hz)",
                  ylabel="|Q(f)|² (normalized)", xlim=(0, args.fmax))

    axs[1, 1].bar(["λ1", "λ2", "λ3"], w["eigenvalues"] / b, color="gray")
    axs[1, 1].set(title=f"b-tensor eigenvalues (b = {b:.0f} s/mm²)", ylabel="λ / b", ylim=(0, 1))

    for ax in axs.flat[:3]:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(name)
    plt.tight_layout()
    output = f"{args.output_dir}/{name}.png"
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"  Saved {output}")

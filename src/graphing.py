"""
Signal and DKI Analysis
Fits dMRI signals for MD, kurtosis, and variance,
performs DKI fitting for FA, MD, AD, and RD, and generates plots/results.

Usage:
    python src/graphing.py <signals.csv>
"""

#csv, DKI metrics, sphere sqr root Variance, multiple different packing and radii for cyl


import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from dipy.core.gradients import gradient_table
from dipy.reconst import dki
from pathlib import Path

#Parse input signal file
parser = argparse.ArgumentParser()
parser.add_argument("signals", type=str)
args = parser.parse_args()

signal_file = args.signals
graph_title = os.path.basename(signal_file)

#Create output dir
os.makedirs(f'graphOutputs/{graph_title}',exist_ok=True)
output = f"graphOutputs/{graph_title}"

#Load signal data
df = pd.read_csv(f"{signal_file}", comment="#")

#Average signal across walkers
df_averaged = df.groupby(['waveform_idx', 'bval'])['signal'].mean()

df_averaged.reset_index().to_csv(output+ '/' + 'poweder_average_signal.csv', index=False)

unique_waveforms = sorted(df['waveform_idx'].unique())

#Store fit params
d = []
md = []
kurt = []
curvesA = []
curvesB = []
curvesC = []
variance = []

x_fit = np.linspace(0, max(df['bval'])/1000, 100)
fig, ax = plt.subplots(figsize=(9,5))

#Fit signal decay for each waveform
for wf in unique_waveforms:
    wf_data = df_averaged.loc[wf]

    b_values = wf_data.index.tolist()
    signals = wf_data.values.tolist()

    b_arr = np.array(b_values)/1000
    signals_arr = np.array(signals)

    #Limit fit range
    mask = b_arr <= 10
    b_mask = b_arr[mask]
    signals_mask = signals_arr[mask]

    #Fit log(signal) to quadratic
    A, B, C = np.polyfit(b_mask, np.log(signals_mask), 2)
    if abs(A) < 1e-9:
        A = 0

    D = -B
    variance.append(A*2)
    kurt.append((6 * A) / (D**2))
    d.append(D)

    curvesA.append(A)
    curvesB.append(B)
    curvesC.append(C)

    #Get waveform name
    raw_path = df[df['waveform_idx'] == wf]['file'].iloc[0]
    clean_filename = os.path.basename(raw_path)
    label_name = clean_filename.replace(".csv", "")

    #Generate fitted signal
    y_fit = np.exp(A * (x_fit**2) + B * x_fit + C)

    ax.scatter(b_arr, signals, marker='o', label=rf"$\bf{{{label_name}\ Data}}$")
    ax.plot(x_fit, y_fit, linestyle='--', label=f"D: {d[-1]:.4e} µm²/ms\nK: {kurt[-1]:.4f}\nV: {variance[-1]:.4e} µm⁴/ms²")

#Plot signal decay
ax.set_yscale('log')
ax.set_xlabel("b-value (ms/µm²)")
ax.set_ylabel("Normalized Signal ($S/S_0$)")
ax.set_title(f"Signal Decay")
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

plt.subplots_adjust(right=0.7)
plt.savefig(f"{output}/signal_{graph_title}.svg", dpi=300, bbox_inches='tight')
plt.close(fig)
#Store freq dependent params
freq = np.array([0, 50, 100])
diffArr = np.array([d[0], d[1], d[2]])
kurtArr = np.array([kurt[0], kurt[1], kurt[2]])
varianceArr = np.array([variance[0], variance[1], variance[2]])

print([d[0], d[1], d[2]])

paramList = [("Diffusivity", diffArr), ("Kurtosis", kurtArr), ("Variance", varianceArr)]

#Plot frequency dependence
for name, param in paramList:

    fig, ax = plt.subplots()
    ax.scatter(freq, param, color='red', marker='o', s=10, zorder=5, label=f'{name} Data Points')

    x1 = np.linspace(0, 200, 100)

    #Linear fit
    lin_params = np.polyfit(freq, param, 1)
    lin_fit_vals = lin_params[0] * freq + lin_params[1]
    lin_error = np.sum((param - lin_fit_vals) ** 2)
    y_linear = lin_params[0] * x1 + lin_params[1]

    #Square root fit
    sqrt_params = np.polyfit(np.sqrt(freq), param, 1)
    sqrt_fit_vals = sqrt_params[0] * np.sqrt(freq) + sqrt_params[1]
    sqrt_error = np.sum((param - sqrt_fit_vals) ** 2)
    y_sqrt = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]

    #Squared fit
    sq_params = np.polyfit(freq**2, param, 1)
    sq_fit_vals = sq_params[0] * freq**2 + sq_params[1]
    sq_error = np.sum((param - sq_fit_vals) ** 2)
    y_squared = sq_params[0] * (x1**2) + sq_params[1]

    #Select best fit
    errors = {"Square Root": sqrt_error, "Linear": lin_error, "Squared": sq_error}
    best_fit_name = min(errors, key=errors.get)
    print(f"{name} Best Fit: {best_fit_name}")

    ls_squared = '-' if best_fit_name == "Squared" else ':'
    ls_linear = '-' if best_fit_name == "Linear" else ':'
    ls_sqrt = '-' if best_fit_name == "Square Root" else ':'

    ax.plot(x1, y_squared, color='#004949', linestyle=ls_squared, linewidth=2, label=f'Squared Fit {"(Best)" if best_fit_name == "Squared" else ""}')
    ax.plot(x1, y_linear, color='#FF6B6B', linestyle=ls_linear, linewidth=2, label=f'Linear Fit {"(Best)" if best_fit_name == "Linear" else ""}')
    ax.plot(x1, y_sqrt, color='#009999', linestyle=ls_sqrt, linewidth=2, label=f'Square Root Fit {"(Best)" if best_fit_name == "Square Root" else ""}')

    #Set y-axis units
    if name == "Diffusivity":
        ax.set_ylabel("Diffusivity (µm²/ms)")
    elif name == "Kurtosis":
        ax.set_ylabel("Kurtosis")
    elif name == "Variance":
        ax.set_ylabel("Variance (µm⁴/ms²)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"Frequency Dependence of {name}")
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{output}/{name}_{graph_title}.svg", dpi=300)
    plt.close(fig)

#Plot walker trajectories if available
try:
    traj_file = f"outputs/traj_{graph_title}"

    print(traj_file)

    data = np.loadtxt(traj_file)

    n_steps = data.shape[0]
    n_walkers = data.shape[1] // 3

    traj = data.reshape(n_steps, n_walkers, 3).transpose(1, 0, 2)

    print("Trajectory shape:", traj.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    #Plot first 10 walkers
    for i in range(10):
        ax.plot(traj[i, :, 0],traj[i, :, 1],traj[i, :, 2],alpha=0.7,linewidth=0.3)

    ax.view_init(elev=0, azim=0)
    ax.set_title("Disimpy Trajectories")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')

    plt.savefig(f"{output}/traj_{graph_title}.png", dpi=300)
    plt.close()

except:
    print("no traj file found")

#Load data for DKI fit
csv_file = args.signals
df = pd.read_csv(csv_file, comment="#")

R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
ref_dir = np.array([1, 0, 0])

count = 0
md = []
ad = []
rd = []
fa = []

#Perform DKI fit for each waveform
for waveform_file, df_wave in df.groupby("file", sort=False):
    if count <= 2:
        print(f"\nWaveform: {waveform_file}")

        bvals = df_wave["bval"].to_numpy()
        signals = df_wave["signal"].to_numpy()

        #Extract gradient directions
        rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
        bvecs = rot_matrices[:,0,:]

        #Run DKI fit
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        fit = dki.DiffusionKurtosisModel(gtab).fit(signals)

        print(f"Fractional Anisotropy (FA): {fit.fa:.3f}")
        fa.append(fit.fa)
        print(f"Mean Diffusivity (MD):     {fit.md:.3e} mm^2/s")
        md.append(fit.md*1000)
        print(f"Axial Diffusivity (AD):    {fit.ad:.3e} mm^2/s")
        ad.append(fit.ad*1000)
        print(f"Radial Diffusivity (RD):   {fit.rd:.3e} mm^2/s")
        rd.append(fit.rd*1000)

        count += 1

#Plot DKI metrics vs frequency
freq = np.array([0, 50, 100])
x1 = np.linspace(0, 200, 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

metrics = [
    ("Mean Diffusivity (MD)", np.array(md), axes[0]),
    ("Axial Diffusivity (AD)", np.array(ad), axes[1]),
    ("Radial Diffusivity (RD)", np.array(rd), axes[2]),
]

for metric_name, diff, ax in metrics:

    #Linear fit
    lin_params = np.polyfit(freq, diff, 1)
    lin_fit_vals = lin_params[0] * freq + lin_params[1]
    lin_error = np.sum((diff - lin_fit_vals) ** 2)
    y_linear = lin_params[0] * x1 + lin_params[1]

    #Square root fit
    sqrt_params = np.polyfit(np.sqrt(freq), diff, 1)
    sqrt_fit_vals = sqrt_params[0] * np.sqrt(freq) + sqrt_params[1]
    sqrt_error = np.sum((diff - sqrt_fit_vals) ** 2)
    y_sqrt = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]

    #Squared fit
    sq_params = np.polyfit(freq**2, diff, 1)
    sq_fit_vals = sq_params[0] * (freq**2) + sq_params[1]
    sq_error = np.sum((diff - sq_fit_vals) ** 2)
    y_squared = sq_params[0] * (x1**2) + sq_params[1]

    #Select best fit
    errors = {
        "Square Root": sqrt_error,
        "Linear": lin_error,
        "Squared": sq_error,
    }
    best_fit_name = min(errors, key=errors.get)

    ls_squared = "-" if best_fit_name == "Squared" else ":"
    ls_linear = "-" if best_fit_name == "Linear" else ":"
    ls_sqrt = "-" if best_fit_name == "Square Root" else ":"

    ax.scatter(freq,diff,color="red",marker="o",s=30,zorder=5,label="Data Points",)
    ax.plot(x1,y_squared,color="#004949",linestyle=ls_squared,linewidth=2,label=f'Squared {"(Best)" if best_fit_name == "Squared" else ""}',)
    ax.plot(x1,y_linear,color="#FF6B6B",linestyle=ls_linear,linewidth=2,label=f'Linear {"(Best)" if best_fit_name == "Linear" else ""}',)
    ax.plot(x1,y_sqrt,color="#009999",linestyle=ls_sqrt,linewidth=2,label=f'Square Root {"(Best)" if best_fit_name == "Square Root" else ""}',)

    #Display best-fit equation and values
    if best_fit_name == "Linear":
        eq = f"y = {lin_params[0]:.3e}x {lin_params[1]:+.3e}"
    elif best_fit_name == "Square Root":
        eq = f"y = {sqrt_params[0]:.3e}x^1/2 {sqrt_params[1]:+.3e}"
    else:
        eq = f"y = {sq_params[0]:.3e}x^2 {sq_params[1]:+.3e}"

    eq_text = (
        f"Best Fit: {best_fit_name}\n"
        f"{eq}\n\n"
        f"0 Hz   = {diff[0]:.3e}\n"
        f"50 Hz  = {diff[1]:.3e}\n"
        f"100 Hz = {diff[2]:.3e}"
    )

    ax.text(0.02,0.98,eq_text,transform=ax.transAxes,fontsize=8,verticalalignment="top",bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"))

    #Fix AD y-axis range

    ax.set_ylim(min(ad)*0.5,max(ad)*1.50) if metric_name == "Axial Diffusivity (AD)" else None

    ax.set_ylabel(f"{metric_name} ($\\mathrm{{µm^2/ms}}$)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"{metric_name} vs Frequency")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()

output_path = f"graphOutputs/{Path(csv_file).name}/MD_AD_RD.png"
plt.savefig(output_path, dpi=300)

#Export results
print(len(variance),len(kurt))

waveform = ["LTE-0Hz","LTE-50Hz","LTE-100Hz","STE-Iso","STE-Aniso"]

with open(f"{output}/results.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["Waveform", "FA", "MD", "AD", "RD","D", "Kurtosis", "Variance"])

    for i in range(3):
        writer.writerow([waveform[i], fa[i], md[i], ad[i], rd[i], d[i], kurt[i], variance[i]])

    for j in range(3, 5):
        writer.writerow([waveform[j], "", "", "", "", d[j], kurt[j], variance[j]])

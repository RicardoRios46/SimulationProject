"""
Walker trajectory plot
Plots the walker trajectories saved by Simulation.py ([trajectory] enabled in
the config). The trajectory file is found next to the signal file:
outputs/<config>/<signal>.csv -> outputs/<config>/<signal>_traj.csv

Output: graphOutputs/<signal>/traj_<signal>.png

Usage (from the project root):
    pixi run -e dipy-env python src/plot_trajectories.py outputs/<config>/<signal>.csv
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import output_dir

parser = argparse.ArgumentParser(description="Plot the walker trajectories of a simulation.")
parser.add_argument("signals", help="Signal CSV written by Simulation.py")
args = parser.parse_args()

traj_file = f"{os.path.splitext(args.signals)[0]}_traj.csv"
if not os.path.exists(traj_file):
    raise SystemExit(f"No trajectory file found: {traj_file}")

name, output = output_dir(args.signals)
print(f"Trajectory file: {traj_file}")

data = np.loadtxt(traj_file)

n_steps = data.shape[0]
n_walkers = data.shape[1] // 3

traj = data.reshape(n_steps, n_walkers, 3).transpose(1, 0, 2)

print("Trajectory shape:", traj.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

#Plot up to the first 10 walkers
for i in range(min(10, n_walkers)):
    ax.plot(traj[i, :, 0], traj[i, :, 1], traj[i, :, 2], alpha=0.7, linewidth=0.3)

ax.view_init(elev=0, azim=0)
ax.set_title("Disimpy Trajectories")
ax.set_ylabel("Y")
ax.set_xlabel("X")
ax.set_zlabel("Z")
ax.set_aspect('equal')

plt.savefig(f"{output}/traj_{name}.png", dpi=300)
plt.close()
print(f"Saved {output}/traj_{name}.png")

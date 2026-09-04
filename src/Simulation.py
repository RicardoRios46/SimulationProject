"""
Monte Carlo Simulation
Runs diffusion simulations using a configured substrate and waveforms,
generates signals, and saves simulation data and trajectories.
Local usage:
    python src/simulation.py <config.toml>
SLURM usage:
    Edit sim_config files on slurm repo to desired substrate name, then batch via: sbatch batch/{init_pos}_{rotation}.sh
    NOTE: if doing extreme large sim (ex: 1 million walkers 100k time step) increase memory in batch file
    to accommodate.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from disimpy import gradients, simulations, substrates
import pandas as pd
import tomli
import json
import os
import argparse

#Generate random seed for simulation
seed = np.random.randint(0, 2**32-1)
print(seed)

#Parse config file argument
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

config_file_path = args.config

config_name = os.path.basename(config_file_path)
config_name,_ = os.path.splitext(config_name)

#Load simulation config
with open(config_file_path, "rb") as f:
    config = tomli.load(f)

#Params
meshName = config["substrate"]["name"]
n_walkers = config["simulation"]["n_walkers"]
n_t = config["simulation"]["n_t"]
periodic = config["substrate"]["periodic"]
diffusivity = config["simulation"]["diffusivity"]
waveforms = config["waveform"]["waveform_file"]
directionFile = config["waveform"]["direction_file"]
b_targets = config["waveform"]["b_targets"]
position = config["substrate"]["position"]

#Load rotation matrix (applied to all waveforms)
rotations = np.loadtxt(f"rotations/{directionFile}", comments="#")
rot_matrix = rotations.reshape(-1, 3, 3)

#Generate unique output filename
def get_unique_filepath(filepath):
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{ext}"

    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{ext}"

    return new_filepath

#Load substrate mesh
def get_substrate(meshName):

    print(meshName)
    data_verts = pd.read_csv(f'substrate/{meshName}/{meshName}_vertices.csv')
    data_faces = pd.read_csv(f'substrate/{meshName}/{meshName}_faces.csv')

    vertices = data_verts.to_numpy()
    faces = data_faces.to_numpy()

    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=periodic,
        init_pos=position
    )

    return substrate

#Read gradient waveform from CSV
def read_shape(filename):
    """ 
    Takes x,y,z CSV vaules from MATLAB. 
    Returns x,y,z list of gradient values.
    """
    x_grad = []
    y_grad = []
    z_grad = []

    with open(filename) as f:
        for line in f:
            vals = line.strip().split(',')
            if len(vals)>1:
                x_grad.append(float(vals[0]))
                y_grad.append(float(vals[1]))
                z_grad.append(float(vals[2]))
            else:
                x_grad.append(float(vals[0]))
                y_grad.append(0)
                z_grad.append(0)

    return x_grad,y_grad,z_grad

#Load substrate
substrate = get_substrate(meshName)

#Create unique signal output file
csv_filename = get_unique_filepath(
    f"outputs/{meshName}_signals_{config_name}.csv"
)

#Create unique metadata output file (kept separate from the CSV, since CSV has no standard comment syntax)
meta_filename = get_unique_filepath(
    f"outputs/{meshName}_signals_{config_name}_metadata.json"
)

metadata_dict = {
    "config_file": config_name,
    "mc_seed": int(seed),
    "substrate": meshName,
    "n_walkers": n_walkers,
    "n_t": n_t,
    "position": position,
    "diffusivity": diffusivity,
    "b_targets": b_targets,
}

with open(meta_filename, mode="w") as meta_f:
    json.dump(metadata_dict, meta_f, indent=4)

csv_columns = ["file", "waveform_idx", "R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33", "bval", "signal"]

shape_signals = []
mega_gradient = []
metadata = []

#Loop through gradient waveforms
for filecount, file in enumerate(waveforms):

    #Load gradient waveform
    x_grad, y_grad, z_grad = read_shape(file)

    time = len(x_grad)*0.02
    time_points = np.arange(0,time,0.02)

    #Create gradient array
    gradient = np.zeros([1,len(time_points),3])

    gradient[0,:,0] = x_grad
    gradient[0,:,1] = y_grad
    gradient[0,:,2] = z_grad

    gradient *= 1e-3

    #Calculate base b-value
    print(f"Bval: {(gradients.calc_b(gradient,0.02e-3)*1e-6)[0]:.0f}")

    #Rotate gradient into all directions
    gradient_final = np.zeros([len(rot_matrix), len(time_points), 3])

    for i in range(0, len(rot_matrix)):
        rot_waveform = gradient @ rot_matrix[i].T
        gradient_final[i, : , : ] = rot_waveform

    #Interpolate gradient to simulation timestep
    gradient_final, dt = gradients.interpolate_gradient(gradient_final, 0.02e-3, int(n_t))

    #Calculate base b-value and target b-values
    b_base = (gradients.calc_b(gradient_final, dt) * 1e-6)

    #Scale gradients to achieve target b-values
    for j, b in enumerate(b_targets):
        if b == 0:
            for i in range(len(rot_matrix)):
                metadata.append([file,filecount + 1,*rot_matrix[i].flatten(),b])
            continue

        scale = np.sqrt(b / b_base[0])
        scaled_gradient = gradient_final * scale
        b_vals = gradients.calc_b(scaled_gradient, dt) * 1e-6

        mega_gradient.append(scaled_gradient)

        #Store waveform metadata
        for i in range(len(rot_matrix)):
            metadata.append([file,filecount + 1,*rot_matrix[i].flatten(),b])

#Combine all gradients
mega_gradient = np.concatenate(mega_gradient, axis=0)

print("Mega gradient shape:", mega_gradient.shape)
print("Metadata entries:", len(metadata))
print(f"\n\nRunning mega simulation.")

#Run sim
signal = simulations.simulation(
    n_walkers=int(n_walkers),
    diffusivity=diffusivity,
    gradient=mega_gradient,
    dt=dt,
    substrate=substrate,
    seed=seed
)

#Normalize signal by walker count
norm_signal = abs(signal / n_walkers)

signal_idx = 0

#Assemble signal rows (b=0 rows get signal=1.0, others pull from norm_signal in order)
csv_rows = []
for row in metadata:
    if row[-1] == 0:
        csv_rows.append([*row, 1.0])
    else:
        csv_rows.append([*row, norm_signal[signal_idx]])
        signal_idx += 1

#Write signals to CSV via pandas
signal_df = pd.DataFrame(csv_rows, columns=csv_columns)
signal_df.to_csv(csv_filename, index=False)

#Run trajectory sim
traj_file = get_unique_filepath(
    f"outputs/traj_{meshName}_signals_{config_name}.csv"
)

trajSignal = simulations.simulation(
    n_walkers=10,
    diffusivity=diffusivity,
    gradient=mega_gradient,
    dt=dt,
    substrate=substrate,
    seed=seed,
    traj=traj_file
)

#Prit sim info
print(f"# MC seed: {seed}\n")
print(f"# Subtrate: {meshName}")
print(f"# Walkers: {n_walkers}. Steps: {n_t}")
print(f"# Position: {position}")
print(f"# Diffusivity: {diffusivity}\n")
print(f"Writing outputs to: {csv_filename}")
print(f"Writing metadata to: {meta_filename}")
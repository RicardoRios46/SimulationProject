#!/bin/bash
#SBATCH --job-name=MCsim
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=hx
#SBATCH --output=/nfs/baron/rrioscar/SimulationProject/slurm_outputs/slurm-%A_%a.out

config="config.toml" 

# Change to working directory
cd /nfs/baron/rrioscar/SimulationProject/ || exit 1

# Run computation for this task's assigned config file
pixi run -e disimpy-env python src/Simulation.py "sim_configs/${config}"

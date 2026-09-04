#!/bin/bash
#SBATCH --job-name=MCsim
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=hx
#SBATCH --output=/nfs/baron/rrioscar/SimulationProject/slurm_outputs/slurm-%A_%a.out


#SBATCH --array=0-1
configs=(
 "config1.toml" 
 "config2.toml" 
)
# NOTE change SBATCH to match thhe size of the list of configs above. For example, if you have 4 configs, use --array=0-3

# Select config based on the current SLURM task ID
config=${configs[$SLURM_ARRAY_TASK_ID]}

# Change to working directory
cd /nfs/baron/rrioscar/SimulationProject/  || exit 1

# Run computation for this task's assigned config file
pixi run -e disimpy-env python src/Simulation.py "sim_configs/${config}"

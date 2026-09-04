#!/bin/bash
#SBATCH --job-name=ext_eul
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1 
#SBATCH --mem=64G
#SBATCH --output=/nfs/scratch/choover/SimulationProject2/slurm_outputs/slurm-%j.out
#SBATCH --partition=hx

# Run computation
cd /nfs/scratch/choover/SimulationProject2/

pixi run python src/Simulation.py sim_configs/extra_euler_config.toml



#!/bin/bash
#SBATCH --job-name=MCsim
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=hx
#SBATCH --output=slurm_outputs/slurm-%j.out

# Run a single simulation config on the cluster.
#
# Usage (always submit from the project root):
#   sbatch batch/sbatch.sh sim_configs/<config>.toml
#
# The #SBATCH values above are defaults. Override them on the command line
# instead of editing this file, e.g.:
#   sbatch -p vx --mem=64G --time=24:00:00 batch/sbatch.sh sim_configs/<config>.toml
#
# Partitions: hx (default, more GPUs available) or vx.
# For very large simulations (e.g. 1 million walkers, 100k time steps),
# increase --mem.

config="$1"

# The job starts in the directory sbatch was called from
cd "$SLURM_SUBMIT_DIR" || exit 1

if [ ! -f pixi.toml ]; then
    echo "Error: submit this job from the project root (pixi.toml not found in $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi

if [ -z "$config" ]; then
    echo "Error: no config given. Usage: sbatch batch/sbatch.sh sim_configs/<config>.toml" >&2
    exit 1
fi

if [ ! -f "$config" ]; then
    echo "Error: config file '$config' not found" >&2
    exit 1
fi

echo "Job $SLURM_JOB_ID on $(hostname), config: $config"

pixi run -e disimpy-env python src/Simulation.py "$config"

#!/bin/bash
#SBATCH --job-name=MCsim
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=hx
#SBATCH --output=slurm_outputs/slurm-%A_%a.out

# Run several simulation configs as a SLURM job array, one config per task.
#
# The configs are read from a list file: one config path per line (relative
# to the project root). Blank lines and lines starting with # are ignored.
#
# Recommended usage (from the project root). The wrapper checks the list and
# sets the array size automatically:
#   bash batch/submit_array.sh sim_configs/<list>.txt
#
# Direct usage: --array must be 0-(N-1) for N configs in the list:
#   sbatch --array=0-<N-1> batch/array_sbatch.sh sim_configs/<list>.txt
#
# The #SBATCH values above are defaults. Override them on the command line,
# e.g. -p vx --mem=64G. Partitions: hx (default, more GPUs available) or vx.

list="$1"

# The job starts in the directory sbatch was called from
cd "$SLURM_SUBMIT_DIR" || exit 1

if [ ! -f pixi.toml ]; then
    echo "Error: submit this job from the project root (pixi.toml not found in $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: this script must be submitted as a job array. Use: bash batch/submit_array.sh <list>.txt" >&2
    exit 1
fi

if [ -z "$list" ] || [ ! -f "$list" ]; then
    echo "Error: config list file '$list' not found" >&2
    exit 1
fi

# Read configs, skipping comments and blank lines
mapfile -t configs < <(grep -vE '^[[:space:]]*(#|$)' "$list" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#configs[@]}" ]; then
    echo "Error: array task $SLURM_ARRAY_TASK_ID but '$list' only has ${#configs[@]} config(s)" >&2
    exit 1
fi

config="${configs[$SLURM_ARRAY_TASK_ID]}"

if [ ! -f "$config" ]; then
    echo "Error: config file '$config' not found" >&2
    exit 1
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname), config: $config"

pixi run -e disimpy-env python src/Simulation.py "$config"

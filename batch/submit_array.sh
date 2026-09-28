#!/bin/bash
# Submit a list of simulation configs as a SLURM job array (batch/array_sbatch.sh).
#
# Checks that every config in the list exists, then submits with the array
# size set to the number of configs.
#
# Usage (from the project root):
#   bash batch/submit_array.sh <list>.txt [--max-parallel K] [--dry-run] [sbatch options...]
#
# List file: one config path per line, relative to the project root.
# Blank lines and lines starting with # are ignored.
#
# Options:
#   --max-parallel K   Run at most K tasks at the same time (--array=0-N%K)
#   --dry-run          Check the list and print the sbatch command without submitting
#   anything else      Passed to sbatch, e.g. -p vx --mem=64G --time=24:00:00
#
# Examples:
#   bash batch/submit_array.sh sim_configs/sweep.txt
#   bash batch/submit_array.sh sim_configs/sweep.txt --max-parallel 4 -p vx

usage="Usage: bash batch/submit_array.sh <list>.txt [--max-parallel K] [--dry-run] [sbatch options...]"

if [ $# -lt 1 ]; then
    echo "$usage" >&2
    exit 1
fi

list="$1"
shift

max_parallel=""
dry_run=false
sbatch_opts=()

while [ $# -gt 0 ]; do
    case "$1" in
        --max-parallel)
            if [ $# -lt 2 ]; then
                echo "Error: --max-parallel needs a value" >&2
                exit 1
            fi
            max_parallel="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        *)
            sbatch_opts+=("$1")
            shift
            ;;
    esac
done

if [ ! -f pixi.toml ]; then
    echo "Error: run this from the project root (pixi.toml not found in $PWD)" >&2
    exit 1
fi

if [ ! -f "$list" ]; then
    echo "Error: config list file '$list' not found" >&2
    exit 1
fi

if [ -n "$max_parallel" ] && ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-parallel must be a positive integer, got '$max_parallel'" >&2
    exit 1
fi

# Read configs the same way array_sbatch.sh does
mapfile -t configs < <(grep -vE '^[[:space:]]*(#|$)' "$list" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')

n=${#configs[@]}
if [ "$n" -eq 0 ]; then
    echo "Error: no configs found in '$list'" >&2
    exit 1
fi

# Check every config before submitting anything
missing=0
for i in "${!configs[@]}"; do
    if [ ! -f "${configs[$i]}" ]; then
        echo "Error: task $i config '${configs[$i]}' not found" >&2
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "$missing of $n config(s) missing, nothing submitted" >&2
    exit 1
fi

array="0-$((n - 1))"
if [ -n "$max_parallel" ]; then
    array="${array}%${max_parallel}"
fi

echo "Configs in '$list' ($n):"
for i in "${!configs[@]}"; do
    echo "  [$i] ${configs[$i]}"
done

cmd=(sbatch --array="$array" "${sbatch_opts[@]}" batch/array_sbatch.sh "$list")
echo "Running: ${cmd[*]}"

if [ "$dry_run" = true ]; then
    echo "Dry run, nothing submitted"
    exit 0
fi

"${cmd[@]}"

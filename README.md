# Monte Carlo Diffusion Simulation Pipeline

Pipeline for building diffusion substrates, running Monte Carlo diffusion MRI
simulations on them, and analyzing the resulting signals.

The project uses:

- [Disimpy](https://github.com/kerkelae/disimpy) for GPU Monte Carlo diffusion simulations
- [Trimesh](https://trimesh.org/) for building substrate meshes (including CATERPillar-generated substrates)
- [DIPY](https://dipy.org/) for DKI analysis
- [Pixi](https://pixi.sh/) for environment management
- SLURM for large scale simulations

## Project Structure

```text
.
├── batch/                  # SLURM batch scripts (single config / job array)
├── CATERPillar_inputs/     # Raw CATERPillar outputs used to build substrates
├── rotations/              # Rotation matrix sets applied to each waveform
├── sim_configs/            # Simulation configuration files (TOML) + templates
├── substrate_configs/      # Substrate generator configuration files (TOML) + templates
├── slurm_outputs/          # SLURM log files
├── src/
│   ├── Simulation.py       # Monte Carlo simulation (disimpy-env)
│   ├── graphing.py         # Signal and DKI analysis (dipy-env)
│   ├── plot_waveforms.py   # Waveform visualization and checks (dipy-env)
│   ├── compare_waveforms.py # Overlay several waveforms to compare them (dipy-env)
│   ├── waveform_utils.py   # Shared waveform calculations
│   └── substrate/          # Substrate generation scripts (trimesh-env)
│       └── archive/        # Unmaintained beaded axon scripts, kept for reference
├── waveforms/              # Gradient waveforms (N x 3 CSV)
├── pixi.toml               # Pixi environments and tasks
├── pixi.lock               # Locked package versions (commit with pixi.toml)
├── CLAUDE.md               # Notes for Claude Code sessions (conventions, project state)
├── LICENSE
└── README.md
```

The following directories are git-ignored and are created when the scripts run:

```text
substrate/<name>/           # Generated substrates (mesh CSVs, params JSON, PNG previews)
outputs/<config>/           # Simulation signals, metadata and trajectories
graphOutputs/<signal_file>/ # Analysis plots and results (signal file name without .csv)
graphOutputs/waveforms/     # Waveform plots and comparisons
```

All scripts use paths relative to the project root, so **always run commands
from the project root directory**.

## Installation

Install [Pixi](https://pixi.sh/) if needed:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

The environments are defined for Linux (`linux-64`) and Windows (`win-64`).
Running simulations requires an NVIDIA GPU with CUDA support (disimpy uses
CUDA through Numba).

Simulations are meant to be run on the Linux cluster. `disimpy-env` is pinned
to older packages (Python 3.9, `numpy<2`, CUDA 11.8) because that is the
configuration disimpy currently runs with, and these versions do not support
recent GPUs (e.g. RTX 50-series). On Windows, use the environments for
creating substrates and running the analysis.

## Environments

The project defines three separate environments, one per stage of the pipeline:

| Environment             | Python | Used for                     | Main packages                                   |
|-------------------------|--------|------------------------------|-------------------------------------------------|
| `trimesh-env`           | 3.13   | Creating substrates          | trimesh, matplotlib, scipy, manifold3d          |
| `disimpy-env`           | 3.9    | Running MC simulations       | disimpy 0.3, cudatoolkit 11.8, tomli            |
| `dipy-env` (`default`)  | 3.13   | Signal/DKI analysis, plotting | dipy, matplotlib                                |

`numpy` and `pandas` are shared by all environments.

Run a command in a specific environment with:

```bash
pixi run -e <environment> python <script>.py
```

or open a shell inside it:

```bash
pixi shell -e <environment>
```

Note that a plain `pixi shell` opens the **default** (`dipy-env`) environment,
which does not include disimpy.

## Workflow Overview

```text
1. Create substrate      trimesh-env   src/substrate/*.py     -> substrate/<name>/
                                        (+ substrate_configs/<config>.toml)
2. Write config          -             sim_configs/<config>.toml
3. Run simulation        disimpy-env   src/Simulation.py      -> outputs/<config>/<config>.csv
4. Analyze signals       dipy-env      src/graphing.py        -> graphOutputs/<config>/
```

## 1. Substrates

### Format

The simulation loads a substrate by name from:

```text
substrate/<name>/<name>_vertices.csv   # columns: x, y, z (with header)
substrate/<name>/<name>_faces.csv      # triangle vertex indices (with header)
```

Vertex coordinates must be in **meters** (the generator scripts build meshes in
µm and scale them by `1e-6` before saving).

Any mesh can be used as long as it is converted to this vertex/face CSV format.

### Generating substrates

The scripts in `src/substrate/` build meshes with trimesh and write them to
`substrate/<name>/`. Run them from the project root.

| Script                          | Substrate                                                   | Parameters            |
|---------------------------------|-------------------------------------------------------------|-----------------------|
| `cylinders.py`                  | Randomly packed parallel cylinders, gamma or fixed radius   | Config file           |
| `spheres.py`                    | Randomly packed spheres, gamma or fixed radius              | Config file           |
| `single_spheres.py`             | One single-sphere substrate per radius                      | Config file           |
| `caterpillar.py`                | Converts a CATERPillar output file to a mesh (see below)    | Config file           |
| `view_substrate.py`             | Re-plots an existing substrate (see below)                  | Command-line options  |

The scripts marked "Config file" use the shared helpers in
`src/substrate/common.py` and read their parameters from a TOML config file
in `substrate_configs/`. Copy a template, edit it, and pass it to the script:

```bash
cp substrate_configs/cylinders_template.toml substrate_configs/my_cylinders.toml
pixi run -e trimesh-env python src/substrate/cylinders.py substrate_configs/my_cylinders.toml
```

Each script has a template, `substrate_configs/<script>_template.toml`. Like
`sim_configs/`, only the templates are version-controlled. Your own configs
stay local, and the parameters of each substrate are saved in its
`<name>_params.json`. Unknown keys (e.g. a typo) and missing required keys are
reported as errors before anything is generated.

`src/substrate/archive/` keeps older generators that are not currently used
or maintained, for reference: the beaded axon scripts (`single_axon.py`,
`substrate_beaded_axon_SA.py`, `substrate_beaded_axon_V.py`). They take their
parameters from variables at the top of each script, do not use the config
files or `common.py`, and may need adjustments to work.

#### Random packing: `cylinders.py`, `spheres.py`

Config parameters (all lengths in µm):

| Parameter                  | Description                                                          |
|----------------------------|----------------------------------------------------------------------|
| `name`                     | *Optional.* Substrate name. If omitted, it is built from the parameters (see below) |
| `seed`                     | *Optional.* Random seed. If omitted, a random seed is generated. The same parameters and seed give the same substrate |
| `n_objects`                | Number of cylinders/spheres to place                                 |
| `domain_size`              | Side of the square (cylinders, xy) or cube (spheres) objects are centered in |
| `min_gap`                  | Minimum gap between object surfaces                                  |
| `radius_distribution`      | `"gamma"` or `"fixed"`                                               |
| `gamma_shape`, `gamma_scale` | Gamma distribution of the radii                                    |
| `radius_min`, `radius_max` | Gamma radii outside this range are redrawn (truncated distribution)  |
| `radius_fixed`             | Radius of every object when `radius_distribution = "fixed"`          |
| `max_attempts`             | *Optional.* Random positions tried per object before it is skipped (default 2000) |
| `max_consecutive_failures` | *Optional.* Stop after this many objects in a row could not be placed (default 100) |
| `cylinder_length`          | Cylinder length along z (`cylinders.py` only)                        |

Objects are placed one at a time at random non-overlapping positions. When
the domain fills up, some objects cannot be placed and are skipped, so fewer
than `n_objects` may be placed. Placement can also be stopped with Ctrl+C,
and the objects placed so far are saved.

When `name` is omitted, the name is built from the parameters and the number
of objects actually placed, with `p` as decimal point, e.g.
`cylinders_2840_gamma_shape0p75_scale0p55_gap0p45` or
`spheres_500_fixed_r5_gap1`.

#### Single spheres: `single_spheres.py`

Creates one substrate with a single sphere for each radius, e.g. to compare
the simulation with the analytical signal of restricted diffusion in a sphere.

| Parameter     | Description                                                             |
|---------------|-------------------------------------------------------------------------|
| `radii`       | List of sphere radii (µm). One substrate is created per radius          |
| `name_prefix` | *Optional.* Substrate names are `<name_prefix>_r<radius>`, e.g. `sphere_r2p5` (default `"sphere"`) |

#### Outputs

Each substrate from a config-file script is saved to `substrate/<name>/`:

| File                        | Contents                                                          |
|-----------------------------|-------------------------------------------------------------------|
| `<name>_vertices.csv`, `<name>_faces.csv` | Mesh in meters (simulation input)                   |
| `<name>_params.json`        | Parameters used, achieved results and provenance (see below)      |
| `<name>_mesh.png`           | 3D preview (µm)                                                    |
| `<name>_radii.png`          | Histogram of the placed radii (random packing only)               |
| `<name>_cross_section.png`  | Top view of the cylinder packing (`cylinders.py` only)            |

`<name>_params.json` records everything needed to trace back or regenerate a
substrate:

- `params`: all parameters used, including defaults and the seed
- `results`: what was actually generated, and whether the mesh is watertight.
  For random packing: the number of objects placed, the volume fraction
  (spheres) or area fraction (cylinders), and the mean, std, min and max
  radius. For single spheres: the radius and the mesh volume compared with the
  ideal sphere volume
- `script`, `config_file`, `created` and `git_commit`: the generator script, the
  config file, the date, and the git commit of the project (marked `-dirty` if
  there were uncommitted changes)

### Viewing a substrate

`view_substrate.py` re-plots an existing substrate without regenerating it,
e.g. to try other viewing angles or to get an SVG for a figure:

```bash
pixi run -e trimesh-env python src/substrate/view_substrate.py <name> [--elev 20] [--azim 45] [--output fig.png]
```

| Option              | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `--output`          | Output file, the extension sets the format (default `substrate/<name>/<name>_view.svg`) |
| `--title`           | Figure title (default: the substrate name)                         |
| `--elev`, `--azim`  | Viewing angles in degrees                                          |
| `--no-equal-aspect` | Do not scale the axes equally (useful for long cylinders)          |

### CATERPillar substrates

CATERPillar outputs (e.g. `CATERPillar_inputs/single_axon_run3.csv`) describe
each cell as a chain of overlapping spheres, one per row, with the columns
`cell_type cell_id component component_id X Y Z inner_radius outer_radius`
(µm). `caterpillar.py` creates one sphere per row and merges the spheres of
each cell (`cell_type` + `cell_id`) into one surface, so every axon stays a
separate compartment. The cells are then combined into one substrate:

```bash
cp substrate_configs/caterpillar_template.toml substrate_configs/my_caterpillar.toml
pixi run -e trimesh-env python src/substrate/caterpillar.py substrate_configs/my_caterpillar.toml
```

| Parameter              | Description                                                             |
|------------------------|-------------------------------------------------------------------------|
| `input_file`           | CATERPillar output file, relative to the project root                   |
| `name`                 | *Optional.* Substrate name (default `caterpillar_<input file name>`)    |
| `radius_column`        | *Optional.* `"inner_radius"` (the axon itself, default) or `"outer_radius"` (including the myelin sheath) |
| `cell_types`           | *Optional.* Only use these cell types, e.g. `["axon"]` (default: all rows) |
| `sphere_subdivisions`  | *Optional.* Icosphere resolution of each sphere: 1 = 80 faces, 2 = 320 faces (default 1) |
| `smoothing_iterations` | *Optional.* Laplacian smoothing iterations after merging, 0 = off (default 1). The total volume is kept constant |

The results in `<name>_params.json` include the number of cells and spheres,
the number of separate volumes in the final mesh (normally one per cell), and
the mesh volume.

## 2. Simulation Configuration

Each simulation is defined by a TOML file in `sim_configs/`. Start from one of
the templates:

- `sim_configs/config_template.toml`: every option, with comments
- `sim_configs/config_template_minimal.toml`: the same options, without comments

**The config filename (without extension) is used as the base name for all
output files**, so give each config a descriptive name.

`sim_configs/` is git-ignored, so only the two templates are version-controlled.
Your own configs stay local, and each run's full config is saved in its
metadata JSON.

### Options

`[substrate]`

| Key        | Description                                                              |
|------------|--------------------------------------------------------------------------|
| `name`     | Substrate folder name in `substrate/`                                    |
| `periodic` | Treat the mesh as periodic (`true`/`false`)                              |
| `position` | Initial walker positions, passed to disimpy `init_pos` (e.g. `"intra"`, `"extra"`) |

`[simulation]`

| Key           | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `n_walkers`   | Number of random walkers                                                    |
| `n_t`         | Number of simulation time steps                                             |
| `diffusivity` | Diffusivity in m²/s (e.g. `2e-9`)                                           |
| `seed`        | *Optional.* Fixed Monte Carlo seed. If omitted, a random seed is generated. The seed is always saved in the metadata file |

`[waveform]`

| Key               | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `waveform_file`   | List of waveform files in `waveforms/`                                      |
| `rotation_file`   | Rotation file in `rotations/`. Either a single file applied to all waveforms, or a list with exactly one file per waveform (in the same order) |
| `b_targets`       | Target b-values in s/mm². Each waveform/rotation is scaled to each b-value. `0` is not simulated and is written with `signal = 1.0` |
| `raster_time_ms`  | *Optional.* Time step of the waveform files in ms. Default `0.02` (20 µs)   |
| `gradient_scale`  | *Optional.* Factor that converts waveform values to T/m. Default `1e-3` (waveforms in mT/m) |

`[trajectory]` *(optional section; disabled if omitted)*

| Key         | Description                                              |
|-------------|----------------------------------------------------------|
| `enabled`   | Run an additional small simulation that saves walker trajectories |
| `n_walkers` | Number of walkers in the trajectory simulation (default `10`) |

### Input file formats

**Waveforms** (`waveforms/*.csv`): an N x 3 CSV with no header and one row per
time step (`x,y,z` gradient amplitudes). Both LTE and STE waveforms use this
format, e.g. as exported from MATLAB. Rows are spaced by `raster_time_ms`.
Waveforms must already include the effect of the 180° refocusing pulse (sign
flip of the second half).

To check a waveform before simulating it, plot it with:

```bash
pixi run -e dipy-env python src/plot_waveforms.py waveforms/<file>.csv [more files...]
```

For each file this saves `graphOutputs/waveforms/<file>.png` with the gradient
g(t), the dephasing q(t), the encoding power spectrum |Q(f)|² per axis, and the
normalized b-tensor eigenvalues, and prints a summary: duration, maximum
gradient, b-value at the file amplitude, b-tensor eigenvalues, whether q
returns to 0 at the end (refocusing), and the centroid frequencies.

The centroid frequency is the power-weighted mean frequency of the dephasing
spectrum |Q(f)|², computed over the full spectrum. It is shown per axis
(dashed lines) and combined over the three axes (solid line), using their
total power. The combined value does not depend on the waveform orientation;
the per-axis values do. Note that the gradient spectrum |G(f)|² = (2πf)²|Q(f)|²
would give higher centroids.

To compare several waveforms, e.g. to check that the axes of an STE match its
LTE component files, overlay them with:

```bash
pixi run -e dipy-env python src/compare_waveforms.py waveforms/STEiso.csv \
    waveforms/STEiso_LTE1.csv waveforms/STEiso_LTE2.csv waveforms/STEiso_LTE3.csv \
    --name STEiso_components --sum-check
```

This saves `graphOutputs/waveforms/<name>.png` with one row per axis (only the
axes with gradient in some file) and the gradient, dephasing and encoding
spectrum (with centroid lines) of every file overlaid. The first file is drawn
as a thick transparent band, so files that match it show inside the band. A
summary table (duration, b-value, b-tensor eigenvalues, refocusing, centroid
frequencies) is printed and saved as `<name>.csv`.

| Option           | Description                                                        |
|------------------|--------------------------------------------------------------------|
| `--name`         | Output name (default `compare_<first file>`)                       |
| `--labels`       | Legend labels, one per file (default: file names)                  |
| `--shared-scale` | Normalize all spectra by one maximum to also compare encoding power (default: each curve normalized to its own maximum, to compare frequency content) |
| `--sum-check`    | Treat the first file as the reference and the others as its components (e.g. an STE and its LTE component files): draw the sum of the components' gradients as a dashed black line and print the maximum difference per axis. All files must have the same number of rows |

`--raster-time-ms`, `--gradient-scale`, `--fmax` and `--output-dir` work as in
`plot_waveforms.py`. Both scripts share their calculations in
`src/waveform_utils.py`. Options: `--raster-time-ms` and
`--gradient-scale` (same defaults as the simulation config), `--fmax` for the
spectrum range, and `--output-dir`.

**Rotations** (`rotations/*.txt`): one 3x3 rotation matrix per line, flattened
row-major into 9 space-separated values. Lines starting with `#` are comments.
Each waveform is rotated by every matrix in its rotation file.

## 3. Running a Simulation

### Local

```bash
pixi run -e disimpy-env python src/Simulation.py sim_configs/<config>.toml
```

All waveforms × rotations × non-zero b-values are combined into a single
gradient array and simulated in one disimpy run.

### SLURM

Two batch scripts are provided in `batch/`. **Always submit from the project
root**: jobs run in the directory `sbatch` was called from, and logs are
written to `slurm_outputs/`.

After updating the project on the cluster (e.g. `git pull`), update the
environment on the login node before submitting:

```bash
pixi install -e disimpy-env
```

Otherwise the first `pixi run` in each job installs the changes itself, and
the tasks of a job array that start at the same time try to update the same
environment at once and can fail (e.g. `Failed to update PyPI packages for
environment 'disimpy-env'`).

#### Single config: `sbatch.sh`

Pass the config file as an argument:

```bash
sbatch batch/sbatch.sh sim_configs/<config>.toml
```

The log is written to `slurm_outputs/slurm-<jobid>.out`.

The `#SBATCH` values in the script are defaults (partition `hx`, 1 GPU, 4 CPUs,
32 GB memory, 12 h). Override them on the command line instead of editing the
script:

```bash
sbatch -p vx --mem=64G --time=24:00:00 batch/sbatch.sh sim_configs/<config>.toml
```

The available partitions are `hx` (default, more GPUs available) and `vx`. For
very large simulations (e.g. 1 million walkers with 100k time steps), increase
`--mem`.

#### Multiple configs: `array_sbatch.sh`

Runs several configs as a job array, one config per task. Write a list file
with one config path per line (relative to the project root). Blank lines and
lines starting with `#` are ignored, so you can comment out configs to skip
them:

```text
# sim_configs/sweep_radius.txt
sim_configs/cyl_r0.5.toml
sim_configs/cyl_r1.0.toml
# sim_configs/cyl_r1.5.toml
sim_configs/cyl_r2.0.toml
```

Submit it with the wrapper `submit_array.sh`, which checks that every config in
the list exists and sets the array size automatically:

```bash
bash batch/submit_array.sh sim_configs/sweep_radius.txt
```

Wrapper options:

| Option             | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `--max-parallel K` | Run at most K tasks at the same time                          |
| `--dry-run`        | Check the list and print the `sbatch` command without submitting |
| anything else      | Passed to `sbatch`, e.g. `-p vx --mem=64G --time=24:00:00`    |

```bash
bash batch/submit_array.sh sim_configs/sweep_radius.txt --max-parallel 4 -p vx
```

Each task writes its log to `slurm_outputs/slurm-<arrayjobid>_<task>.out`,
where `<task>` is the config's position in the list (starting from 0).

`array_sbatch.sh` can also be submitted directly. In that case `--array` must
be `0-<N-1>` for N configs in the list:

```bash
sbatch --array=0-2 batch/array_sbatch.sh sim_configs/sweep_radius.txt
```

## Simulation Outputs

Outputs are written to a folder named after the config file,
`outputs/<config>/`, and the files are named after the config too. Existing
signal files are never overwritten: running the same config again writes to
the same folder with a numeric suffix (`_1`, `_2`, ...), and the metadata and
trajectory files of that run take the same name (`<signal>` below), so all
files from one run always share a prefix.

| File                               | Contents                                                        |
|------------------------------------|-----------------------------------------------------------------|
| `<config>.csv`                     | Signals, one row per waveform × rotation × b-value (`<config>_1.csv`, ... for later runs) |
| `<signal>_metadata.json`           | The full config plus the Monte Carlo seed used. Named after the signal file of the same run, e.g. `<config>_1.csv` → `<config>_1_metadata.json` |
| `<signal>_traj.csv`                | Walker trajectories (only if `[trajectory] enabled = true`). Named after the signal file of the same run, e.g. `<config>_1.csv` → `<config>_1_traj.csv` |

The signal CSV has the columns:

```text
file, waveform_idx, R11, R12, R13, R21, R22, R23, R31, R32, R33, bval, signal
```

where `R11..R33` is the rotation matrix applied, `bval` is in s/mm², and
`signal` is normalized by the number of walkers.

The metadata file is only written after the simulation (and trajectory
simulation, if enabled) completes successfully.

## 4. Signal & DKI Analysis

Analyze a signal file with:

```bash
pixi run -e dipy-env python src/graphing.py outputs/<config>/<signal_file>.csv
```

or with the equivalent Pixi task:

```bash
pixi run fitsGraph outputs/<config>/<signal_file>.csv
```

> **Current assumption:** `graphing.py` expects exactly five waveforms, in this
> order: LTE 0 Hz, LTE 50 Hz, LTE 100 Hz, STE isotropic, STE anisotropic. The
> frequency-dependence fits use the first three (0, 50, 100 Hz), and DKI is fit
> for the LTE waveforms only.

The analysis performs:

- Powder averaging of the signal over rotations
- 2nd order fit of log-signal decay: diffusivity, kurtosis and variance
- Frequency-dependence fits of D, K and V (linear, square root and squared
  models). The model with the lowest least-squares error is reported as the best fit
- DKI fit (DIPY) for FA, MD, AD and RD, and their frequency dependence
- Walker trajectory plot, when a trajectory file is available

### Analysis Outputs

Results are saved in `graphOutputs/<signal_file>/`:

| File                             | Contents                                          |
|----------------------------------|---------------------------------------------------|
| `poweder_average_signal.csv`     | Powder-averaged signal per waveform and b-value   |
| `signal_<signal_file>.svg`       | Signal decay and fits                             |
| `Diffusivity_/Kurtosis_/Variance_<signal_file>.svg` | Frequency dependence of D, K, V  |
| `MD_AD_RD.png`                   | Frequency dependence of DKI metrics               |
| `traj_<signal_file>.png`         | Walker trajectories (if available)                |
| `results.csv`                    | FA, MD, AD, RD, D, kurtosis and variance per waveform |

## Typical Workflow

```bash
# 1. Create a substrate from a config
cp substrate_configs/cylinders_template.toml substrate_configs/my_cylinders.toml
#    ...edit geometry and radius distribution...
pixi run -e trimesh-env python src/substrate/cylinders.py substrate_configs/my_cylinders.toml

# 2. Create a config from the template
cp sim_configs/config_template.toml sim_configs/my_run.toml
#    ...edit substrate name, waveforms, rotations, b-values...

# 3. Run the simulation (locally, or via sbatch on the cluster)
pixi run -e disimpy-env python src/Simulation.py sim_configs/my_run.toml

# 4. Analyze the signals
pixi run fitsGraph outputs/my_run/my_run.csv
```

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
├── slurm_outputs/          # SLURM log files
├── src/
│   ├── Simulation.py       # Monte Carlo simulation (disimpy-env)
│   ├── graphing.py         # Signal and DKI analysis (dipy-env)
│   └── substrate/          # Substrate generation scripts (trimesh-env)
├── waveforms/              # Gradient waveforms (N x 3 CSV)
├── pixi.toml               # Pixi environments and tasks
└── README.md
```

The following directories are git-ignored and are created when the scripts run:

```text
substrate/<name>/           # Generated substrates (vertices, faces, PNG preview)
outputs/                    # Simulation signals, metadata and trajectories
graphOutputs/<signal_file>/ # Analysis plots and results
```

All scripts use paths relative to the project root, so **always run commands
from the project root directory**.

## Installation

Install [Pixi](https://pixi.sh/) if needed:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

The environments are defined for `linux-64` only. Running simulations requires
an NVIDIA GPU with CUDA support (disimpy uses CUDA through Numba).

## Environments

The project defines three separate environments, one per stage of the pipeline:

| Environment             | Python | Used for                     | Main packages                                   |
|-------------------------|--------|------------------------------|-------------------------------------------------|
| `trimesh-env`           | 3.13   | Creating substrates          | trimesh, matplotlib                             |
| `disimpy-env`           | 3.9    | Running MC simulations       | disimpy 0.3, cudatoolkit 11.8, manifold3d, tomli |
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
2. Write config          -             sim_configs/<config>.toml
3. Run simulation        disimpy-env   src/Simulation.py      -> outputs/<config>.csv
4. Analyze signals       dipy-env      src/graphing.py        -> graphOutputs/<config>.csv/
```

## 1. Substrates

### Format

The simulation loads a substrate by name from:

```text
substrate/<name>/<name>_vertices.csv   # columns: x, y, z (with header)
substrate/<name>/<name>_faces.csv      # triangle vertex indices (with header)
```

Vertex coordinates must be in **meters** (the generator scripts build meshes in
µm and scale them by `1e-6` before saving). The generator scripts also save a
`<name>_mesh.png` preview in the same folder.

Any mesh can be used as long as it is converted to this vertex/face CSV format.

### Generating substrates

The scripts in `src/substrate/` build meshes with trimesh and write them to
`substrate/<name>/`. Parameters (radii, spacing, packing, output name, ...) are
set by editing the variables at the top of each script. Run them with:

```bash
pixi run -e trimesh-env python src/substrate/<script>.py
```

| Script                          | Substrate                                                        |
|---------------------------------|------------------------------------------------------------------|
| `substrateCylinder.py`          | Packed cylinders, gamma or uniform radius distribution          |
| `substrateSphere.py`            | Single spheres over a range of radii                             |
| `randomSpheres.py`              | Randomly packed spheres, gamma radius distribution               |
| `single_axon.py`                | Single beaded axon                                               |
| `substrate_beaded_axon_SA.py`   | Beaded axons at matched surface area                             |
| `substrate_beaded_axon_V.py`    | Beaded axons at matched volume                                   |
| `substrateCATERPillar.py`       | Converts a CATERPillar output file to a mesh (see below)         |
| `viewSub.py`                    | Re-plots an existing substrate without regenerating it           |

These scripts are not updated often, so some may need small adjustments to
work with the current format.

### CATERPillar substrates

CATERPillar outputs (e.g. `CATERPillar_inputs/single_axon_run3.csv`) describe
each axon as a chain of spheres (`X Y Z inner_radius outer_radius` columns).
`substrateCATERPillar.py` creates one sphere per row, merges them into a single
watertight mesh, and saves it in the vertex/face format above. Edit the input
file and substrate name at the top of the script.

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

Two batch scripts are provided in `batch/`:

- `sbatch.sh`: runs a single config (set the `config` variable)
- `array_sbatch.sh`: runs several configs as a job array. List them in the
  `configs` array and set `#SBATCH --array=0-<N-1>` to match the number of configs

```bash
sbatch batch/sbatch.sh
sbatch batch/array_sbatch.sh
```

Before submitting, edit the scripts for your cluster: the working directory
(`cd ...`), the `--output` log path (`slurm_outputs/`), the `--partition`, and
the resource requests. For very large simulations (e.g. 1 million walkers with
100k time steps), increase `--mem`.

## Simulation Outputs

Outputs are written to `outputs/`, named after the config file. Existing files
are never overwritten: a numeric suffix (`_1`, `_2`, ...) is added instead.

| File                               | Contents                                                        |
|------------------------------------|-----------------------------------------------------------------|
| `<config>.csv`                     | Signals, one row per waveform × rotation × b-value              |
| `<config>_metadata.json`           | The full config plus the Monte Carlo seed used                  |
| `<config>_traj.csv`                | Walker trajectories (only if `[trajectory] enabled = true`)     |

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
pixi run -e dipy-env python src/graphing.py outputs/<signal_file>.csv
```

or with the equivalent Pixi task:

```bash
pixi run fitsGraph outputs/<signal_file>.csv
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
# 1. Create a substrate (edit parameters in the script first)
pixi run -e trimesh-env python src/substrate/substrateCylinder.py

# 2. Create a config from the template
cp sim_configs/config_template.toml sim_configs/my_run.toml
#    ...edit substrate name, waveforms, rotations, b-values...

# 3. Run the simulation (locally, or via sbatch on the cluster)
pixi run -e disimpy-env python src/Simulation.py sim_configs/my_run.toml

# 4. Analyze the signals
pixi run fitsGraph outputs/my_run.csv
```

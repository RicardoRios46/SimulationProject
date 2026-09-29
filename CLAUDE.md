# CLAUDE.md

Context for Claude Code sessions on this project. User-facing documentation
(usage, config options, outputs) is in README.md; read it for details. This
file covers conventions, workflow and project state not obvious from the code.

## Project

Monte Carlo diffusion MRI simulation pipeline:

1. Build substrate meshes (`src/substrate/`, `trimesh-env`)
2. Simulate with disimpy on GPU (`src/Simulation.py`, `disimpy-env`)
3. Analyze signals / DKI (`src/plot_signal.py`, `src/fit_signal.py`,
   `src/plot_trajectories.py`, sharing `src/analysis_utils.py`; `dipy-env`)

Always run scripts from the project root; all paths are relative to it.

## Scientific context

The user is a dMRI researcher; Claude helps with the code. Scientific and
modeling decisions (e.g. which radius to use, which fit terms, protocol
design) are the user's: point out physical assumptions and ask rather than
choosing silently.

Goal: explore the behavior of the diffusion signal for advanced gradient
waveform designs, using Monte Carlo simulations (disimpy) in simulated
microstructure (packed cylinders and spheres, CATERPillar axons).

- LTE: linear tensor encoding, oscillating gradients at 0, 50 and 100 Hz.
- STE: spherical tensor encoding, in isotropic (STEiso) and anisotropic
  (STEaniso) variants.
- Powder average: signal averaged over gradient rotations (files in
  `rotations/`) for each waveform and b-value; fitting log(signal) vs b gives
  D, kurtosis and variance for each waveform.
- DKI (DIPY) is fitted on the LTE data for FA, MD, AD and RD.
- Frequency dependence of D, K and V is compared between linear, square root
  and squared models.
- V_iso (variance from STE powder-average fits) is **not computed yet**; it is
  a planned analysis (see Analysis under Open work).

## Environments (pixi)

Three environments, one per stage. Run with `pixi run -e <env> python ...`.

- `trimesh-env` (Python 3.13): substrates. trimesh, matplotlib, scipy, manifold3d
- `disimpy-env` (Python 3.9): simulations. **Deliberately pinned** to numpy<2,
  numba 0.60, cudatoolkit 11.8, disimpy 0.3: the only combination disimpy is
  known to run with. Do not bump these casually. Updating disimpy for numpy>=2
  is planned as a separate piece of work.
- `dipy-env` / `default` (Python 3.13): analysis and plotting

Platforms: `linux-64` and `win-64`. After changing `pixi.toml`, update the
lock with `pixi install -a` (works on Windows) and commit `pixi.toml` and
`pixi.lock` together.

Where things run:
- Simulations run on the Linux SLURM cluster (partitions `hx`, the default
  with more GPUs, and `vx`). `batch/sbatch.sh`, `batch/submit_array.sh` and
  `batch/array_sbatch.sh` (also submitted directly) were verified there on
  2026-09-29 with small test configs on the template spheres and cylinders
  substrates.
- The user's Linux workstation has no SLURM but can SSH to the login node,
  which runs from its own copy of the project. Source code reaches it only
  through git (push `dev`, pull there); a sync script copies the git-ignored
  folders (`outputs/`, `slurm_outputs/`, substrates, configs). Claude
  prepares files here and gives the user commands to run on the login node.
  After a pull that changes `pixi.lock`, run `pixi install -e disimpy-env`
  there before submitting (see README).
- On the user's Windows workstation (RTX 5060), disimpy-env installs but GPU
  kernels fail (numba reports an invalid device number): the pinned stack is
  too old for that GPU. Use Windows only for substrates and analysis.
- Substrate and analysis scripts can be run locally to test changes.

## Conventions

- Units: all lengths in substrate configs and params are µm; mesh CSVs
  (`substrate/<name>/<name>_vertices.csv`) are in meters, as disimpy expects.
- Substrate generators (`cylinders.py`, `spheres.py`, `single_spheres.py`,
  `caterpillar.py`) read a TOML config from `substrate_configs/` and use
  `src/substrate/common.py`: `load_params` / `load_packing_params` (rejects
  unknown and missing keys), `save_substrate` (mesh CSVs, previews,
  `<name>_params.json` with params, results and provenance incl. git commit).
  New generators should follow this pattern and the shared parameter names
  (`n_objects`, `domain_size`, `min_gap`, `radius_distribution`,
  `gamma_shape`, `gamma_scale`, `radius_min`, `radius_max`, `radius_fixed`,
  `seed`, ...).
- Numbers in generated names use `p` as decimal point (`format_value`),
  e.g. `cylinders_2840_gamma_shape0p75_scale0p55_gap0p45`.
- Config folders `sim_configs/` and `substrate_configs/` are git-ignored
  except the `*_template.toml` files. `substrate/`, `outputs/`,
  `graphOutputs/` are generated and git-ignored.
- Simulation outputs go in a folder per config, `outputs/<config>/`, and the
  files from one run share the signal file's name: `<signal>.csv`,
  `<signal>_metadata.json`, `<signal>_traj.csv` (numeric suffix `_1`, `_2`,
  ... added to the signal file if it exists). the analysis scripts write to
  `graphOutputs/<signal>/` (signal file name without `.csv`).
- SLURM: submit from the project root. `batch/sbatch.sh <config>` for one
  config; `batch/submit_array.sh <list>.txt` for a job array (list file: one
  config path per line, `#` comments allowed).
- Waveform files (`waveforms/*.csv`): N x 3, mT/m, 0.02 ms per row by
  default, and already include the effect of the 180° pulse (sign flip). The
  `*_LTE1/2/3` files are the three LTE components of the STE waveforms. Check
  new waveforms with `src/plot_waveforms.py` (g, q, spectrum, centroid
  frequencies, b-tensor) and compare several with `src/compare_waveforms.py`
  (`--sum-check` verifies an STE equals the sum of its LTE components; true
  for the current STEiso and STEaniso files);
  both use the calculations in `src/waveform_utils.py`. At file amplitude the current waveforms all have
  b = 4500 s/mm² (components 1500). STEiso and STEaniso both have an isotropic
  b-tensor; they differ in the spectral content per axis. Centroid frequencies
  (dephasing spectrum): LTE0/50/100hz 8.4/47.3/98.5 Hz; STEiso ~47.6 Hz on all
  axes; STEaniso x/y/z 8.4/47.9/98.8 Hz (combined 51.3 Hz).
- Shell scripts must keep LF line endings (enforced in `.gitattributes`).
- `src/substrate/archive/` holds unmaintained beaded axon scripts kept for
  reference; don't update them unless asked.

## Workflow with the user

- Commit to the `dev` branch, never directly to `main`. The user reviews
  `dev` and merges into `main` themselves. Push only when asked.
- Work step by step: explain findings and proposed changes, make one change
  at a time, test it, then ask before committing. One commit per logical step.
- Prefer readable, simple code over defensive handling of unlikely cases
  (e.g. the user asked to drop Windows line-ending handling in list files).
- Test by running the scripts with pixi. Regression reference with the
  template configs (seed 123):
  - `spheres.py`: 500 placed, volume_fraction 0.1726, radius_mean 3.347
  - `cylinders.py`: 2840 of 3000 placed, area_fraction 0.3139, radius_mean 0.595
  - `caterpillar.py` (single_axon_run3.csv): 1 cell, 55 spheres, 1 volume

## Open work and known issues

### Simulation and environment

- Update `disimpy-env` so simulations can also run on the local Windows
  workstation (RTX 5060, compute capability 12.0). Likely needs a newer
  stack: Python >=3.10, current numba with the `numba-cuda` package, and
  CUDA >=12.8 (first release supporting this GPU generation), plus updating
  disimpy for numpy>=2. Any change must keep working on the cluster, so test
  there too. Start from the kernel failure described under Environments.
- Periodic boundaries: `periodic = true` has had issues in the past, and the
  goal is to make it work with the new substrates. Known cause to address:
  packed objects near the domain edge extend past it and do not wrap around,
  so the substrates are not proper periodic tiles.

### Substrates

- `caterpillar.py` does not check whether different cells overlap.
- CATERPillar `inner_radius` vs `outer_radius` (myelin): default is
  `inner_radius`; the user is not yet sure which to use (the current input
  file has them equal).
- Substrates generated before the generator refactor use old names
  (e.g. `sphereRadius_0.5`) and old sphere radius sampling (clipped gamma).
  These old substrates and their outputs are not being reproduced: the old
  scripts used no seed, and they have the same boundary issue (objects past
  the domain edge). New work uses substrates from the current generators.
- Create production-ready substrates (spheres, cylinders, CATERPillar) with
  the current generators, once the periodic boundary work is done, to
  replace the old ones in the simulations.

### Analysis (`src/fit_signal.py`)

- `fit_signal.py` still draws the signal decay with the fits
  (`signal_fit_<signal>.svg`, the original graphing.py plot). The user may
  want this as a separate `plot_fit_signal.py` that combines the outputs of
  `plot_signal.py` and `fit_signal.py` (e.g. powder-averaged signal and
  fitted coefficients) into a plot like the original, so `fit_signal.py`
  only fits.
- `fit_signal.py` identifies waveforms from their files (`waveform_info` in
  `analysis_utils.py`): encoding from the b-tensor shape, frequency = centroid
  of |Q(f)|^2 (follows the pending |Q|^2 vs |G|^2 decision). Frequency fits
  and DKI use all LTE waveforms. The frequency-model choice (lowest SSE with
  3 points) is fragile: switching from nominal 0/50/100 Hz to centroids
  changed the best DKI AD/RD models on the test cylinders.
- `fit_signal.py --order 3` adds the b^3 term (k3 = -6E, skewness
  k3/V^1.5); `--fix-intercept` sets C = 0. On the 6 b-value test outputs the
  order-3 fit is unstable for the spheres (V < 0 at 50/100 Hz and STEiso):
  more b-values are needed (simulations being planned). Possible
  improvement: weight the log fit (high-b points are noisier).
- Output filename typo `poweder_average_signal.csv` (kept for now, README
  documents it).
- Compute V_iso from the STE powder-average fits at each frequency. Needs the
  STE waveforms at every frequency (see Waveforms and protocol).

### Waveforms and protocol

- Extend the waveform files so there is an STE waveform at every frequency,
  so V_iso can be calculated precisely (see Analysis).
- Add the M restriction tensor (related to the spectral content of the
  waveform) to `src/plot_waveforms.py` (and the shared `src/waveform_utils.py`).
  The user will point to the literature for how to calculate it.
- The waveform scripts use the dephasing spectrum |Q(f)|^2 for the
  encoding spectrum and the centroid frequencies. The user is checking with a
  collaborator whether the gradient spectrum |G(f)|^2 = (2 pi f)^2 |Q(f)|^2 is
  the usual choice; it may need to change (it gives higher centroids).
- Explore different rotation schemes (files in `rotations/`) and compare
  whether signals and fitted parameters differ between the STEiso and
  STEaniso waveforms.

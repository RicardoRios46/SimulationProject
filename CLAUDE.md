# CLAUDE.md

Context for Claude Code sessions on this project. User-facing documentation
(usage, config options, outputs) is in README.md; read it for details. This
file covers conventions, workflow and project state not obvious from the code.

## Project

Monte Carlo diffusion MRI simulation pipeline:

1. Build substrate meshes (`src/substrate/`, `trimesh-env`)
2. Simulate with disimpy on GPU (`src/Simulation.py`, `disimpy-env`)
3. Analyze signals (`dipy-env`): `src/fit_powder_average.py` (powder-averaged
   signal: plots, cumulant fit D/K/V, frequency dependence per encoding),
   `src/fit_tensor.py` (DKI on the LTE waveforms), `src/plot_trajectories.py`,
   sharing `src/analysis_utils.py`. Each writes its own results CSVs.

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
- Simulation settings (user's decisions, from test runs on 2026-09-29 on the
  template substrates, 100k walkers, `uniform`, L40S GPUs):
  - `n_t = 10000` from now on (step length 0.19 µm at D = 1 µm²/ms over the
    58.16 ms waveforms). n_t test (`sim_configs/ntconv_*`, n_t 1000-20000):
    at n_t 1000 (0.59 µm, close to the cylinder radii) the cylinders' D, MD
    and RD are ~1.5-2% low; they level off by n_t 5000-10000. No trend for
    the spheres.
  - 10 b-values (0-4500 in steps of 500) for tests; 6 were probably enough,
    and run time grows only weakly with the number of gradients, so they can
    be reduced later if time matters.
  - Run time on an L40S for 1800 gradients (5 waveforms x 40 rotations x 9
    b-values) and 100k walkers: ~16 s + ~20 s per 1000 steps (n_t 10000:
    ~3.6 min); 3600 gradients at n_t 1000 took 46 s vs 34 s for 1800. Setup
    (mesh loading) ~30 s for the spheres, ~55-70 s for the cylinders. The
    metadata `run` section records the times of each run.
  - Monte Carlo noise, from the seed test (`sim_configs/seedtest_*`, 5 seeds
    + seed 123, n_t 10000, 100k walkers): SD of the powder-averaged signal
    ~0.0015 from b ~1000 on, about constant in b, so the relative SD grows
    as S falls (spheres 4% at b 4500, cylinders 1%). SD of the fits, spheres
    (nearly Gaussian, mostly extra-cellular): V 8-15%, order-2 K 7-14%,
    order-3 K (b <= 3) up to ~37%; cylinders: K, V ~0.5-2%; D, MD, FA
    precise for both (SD <= 0.005). SD scales as 1/sqrt(walkers), run time
    linearly: for the spheres' V to ~2-5% (LTE50 / STEiso), ~1-2M walkers
    (~36-72 min at n_t 10000, 1800 gradients). V_iso, a difference of
    variances, needs more. The spheres' V < 0 in order-3 fits appeared in
    independent runs, so it is not only noise.
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
  axes; STEaniso x/y/z 8.4/47.9/98.8 Hz (combined 51.3 Hz). TDE50hz and
  TDE100hz: 139.2 ms (6960 rows), isotropic b-tensor, b 4500 per axis (13500
  total at file amplitude), centroids 47.6 and 98.6 Hz on every axis. TDE0hz
  (from `make_tde0hz_waveforms.py`, design `pgse_full`): 6960 rows,
  isotropic, 8.6 Hz on every axis. The TDEs must be simulated in their own
  configs (longer duration).
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

- NEXT ON THE CLUSTER MACHINE: test the TDE simulations (TDE waveforms and
  `config_template_tde.toml` were made on 2026-09-29 on the Windows machine,
  not yet run). On the login node: `git pull` on `dev`, `pixi install -e
  disimpy-env`, create TDE configs from `sim_configs/config_template_tde.toml`
  (set the substrate and rotation file; `n_t = 24000` keeps the 5.8 µs time
  step of n_t 10000 at 58.16 ms), and submit a small test (e.g. the template
  spheres and cylinders, 100k walkers). Check: the run completes, the run
  times in the metadata (~2.4x those at n_t 10000), and the analysis
  (`fit_powder_average.py` should classify TDE0hz/TDE50hz/TDE100hz as STE
  with centroids ~8.6, ~47.6 and ~98.6 Hz, which gives the STE panel its
  first frequency-model fits). A config mixing TDE with 58.16 ms waveforms
  now stops with an error, by design.

- Update `disimpy-env` so simulations can also run on the local Windows
  workstation (RTX 5060, compute capability 12.0). Likely needs a newer
  stack: Python >=3.10, current numba with the `numba-cuda` package, and
  CUDA >=12.8 (first release supporting this GPU generation), plus updating
  disimpy for numpy>=2. Any change must keep working on the cluster, so test
  there too. Start from the kernel failure described under Environments.
  Also on Windows: without a `seed` in the config, `Simulation.py` fails at
  `np.random.randint(0, 2**32-1)` (`high is out of bounds for int32`: numpy<2
  uses 32-bit integers there); fine on Linux.
- Weighted cumulant fit: the signal noise is about constant in b, so the
  noise of log S grows as 1/S and an unweighted log fit gives the noisy
  high-b points too much weight. Consider weights ~S^2 in `fit_cumulant`
  (see Monte Carlo noise under Simulation settings).
- SLURM resources: the batch scripts request 4 CPUs, 32 GB and 12 h. The
  seed test (job 209524, `uniform`, 100k walkers, n_t 10000, ~4.5 min per
  task; reportseff) used ~15% CPU (~0.6 core), memory ~1-1.3 GB for most
  tasks and ~6.4-7.4 GB for two cylinder tasks, and 0.6% of the time. The
  defaults (`#SBATCH` in `batch/sbatch.sh` and `batch/array_sbatch.sh`) could
  be reduced, e.g. 1 CPU, 16 GB, 4 h, to not take cluster resources that are
  not used. The user decided to wait: `intra` runs take much longer (the
  initial positions inside the objects are computed first) and may need more
  resources, so check them first (`seff <jobid>`, reportseff, or
  `sacct -j <jobid> --format=JobID,MaxRSS,TotalCPU,Elapsed`). Memory also
  grows with walkers, gradients and n_t.
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
- Denser sphere packing: the template spheres substrate has a volume fraction
  of only 0.17, so most of the signal (with `uniform` walkers) comes from the
  extra-cellular space. The user attributes the unstable order-3 fits of the
  spheres to this (see Analysis). A sphere substrate with more packing
  (e.g. more `n_objects`, smaller `min_gap`) may be needed to reduce the
  extra-cellular space; this matters when measuring the intra-cellular signal.

### Analysis (`src/fit_powder_average.py`, `src/fit_tensor.py`)

- Both scripts identify waveforms from their files (`waveform_info` in
  `analysis_utils.py`): encoding from the b-tensor shape, frequency = centroid
  of |Q(f)|^2 (follows the pending |Q|^2 vs |G|^2 decision). The frequency
  fits of D/K/V have one panel per encoding (LTE, STE), ready for STE
  waveforms at several frequencies; STE has only 2 waveforms now (no fits).
  STEaniso is classed as STE, but its axes have different frequencies, so its
  combined centroid is not a single frequency: consider leaving it out of the
  STE frequency series. The frequency-model choice (lowest SSE with 3
  points) is fragile: switching from nominal 0/50/100 Hz to centroids
  changed the best DKI AD/RD models on the test cylinders.
- `fit_powder_average.py --order 3` adds the b^3 term (k3 = -6E, skewness
  k3/V^1.5); `--fix-intercept` sets C = 0. Pilot runs (2026-09-29,
  `sim_configs/pilot_*.toml`: 19 b-values 0-4500, 100k walkers, `uniform`,
  n_t 1000, template substrates) showed it is not only Monte Carlo noise: for the
  cylinders order 3 is stable, for the spheres V is ~0 or negative at
  50/100 Hz and STEiso (signal nearly Gaussian, K ~0.05, mostly
  extra-cellular). Order-2 K rises with the maximum b fitted (truncation
  bias); order 3 is stable up to b ~2.5-3 ms/µm² and drifts beyond. Options:
  a `--b-max` option for the fit (fit_cumulant already has b_max; the user
  will decide on it when the intra/extra runs start), weighting the log fit
  (high-b points are noisier), intra/extra positions, denser spheres (see
  Substrates). Note: identical reruns with the same seed only show that the
  simulation is deterministic, not that the noise is small; the n_t test
  showed the noise is large for the nearly Gaussian spheres signal (see
  Simulation settings).
- Compute V_iso from the STE powder-average fits at each frequency. Needs the
  STE waveforms at every frequency (see Waveforms and protocol).
  Diffusion time: the TDE waveforms last 139.2 ms (each axis encoded in its
  own time window), the LTE/STEiso/STEaniso waveforms 58.16 ms. Comparing V
  (or V_iso) between a TDE and an LTE/STE at the same centroid frequency also
  compares different encoding times; revisit this when comparing V_iso
  between waveforms (user's note, 2026-09-30).

### Waveforms and protocol

- STE at every LTE frequency (done 2026-09-30): TDE (triple diffusion
  encoding) waveforms `TDE50hz`, `TDE100hz` built by
  `src/make_tde_waveforms.py` from the LTE files (x, y, z first halves, gap,
  x, y, z second halves; 139.2 ms). 0 Hz: the same construction fails (the
  PGSE half does not return q to 0, so the axes overlap in q: eigenvalues/b
  0.78, 0.19, 0.03 for x-y-z; no order of the second halves fixes it, z-y-x
  is worse, 0.89, 0.09, 0.03), so `make_tde_waveforms.py` now stops for 0 Hz.
  `src/make_tde0hz_waveforms.py` builds it instead: the waveforms are only
  for simulations (not for a scanner), so the refocusing pulse gap is
  ignored; three equal windows (one per axis) over the TDE50hz length, each
  a + and a - lobe (LTE0hz ramp) with an optional plateau, so q returns to 0
  before the next axis. Designs `bipolar`, `pgse` (with lead/tail zeros),
  `bipolar_full`, `pgse_full` (one zero row at each end): 9.6, 9.4, 8.9,
  8.6 Hz, all isotropic. The user chose `pgse_full` (closest to LTE0's
  8.4 Hz) as `waveforms/TDE0hz.csv`; the collaborator had suggested the
  bipolar one. The candidates are git-ignored and the script is kept to
  revisit them. With one refocusing pulse a scanner version would need the
  pulse in the middle (explored: a design with y straddling the gap), but
  that is not needed here.
  Also tried (2026-09-30, not saved): the same LTE waveform on x, y and z at
  the same time ("xyz - pulse - xyz") is just an LTE along (1,1,1), so the
  b-tensor is linear (1, 0, 0) at every frequency, also with sign flips. The
  LTE0/50/100 waveforms on x/y/z at the same time (each scaled to equal b)
  are not isotropic either (0.45, 0.33, 0.22) and do not match STEaniso,
  which uses its own timings. An STE needs q_x, q_y, q_z of equal size with
  zero overlaps (integral of q_i q_j = 0 for i != j): either at different
  times, each returning to 0 first (sequential TDE), or at the same time with
  different shapes (STEiso, STEaniso). All three axes near 0 Hz may not be
  possible with one refocusing pulse; STEaniso's x axis reaches 8.4 Hz
  because y and z are at higher frequencies.
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

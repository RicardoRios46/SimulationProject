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
  with more GPUs: 7 nodes with 2 GPUs (L40S) and 500 GB RAM each, and `vx`). `batch/sbatch.sh`, `batch/submit_array.sh` and
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
- Substrate generators (`cylinders.py`, `spheres.py`, `single_spheres.py`)
  read a TOML config from `substrate_configs/` and use
  `src/substrate/common.py`: `load_params` / `load_packing_params` (rejects
  unknown and missing keys), `save_substrate` (mesh CSVs, previews,
  `<name>_params.json` with params, results and provenance incl. git commit).
  New generators should follow this pattern and the shared parameter names
  (`n_objects`, `domain_size`, `min_gap`, `radius_distribution`,
  `gamma_shape`, `gamma_scale`, `radius_min`, `radius_max`, `radius_fixed`,
  `seed`, ...). Exception: `caterpillar.py` takes the CATERPillar CSV and
  command-line options (user's decision: the substrate is defined by the
  CATERPillar JSON; the options only set the mesh). Everything for one
  CATERPillar substrate is in `substrate/caterpillar_<run>/`, with
  CATERPillar's JSON, CSV and growth info in its `caterpillar/` subfolder
  (the converter copies them there if the CSV is elsewhere); JSON configs in
  `substrate_configs/caterpillar/` (only `*_template.json` tracked;
  CATERPillar's `example_config.json` kept there untracked for reference).
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
    ~3.6 min); 3600 gradients at n_t 1000 took 46 s vs 34 s for 1800, and
    the TDE test (1080 gradients, n_t 24000) ~19 s per 1000 steps: the run
    time depends mostly on steps x walkers, little on the gradients.
  - n_t for thin axons (job 210509, `sim_configs/ntconv2_*`, cylinders
    with mean r 0.35 µm, min 0.1 µm, gap 0.1 µm, intra and extra, 100k
    walkers, n_t 10000/20000/50000/100000 = step 0.19/0.13/0.084/0.059 µm):
    intra is converged from n_t 10000 (D, K, V, RD, FA unchanged within
    noise; RD 0.0032 at every n_t for LTE0); extra changes slightly from
    10000 to 20000 (LTE0 RD 0.743 -> 0.747-0.749, DKI AD 1.013 -> 1.001),
    then stays within the noise up to 100000. Recommendation: n_t 20000 for
    the 58.16 ms waveforms (and 48000 for the 139.2 ms TDE, same 2.9 µs
    step); the production cylinders have thicker axons (mean r 0.49, min
    0.15 µm), so this is conservative. Timing: ~23 s per 1000 steps for
    1800 gradients and 100k walkers (n_t 100000: 39 min), setup ~130 s. Setup
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
  - TDE test (2026-09-30, job 210439, `sim_configs/tde_*.toml` from
    `config_template_tde.toml`: TDE0hz/50hz/100hz, n_t 24000, 100k walkers,
    seed 123, template substrates): ran fine after fixing the time point count
    in `Simulation.py` (6961 points for 6960 rows); 464 s (spheres) and
    427 s (cylinders) of simulation. `fit_powder_average.py` classifies the
    three TDEs as STE (8.6, 47.6, 98.6 Hz) and fits the STE frequency
    models; `fit_tensor.py` has nothing to fit (no LTE). TDE50hz matched
    STEiso within the seed-test SD (spheres V 0.0448 vs 0.0446 +- 0.0035,
    cylinders 0.0556 vs 0.0555 +- 0.0010). Spheres: STE ~ LTE at 50/100 Hz
    (no microscopic anisotropy); TDE0hz V 0.1135 vs LTE0hz 0.1212 +- 0.0015.
    Cylinders: STE V ~0.054 at every frequency vs LTE ~0.11.
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
  (dephasing spectrum |Q(f)|^2, checked with the collaborators on
  2026-09-30 as the right choice; the gradient spectrum |G(f)|^2 =
  (2 pi f)^2 |Q(f)|^2 would give higher centroids): LTE0/50/100hz 8.4/47.3/98.5 Hz; STEiso ~47.6 Hz on all
  axes; STEaniso x/y/z 8.4/47.9/98.8 Hz (combined 51.3 Hz). TDE50hz and
  TDE100hz: 139.2 ms (6960 rows), isotropic b-tensor, b 4500 per axis (13500
  total at file amplitude), centroids 47.6 and 98.6 Hz on every axis. TDE0hz
  (from `make_tde0hz_waveforms.py`, design `pgse_full`): 6960 rows,
  isotropic, 8.6 Hz on every axis. The TDEs must be simulated in their own
  configs (longer duration).
- Periodic substrates (working, tested 2026-09-30): the substrate and
  simulation templates use `periodic = true`. From the disimpy 0.3 code:
  the periodic voxel is the bounding box of the mesh (+ padding); walkers
  are never wrapped (correct phases) and see the geometry through periodic
  subvoxels, so surfaces cut open at the voxel faces continue in the next
  tile; disimpy's own `intra`/`extra` sampling (ray cast along +x) is not
  periodic. So: `spheres.py`/`cylinders.py` with `periodic = true` build
  true periodic tiles (`common.tile_periodic`: minimum-image placement,
  wrapped copies, mesh cut open at the faces, bounding box = tile checked;
  cylinders are open tubes with z period = domain_size, replacing the old
  long-cylinder hack) and write `<name>_objects.csv`; `Simulation.py`
  samples `intra`/`extra` positions from that file (periodic distances,
  < 1 s for 100k walkers; metadata `run.initial_positions`). Use
  `periodic = true` in the simulation only with `_periodic` substrates.
  Cluster tests (job 210468, `sim_configs/pbc_*`, template tiles and copies
  with every object shifted by (0.37, 0.61, 0.23) x the tile, 100k walkers,
  n_t 10000): shifted vs original signals agree within the Monte Carlo
  noise for uniform/intra/extra (max |dS| 0.0006-0.0021); uniform = volume-
  fraction mix of intra and extra (within 0.002); 20 intra trajectories per
  substrate never leave their object (max distance/radius 0.9997), incl.
  walkers crossing the tile faces. Setup ~30 s (spheres), ~150 s (periodic
  cylinders, 450k faces); intra/extra start as fast as uniform.
  AD check (cylinders, D0 = 1 µm²/ms): the signal along the axis is free
  diffusion with D0 in every compartment (direct fit of the rotation 5 deg
  from z matches D0 cos^2 + RD sin^2 within 0.002 at 0/50/100 Hz); DKI AD
  from `fit_tensor.py` is 0.99-1.00 at 50/100 Hz but 1.01 (intra, extra)
  and 1.02 (uniform) at 0 Hz: a DKI model bias where the radial kurtosis is
  large (intra K ~1.6), not a simulation error.
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
  - `spheres.py` (template: 150 µm tile, 480 spheres, gamma 5.76/1.04 µm,
    radii 2-12 µm, gap 1 µm, periodic): 479 placed, volume_fraction 0.1802,
    radius_mean 5.981
  - `cylinders.py` (template: 60 µm tile, 1200 cylinders, gamma 4/0.12 µm,
    radii 0.15-1.5 µm, gap 0.1 µm, periodic): 1198 placed, area_fraction
    0.2971, radius_mean 0.4861

## Open work and known issues

### Simulation and environment

- TDE vs STE and LTE: in the first TDE test (see Simulation settings)
  TDE50hz matched STEiso within the noise (same 47.6 Hz centroid, 139.2 vs
  58.16 ms), so the diffusion time difference did not show in D, K, V; one
  seed only, check with more seeds before relying on it.
- Combined analysis across runs (planned): each run is analysed separately
  now. Needed: combine the results of the LTE/STE runs (LTE0/50/100, STEiso,
  STEaniso; 58.16 ms) and the TDE runs (TDE0/50/100; 139.2 ms, their own
  configs) per frequency, e.g. V_LTE - V_STE towards V_iso, including both
  STEiso and STEaniso. Simulate them with the `intra`, `extra` and `uniform`
  walker positions. Order (user's decision): first make the substrates
  production ready (see Substrates), then these simulations and the
  combined analysis.

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
  not used. The user decided to wait: `intra` runs took much longer (disimpy
  computed the initial positions inside the objects; now `Simulation.py`
  samples them from the objects file in < 1 s, see Periodic boundaries), and
  may need more resources, so check them first (`reportseff <jobid>`, or
  `sacct -j <jobid> --format=JobID,MaxRSS,TotalCPU,Elapsed`). Memory also
  grows with walkers, gradients and n_t. Host memory peak at the end of a
  run: disimpy copies all phases (gradients x walkers, float64) to the CPU
  and computes exp(1j * phases) over the whole array, so the peak is ~5x
  the phase array: 1800 gradients x 1M walkers ~72 GB, 1080 x 1M ~43 GB.
  The V_iso run (2026-09-30, `sim_configs/viso_*`, 1M walkers) was
  submitted with --mem=128G; check its real use before choosing defaults.
  The cluster has no `seff`; use `reportseff <jobid>` (Python wrapper; its
  MemEff column x the requested memory gives the peak), as for job 209524. If nodes cannot give that much, Simulation.py could run the
  walkers in chunks (e.g. 4 x 250k with different seeds) and sum the
  signals.
- Periodic CATERPillar substrates: `caterpillar.py` has no `periodic`
  option or objects file yet (its `intra`/`extra` runs fall back to
  disimpy's sampling, which is fine only if non-periodic). Whether it can be
  periodic depends on whether the CATERPillar output is a periodic tile.

### Substrates

- `caterpillar.py` does not check whether different cells overlap.
- CATERPillar `inner_radius` vs `outer_radius` (myelin): default is
  `inner_radius`; to be replaced by using both surfaces (see the converter
  work in the CATERPillar notes). The old example `single_axon_run3.csv`
  was removed (2026-09-30); new configurations will be made.
- Substrates generated before the generator refactor use old names
  (e.g. `sphereRadius_0.5`) and old sphere radius sampling (clipped gamma).
  These old substrates and their outputs are not being reproduced: the old
  scripts used no seed, and they have the same boundary issue (objects past
  the domain edge). New work uses substrates from the current generators.
- Create production-ready substrates (spheres, cylinders, CATERPillar) with
  the current generators, to replace the old ones in the simulations. The
  periodic boundary work is done for spheres and cylinders (see Periodic
  substrates under Conventions); CATERPillar still needs it.
- Realistic substrates for ex-vivo mouse brain (literature check and user's
  decisions, 2026-09-30). The simulations accompany a paper with ex-vivo
  mouse data: approach that setting, but it need not be matched exactly.
  Literature: ECS 15-25% in cryo-fixed (near in vivo) mouse neocortex vs
  < 5% after aldehyde fixation (Korogod et al. 2015 eLife); mouse cortex
  ~9.2e4 neurons/mm^3, pyramidal soma volume ~1100 µm^3 (r ~6.4 µm), so
  neuronal somata fill only ~10% of cortex; the rest is mostly neuropil
  (axons 31-36%, dendrites 23-35%), not free water. SANDI (Palombo et al.)
  uses soma radii 2-10 µm (microglia to large neurons), neurites <= 1.5 µm;
  ex-vivo mouse cortex fits r ~6-10 µm. Mouse corpus callosum inner axon
  diameters 0.47-0.88 µm (older EM studies) to 1.03 +- 0.41 µm (3D EM,
  Lee et al.), g-ratio ~0.6; gamma is the usual fit but GEV/log-normal fit
  better (Sepehrband et al. 2016). Rigid spheres cannot pack above ~0.64-0.7
  (random close packing); dense axon generators (CACTUS, up to 95%) use
  growth/optimisation, not random sequential placement.
  Decisions: the spheres represent soma in gray matter, so a realistic soma
  volume fraction is ~10-20%, which the current placement reaches without
  distorting the radii; no denser sphere packing is needed. With a
  spheres-only substrate the meaningful simulation is `intra` (soma);
  `extra`/`uniform` are dominated by nearly free extra-cellular water,
  whereas in tissue that space is mostly neuropil. Move the soma radii
  towards mouse values and the axon radii to a smaller mean. Chosen
  (templates since 2026-09-30): spheres mean r ~6 µm, SD ~2.2 (gamma 5.76 /
  1.04 µm, 2-12 µm), 480 in a 150 µm tile -> volume fraction 0.18;
  cylinders: inner radius gamma alpha 4, beta 0.12 µm (mean ~0.49 µm,
  SD ~0.22, diameter ~1 µm), radii 0.15-1.5 µm, 1200 in a 60 µm tile with
  gap 0.1 µm -> area fraction 0.30 (kept in the plausible intra-axonal
  range; revisit with myelin). Why these values (user's decision, to back
  it in the paper): they are the values the CATERPillar authors validated
  against the mouse corpus callosum 3D EM of Lee et al. 2019 (8-week
  C57BL/6 female, genu; inner diameter 1.03 +- 0.41 µm, g-ratio ~0.6; see
  the CATERPillar notes below); radius_min 0.15 µm is CATERPillar's
  MinRadius. The same values are used for our cylinders and for the
  CATERPillar configs. (A first choice, gamma 5.44/0.0643, mean r 0.35 µm
  from the older 2D EM studies, was replaced before use; the n_t
  convergence test ntconv2 was run on that substrate, whose axons are
  thinner, so it is a conservative check.) A gap of 0.45 µm (the old value)
  reached only 0.22 with thin axons and skipped large ones. Template values before this: spheres gamma shape 2, scale 1.5 µm
  (mean r ~3.3 µm); cylinders shape 0.75, scale 0.55 µm (mean r ~0.59 µm).
  Packing scan for the record (placement only, 100 µm periodic tile,
  spheres template radii, 4000 drawn): in drawn order 0.31 (gap 1),
  0.37 (0.5), 0.40 (0.2), with the placed mean radius falling to ~2.4-2.6 µm
  (large spheres skipped); largest first 0.54 (gap 1), 0.64 (0.2), 0.675
  (0.2, 20000 attempts, 750 s), but bimodal: nearly all 8-12 µm and < 2 µm
  spheres placed, only 20-30% of 3-8 µm.
- Myelin (future): the cylinders have no myelin (all space outside is free
  water). Later, myelinated axons could be modelled as an excluded ring
  between an inner (axon) and outer (fibre) radius, g-ratio ~0.6-0.7, with
  both radii in the objects file. Revisit the intra-axonal area fraction
  (0.30 now) then.
- CATERPillar (https://github.com/Mic-map/CATERPillar; paper bioRxiv
  2025.06.20.660694). STATUS (2026-09-30, paused to run the V_iso
  simulations; resume here): done: findings below, folder layout and
  `caterpillar.py` taking the CSV + options (commit 4ac1700), cylinders using
  CATERPillar's validated axon radii. Next: (1) the JSON templates
  `substrate_configs/caterpillar/wm_mouse_template.json` and
  `gm_mouse_template.json` with the values in the plan below (WM needs the
  user's decision on the myelinated/unmyelinated fractions first); (2) the
  converter work listed below (objects file with both-surface myelin, then
  the central start region); (3) a first small run (50 µm voxel) to check
  CATERPillar's run time and the mesh size.
  Findings from its code and paper (2026-09-30):
  - Config (JSON, run as `./CATERPillar --config <file>.json`): `Alpha`,
    `Beta` are a gamma of the INNER axon radius (µm; the code adds pi r^2 to
    the ICVF and compares with `MinRadius`); myelinated outer radius =
    inner + thickness, thickness = K1 + K2 D + K3 ln D (D inner diameter;
    K1-K3 = 0.35, 0.006, 0.024 from Lee et al., mouse). The paper matched
    mouse corpus callosum (Lee et al. 2019) with alpha 4, beta 0.12 (used
    for our cylinders too), epsilon 0.4, beading 0.3; alpha 4, beta 0.25 is
    advised for unmyelinated axons (~outer radius). `AxonsICVF` /
    `AxonsWithMyelinICVF` in % (myelinated incl. myelin; up to ~70% total,
    e.g. 150 µm voxel < 12 h with 20 threads, ~300 MB each). `FODF_c2` =
    <cos^2> of axon angle to z (1 aligned, 1/3 isotropic);
    `NumberOfPopulations` 1-3 perpendicular bundles. Glia = two populations
    of soma + processes modelled on astrocytes (protoplasmic in GM, fibrous
    in WM); soma radius normal (mean/std); no neurons or oligodendrocytes.
    Axons are placed largest first. Bug in example_config.json: the key
    `OndulationFactor` is ignored (the code reads `UndulationFactor`,
    default 5).
  - Output `<Filename>.csv` (cell_type cell_id component component_id X Y
    Z inner_radius outer_radius, µm) and `_growth_info.txt`. Not periodic:
    the authors extended MC/DC with mirror boundaries and started walkers
    in a central region 30 µm from the edges.
  - Fractions (open, user's decision): Lee et al. give none (they
    segmented only myelinated axons, 36 x 48 x 20 µm, genu). Reports of
    the myelinated share of mouse corpus callosum axons differ (e.g. ~90%
    of fibres in a recent EM study vs ~30% by count in older studies);
    check before fixing AxonsWithMyelinICVF / AxonsICVF.
  - Plan: two configs for ex-vivo mouse, WM (corpus callosum: alpha 4,
    beta 0.12, MinRadius 0.15, K1-K3 of Lee, FODF_c2 ~0.9 for the ~18 deg
    dispersion of Lee, epsilon 0.4, beading 0.3, ~60% total ICVF, small
    fibrous-astrocyte fraction) and GM (approximation accepted by the user:
    ~25-30% dispersed unmyelinated axons, glia pop 1 as neuron somas
    r ~6.4 +- 1.5 µm ~10% + processes as dendrites ~20%, pop 2 as
    protoplasmic astrocytes r ~4 µm). Folder layout for the configs,
    runs and conversions: see the proposal to be agreed (below).
  - Converter work (`caterpillar.py` + `Simulation.py`), to do: (1) myelin
    with both surfaces (inner and outer), no walkers in the myelin; (2) an
    objects file (sphere centres, inner/outer radii per cell) for the
    intra/extra sampling; (3) an option to start walkers only in a central
    region (edge buffer), since the substrates are not periodic and
    disimpy has walls or periodic boundaries only; (4) check the mesh size
    and boolean-union time for thousands of axons (start with a 50 µm voxel).
- MC/DC simulator (used by the CATERPillar authors, with their mirror
  boundary extension): the user may be able to compile it with those
  changes, but the decision is to keep disimpy, which already works here.
- Gray matter substrate generators: the user has seen other simulators
  aimed at gray matter; look for them later (CATERPillar is WM-focused and
  its GM use here is an approximation).

### Analysis (`src/fit_powder_average.py`, `src/fit_tensor.py`)

- Both scripts identify waveforms from their files (`waveform_info` in
  `analysis_utils.py`): encoding from the b-tensor shape, frequency = centroid
  of |Q(f)|^2 (checked with collaborators, see Waveform files). The frequency
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
- Explore different rotation schemes (files in `rotations/`) and compare
  whether signals and fitted parameters differ between the STEiso and
  STEaniso waveforms.

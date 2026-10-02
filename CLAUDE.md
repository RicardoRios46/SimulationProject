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
   `src/fit_tensor.py` (DKI on the LTE waveforms), `src/fit_viso.py` (V_iso
   and V_aniso across runs), `src/aggregate_seeds.py` (mean, SD, SEM over
   seed repeats of fit_viso), `src/plot_trajectories.py`, sharing
   `src/analysis_utils.py`. Each writes its own results CSVs.

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
- Frequency dependence of D, K and V: plotted against the centroid
  frequency; the linear / square root / squared model choice is optional
  (`--frequency-models`, see Analysis under Open work).
- V_iso = V of the STE (TDE) fits, V_aniso = V_LTE - V_iso (`fit_viso.py`).

## Reports

- `reports/` is git-ignored (Typst, elsearticle template; Typst at
  `/home/ricardo/software/typst/typst`, compile with `--font-path fonts`).
  `reports/viso_2026-10/` (V_iso consistency TDE vs STE_I, STE_A - STE_I,
  order 2 vs 3; shown to the collaborators) is ARCHIVED, read-only
  (`chmod -R a-w`): do not update it. The follow-up results (see "V_iso
  follow-up" under Simulation settings) are in `reports/viso2_2026-10/`
  (finished 2026-10-02, same authors; `make_figures.py` reads the
  aggregate_seeds.py folders and the per-seed fit_viso folders; compile
  with `--font-path fonts` from that folder). Layout chosen by the user:
  Fig. 1 substrates; Fig. 2 D(f) (TDE and LTE) + forest plot of the paired
  D differences; Fig. 3 V_iso(f) and V_aniso(f) + forest plot of the V
  differences; Fig. 4 V_iso(TDE50) vs fit order with the expected mixture
  value for uniform; Table 1 V of the isotropic encodings (TDE series,
  STE_I, STE_A), Table 2 V_aniso; supplement: b_max test (figure and
  table), kurtosis arranged as Fig. 2, padded waveforms, waveform table,
  V-difference table. Values as mean +- SD, differences as mean +- 95% CI
  (t_0.975,5 x SEM); t only to say whether a difference is above the noise
  (t = mean/SEM is not an effect size). Report abbreviations: STE_I =
  STEiso, STE_A = STEaniso.

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
- V_iso production run (2026-09-30, job 210518, `sim_configs/viso_*`,
  outputs `outputs/viso_<substrate>_<lte_ste|tde>_<position>/`): template
  substrates (spheres_479..., cylinders_1198..., periodic), set A = LTE0/50/100
  + STEiso + STEaniso (n_t 20000), set B = TDE0/50/100 (n_t 48000),
  intra/extra/uniform, 1M walkers, 10 b-values, seeds 1001-1012, --mem=128G.
  Simulation time: set A 25-29 min, TDE 38-61 min (cylinders slower);
  1M walkers ran ~2.7x faster per walker than 100k (set A ~7.5 s per 1000
  steps per 100k walkers vs ~23 s at 100k: larger batches use the GPU
  better), so the 100k timing formula overestimates large runs. Memory
  peak (reportseff MemEff x 128 GB): set A 62-71 GB, TDE 28-39 GB (~5x the
  phase array, as estimated). Order-2 fits (all b) per compartment: spheres
  intra STE ~ LTE at each frequency (isotropic); cylinders intra V_STE
  ~0.0005 vs V_LTE ~0.049 (all anisotropic); cylinders uniform V_STE
  ~0.058 at every frequency (~two-compartment mixing f(1-f)(dD)^2);
  STEaniso behaves like the average of its axes (spheres intra D 0.650 vs
  LTE mean 0.646); TDE50 vs STEiso at 47.6 Hz: spheres intra V 0.0214 vs
  0.0237, uniform 0.0165 vs 0.0192 (~2-2.5 noise SD; possibly the diffusion
  time, 139 vs 58 ms; cylinders uniform 0.0588 vs 0.0587); cylinders intra
  LTE D 0.300 vs STE 0.336 (= (D_ax + 2 D_rad)/3): order-2 truncation bias
  at K ~1.6 over b <= 4.5. User's decisions for the analysis: V_iso(f) = V
  of the STE fits, mainly the TDE series, also showing STEiso and STEaniso
  (mixed frequency); V_aniso(f) = V_LTE(f) - V_STE(f); cumulant order 3
  (whether the order matters is a sub-objective, to be reported).
  `src/fit_viso.py` results (2026-10-01, `graphOutputs/viso/<substrate>_
  <position>_order<2|3>/`): the order matters: cylinders intra V_aniso
  0.084 (order 3) vs 0.049 (order 2), spheres intra V_iso at 47.6 Hz 0.0107
  vs 0.0214, cylinders uniform V_iso 0.066 vs 0.058. Order 3: cylinders
  intra V_iso ~0.001, V_aniso ~0.084 flat in frequency; cylinders uniform
  V_iso ~0.066, V_aniso ~0.06, flat; spheres intra V_iso 0.015 / 0.011 /
  0.004 at 8.6 / 47.6 / 98.6 Hz, V_aniso within +-0.004 of 0 (noise and
  model error; should be 0 for spheres). STEaniso - STEiso: ~0 for the
  cylinders (no frequency dependence across 8-99 Hz); spheres D -0.12
  (intra), V +0.010 (uniform, both orders; possibly the different
  diffusivities per axis acting as variance), intra -0.003 (order 3) /
  -0.010 (order 2), extra +-0.005 (sign changes with order: noise).
  Each value is from one run (one seed): no error bars yet.
- V_iso follow-up (collaborators' suggestions, 2026-10-01; results for the
  next report): `sim_configs/viso2/` (108 configs; `list_rep1.txt`,
  `list_rep2-6.txt`), outputs `outputs/viso2_<sub>_<lte|long>_<position>_rep<1-6>/`,
  sub = cylfixed / cylgamma / sphgamma (the 3 µm diameter substrates, see
  Substrates), set `lte` = LTE0/50/100 (n_t 20000), set `long` = STEiso_pad,
  STEaniso_pad, TDE0/50/100 (139.2 ms, n_t 48000), intra/extra/uniform, 1M
  walkers, 10 b-values, 6 seeds (20000 + rep*100 + index), --mem=100G. The
  earlier seed repeats of the template-substrate V_iso run were dropped for
  this. Seed 1 (job 211809): lte 20 min (cylinders) / 32 min (spheres),
  long 62-66 / 77-80 min, setup 50-80 s; memory peak lte 26-42 GB, long
  58-71 GB; CPU ~21% of 4. ~14 GPU-h per seed. Seed-1 results (order 3):
  STEiso_pad - TDE50 within 0.0015 everywhere; fixed-radius cylinders intra
  V_iso ~0.001 flat, gamma cylinders intra 0.001 / 0.009 / 0.021 and gamma
  spheres intra 0.002 / 0.005 / 0.020 at 8.6 / 47.6 / 98.6 Hz (radius
  spread -> V_iso grows with frequency); spheres V_aniso ~0; STE_A - STE_I
  negative in every uniform case (down to -0.009, spheres). Seeds 2-6: job
  211849 (90 tasks, same times; memory peak max 71.3 GB over all 108 tasks,
  so 100G has ~29% headroom; CPU ~21% of 4, so --cpus-per-task=1 would do).
  Analysis (2026-10-02): fit_viso per seed at orders 2-5
  (`graphOutputs/viso/viso2_<sub>_<pos>_rep<r>_order<o>/`) and with
  `--b-max` 2.0-4.0 at orders 2-4 (`..._order<o>_bmax<2p0..4p0>/`),
  aggregated over the 6 seeds (`..._order<o>[_bmax<b>]_seeds/`). Results
  (order 3, all b, mean +- SD, n 6): V_iso intra cylfixed 0.0012 / 0.0012 /
  0.0010 (flat), cylgamma 0.0013 / 0.0086 / 0.0208, sphgamma 0.0016 /
  0.0050 / 0.0204 (+-0.0002) at 8.6 / 47.6 / 98.6 Hz; V_aniso intra
  cylfixed 0.085 / 0.079 / 0.067, cylgamma 0.083 / 0.067 / 0.050, spheres
  ~0 (+-0.001). STE_I,pad - TDE50 (V): intra within +-0.0002; extra and
  uniform ~+0.001 (t 4-9 at orders 4-5: small but real, same centroid but
  different spectral shape). STE_A - STE_I (V): uniform -0.0036 / -0.0027 /
  -0.0092 (cylfixed / cylgamma / sphgamma, |t| 12-90), extra within noise,
  sphgamma intra +0.0009 (t 89, tiny but precise: intra signal in small
  spheres has very low MC noise). D: STE_I - TDE50 within the noise
  everywhere (<= 0.5e-3); STE_A - STE_I positive where D depends on f
  (intra +11.5e-3 cylfixed, +8.6e-3 cylgamma, +22.5e-3 sphgamma): STE_A's D
  ~ the mean of D(f) at its three axis frequencies, above D at 47.6 Hz as
  D(f) is convex. Fit order: intra/extra converge from
  order 3-4; uniform does NOT (two-compartment mixture, D_intra << D_extra):
  sphgamma uniform V(TDE50) 0.097 / 0.171 / 0.159 / 0.135 at orders 2-5.
  Check against the expected mixture V = f V_i + (1-f) V_e + f(1-f)(D_i -
  D_e)^2 (impermeable, f = volume fraction, intra/extra from order 4, b <=
  2.5): TDE50 expected 0.135 (sphgamma), 0.053 (cylfixed), 0.043
  (cylgamma); order 4 matches for the cylinders at every b_max (0.052,
  0.042), order 2 at b <= 2 (0.135) and order 4 at b <= 2.5 (0.138) for the
  spheres; order 3 at all b overestimates uniform V by ~20-27% (0.171,
  0.066, 0.052) and drifts with b_max. User's decision (2026-10-02): keep
  order 3 in the main results with a caveat for uniform; the b_max test
  goes in the supplementary material of the next report.
- Cylinder radius sweep for V_aniso (planned 2026-10-02, user's
  decisions): fixed radii 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6 µm, substrates
  `substrate_configs/cylinders_fixed_r<R>.toml` ->
  cylinders_1200_fixed_r<R>_gap0p1_periodic (1200 cylinders, tile
  R x 112 µm, area fraction 0.300; r1p5 is the viso2 substrate), intra
  only, D_0 = 1 µm²/ms, 1M walkers, 3 seeds (30000 + rep*100 + index).
  All waveforms have the same length (user's decision): LTE0/50/100_pad
  (`pad_waveforms.py LTE0hz LTE50hz LTE100hz`, same b and centroids),
  STEiso_pad, STEaniso_pad and TDE0/50/100 (139.2 ms) in ONE config per
  radius and seed (n_t 48000, rotation_file list: fibRot40 for LTE,
  ESRD2_0040 for STE/TDE): `sim_configs/radii/` (27 configs, `list.txt`).
  2880 gradients -> memory peak ~1.6x the viso2 long set (58-71 GB), so
  ~95-115 GB: submit with --mem=160G. Run time ~65-80 min per config
  (depends mostly on steps x walkers). Expected (one-mode model, checked on viso2 cylfixed r 1.5 within
  ~5%): V_aniso = (4/45)(D_0 - D_rad)^2, D_rad(f) ~ D_0 w^2/(a1^2 + w^2),
  a1 = 1.841^2 D_0/R^2, f_c = a1/2pi; V_aniso changes most across 8-100 Hz
  for R ~1.5-4 µm (drop 27% at 1.5, 57% at 2, 81% at 2.5, 92% at 3),
  flat at ~0.089 below ~1 µm, already small at 8 Hz above ~5 µm. f_c
  scales as D_0/R^2 (an ex-vivo D_0 ~0.5 shifts this to radii ~1.4x
  smaller). The "0 Hz" PGSE spectrum is broad, so the model underestimates
  D_rad there for large R.
- Waveform files (`waveforms/*.csv`): N x 3, mT/m, 0.02 ms per row by
  default, and already include the effect of the 180° pulse (sign flip). The
  `*_LTE1/2/3` files are the three LTE components of the STE waveforms. Check
  new waveforms with `src/plot_waveforms.py` (g, q, spectrum, centroid
  frequencies, b-tensor) and compare several with `src/compare_waveforms.py`
  (`--sum-check` verifies an STE equals the sum of its LTE components; true
  for the current STEiso and STEaniso files);
  both use the calculations in `src/waveform_utils.py`. `STEiso_pad` /
  `STEaniso_pad` (`src/pad_waveforms.py`): zeros appended to 6960 rows
  (139.2 ms, TDE length; same b-tensor and spectrum) so they run in the TDE
  configs (user's decision: equal durations for the comparison; a longer
  pause around the 180° pulse was not used, it would change STEaniso's b_x). At file amplitude the current waveforms all have
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
  (axons 31-36%, dendrites 23-35%), not free water. SANDI (Palombo et al.
  2020): simulations with soma radii 2-10 µm (microglia to large neurons),
  neurites <= 1.5 µm; in-vivo human fits 2-12 µm (mean 10 +- 3); one
  ex-vivo mouse brain, cortex matched r ~6-10 µm (soma signal fraction
  60-65%, a model fraction, not the histological volume fraction). In-vivo
  mouse SANDI (Ianus et al. 2022 NeuroImage 254:119135; 9.4 T, Delta 20 ms,
  b <= 12.5): apparent radius 6-9 µm in GM. SANDI radii are MR-apparent
  (tail-weighted towards large cells). Mouse corpus callosum inner axon
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
  1.04 µm, 2-12 µm), 480 in a 150 µm tile -> volume fraction 0.18 (placed
  mean r 5.98 µm; MR-effective radius (<r^7>/<r^3>)^(1/4) ~8.7 µm, within
  the mouse SANDI range; kept, user's decision 2026-10-01; planned: extra
  scenarios with smaller radii, not meant to be realistic, to explore the
  effect of smaller sizes on V_iso);
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
- Substrates with diameter 3 µm (mean r 1.5 µm; not realistic, to explore
  smaller soma and thicker axons; user's decisions 2026-10-01, configs in
  `substrate_configs/`, seed 123): `cylinders_d3um_fixed.toml` ->
  cylinders_1200_fixed_r1p5_gap0p1_periodic (168 µm tile, area fraction
  0.300); `cylinders_d3um_gamma.toml` ->
  cylinders_1200_gamma_shape4_scale0p375_gap0p1_periodic (template shape,
  r 0.45-4.5 µm, 188 µm tile, 0.298, r 1.52 +- 0.69);
  `spheres_d3um_vf0p40.toml` -> spheres_1335_gamma_shape5p76_scale0p26_
  gap0p25_periodic (template shape, r 0.5-3 µm, gap 0.25, 40 µm tile,
  1335/1340 placed, volume fraction 0.398, r 1.47 +- 0.54, 5.2M faces).
  Volume fraction 0.40 needs `placement_order = "largest_first"` (option in
  `common.place_objects`, default "drawn"): in drawn order placement stalls
  at 0.31-0.36 and skips large spheres (placed mean r 1.31-1.34 vs 1.47
  drawn); largest first places all of them, keeping the distribution.
- Myelin (future): the cylinders have no myelin (all space outside is free
  water). Later, myelinated axons could be modelled as an excluded ring
  between an inner (axon) and outer (fibre) radius, g-ratio ~0.6-0.7, with
  both radii in the objects file. Revisit the intra-axonal area fraction
  (0.30 now) then.
- CATERPillar (https://github.com/Mic-map/CATERPillar; paper bioRxiv
  2025.06.20.660694). STATUS (2026-09-30, paused to run the V_iso
  simulations; resume here): done: findings below, folder layout and
  `caterpillar.py` taking the CSV + options (commit 4ac1700), cylinders using
  CATERPillar's validated axon radii, JSON templates
  `substrate_configs/caterpillar/wm_mouse_template.json` and
  `gm_mouse_template.json` (2026-10-01; keys checked against the code).
  Next: (1) the converter work listed below (objects file with both-surface
  myelin, then the central start region); (2) a first small run (WM, 50 µm
  voxel) to check CATERPillar's run time and the mesh size.
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
  - WM fractions (user's decision, 2026-10-01): AxonsWithMyelinICVF 55,
    AxonsICVF 9 (total ~64%). Sources: Papazoglou et al. 2024, NMR Biomed
    (PMC11475374): EM of ex-vivo perfusion-fixed mouse corpus callosum and
    fornix, myelinated (inner) axon volume fraction ~0.35 in controls,
    unmyelinated ~30-33% of the axon volume, citing Jelescu et al. 2016
    NeuroImage 132 (mouse, 30%) and Abdollahzadeh et al. 2019 Sci Rep
    (PMC6465365; SBEM of rat corpus callosum, 33%), so total axonal ~0.5.
    With Lee et al.'s myelin thickness (g ~0.56-0.62 for our radii) the
    fibre volume is ~2.6x the inner volume, so inner 0.35 + 0.15
    unmyelinated would exceed 100% (real tissue probably has a larger
    aggregate g ~0.7), and CATERPillar reaches ~70% in practice: the
    templates keep the ~30% unmyelinated proportion and Lee's g-ratio and
    scale the total (inner ~0.21 + myelin ~0.34 + unmyelinated 0.09;
    extra-axonal ~36%, above the < 5-15% of fixed tissue; accepted, the
    substrate need not be exactly realistic). The "~90% myelinated" figure
    found earlier comes from Riise & Pakkenberg 2011, a HUMAN corpus
    callosum study: not used. CATERPillar draws unmyelinated axons from the
    same radius gamma (real ones are thinner, ~0.2-0.3 µm diameter).
  - GM template (approximation): voxel 100 µm, unmyelinated axons 25%
    (FODF_c2 0.33, alpha 4, beta 0.06 µm, MinRadius 0.15), glia pop 1 as
    neuron somas (r 6.4 +- 1.5 µm, soma 10%, processes as dendrites 20%,
    length 40 +- 10 µm, 6 primary), pop 2 protoplasmic astrocytes (r 4 +-
    0.5 µm, soma 2%, processes 5%, length 25 +- 5 µm); total ~62%. WM
    template glia: fibrous astrocytes (pop 1, soma 1%, processes 2%, r 4.5
    +- 0.5 µm, length 25 +- 5 µm); pop 2 off. Glia process lengths and
    numbers are rough choices, not from a specific source.
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
- Frequency models are OPTIONAL since 2026-10-01 (user's decision; we are
  not choosing between linear / square root / squared any more):
  `--frequency-models` in fit_powder_average / fit_tensor / fit_viso, off by
  default (no `*_frequency_fit.csv` then). Planned instead (note only, fit
  stability to be explored first): a power law D(f) = D_0Hz + Lambda (f -
  f_min)^theta, f_min the frequency closest to 0 (likewise for K, V). With 3
  frequencies its 3 parameters fit exactly (no residual), so it needs more
  frequencies or constraints (e.g. fixed theta, pooled fits) to be stable.
- `fit_viso.py --b-max` (default 10: all b) fits only b <= b_max; the
  value is the `BMax` column of viso_fit.csv, and aggregate_seeds.py
  refuses to mix b_max values (or orders).
- Cumulant order: `--order 2-5` (fit_powder_average, fit_viso; k4 = 24F,
  k5 = -120G); 4 and 5 are exploratory (user's request). First look
  (template substrates, one seed, intra): orders 4 and 5 agree within ~2%,
  V shifts a further 8-20% from order 3 (cylinders LTE0 V 0.086 -> 0.093,
  spheres TDE50 0.0107 -> 0.0128), D barely changes; check with the seeds.
- Frequency-model choice (with --frequency-models): SSE by default; `--model-criterion aic` (all
  three analysis scripts, `analysis_utils.fit_frequency_models`) uses AIC,
  and the frequency-fit CSVs always give SSE, AIC and Akaike weights (added
  2026-10-01 at the user's request). With 3 frequencies and three
  2-parameter models AIC ranks exactly as SSE, AICc is undefined
  (n - k - 1 = 0) and the weights are overconfident (e.g. 0.998 for one
  model). AIC needs more data points (frequencies) to be meaningful, e.g.
  >= 4-5 frequencies and models with different numbers of parameters (such
  as a power law a + b f^p). Plots show models up to 150 Hz.
- V_aniso(f) = V_LTE(f) - V_iso(f) must be included in the next report
  (user's note, 2026-10-01).
- Order 3 is the default of `fit_powder_average.py` and `fit_viso.py` since
  2026-10-01 (order 2 biases D, K, V where K is large: cylinders intra LTE
  D 0.300 at order 2 vs 0.331-0.340 at order 3, matching STE 0.339 and
  (D_ax + 2 D_rad)/3 ~0.335). The signal plot shows D, K, V and k3 (the
  skewness, which blows up when V ~ 0, is only in the CSV).
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
- Diffusion time / spectrum: the TDE waveforms last 139.2 ms, the
  LTE/STEiso/STEaniso waveforms 58.16 ms. The user's view (2026-10-01): the
  full diffusion (dephasing) spectrum of each waveform, not the encoding
  time or a single centroid, should describe the signal; explore this idea
  in the literature. The padded STE waveforms make the STE/TDE durations
  equal in the follow-up runs.

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

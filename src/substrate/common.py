"""
Shared helpers for the substrate generator scripts.

Conventions used by all generators:
- Parameters are read from a TOML config file given on the command line
  (see load_params(), load_packing_params() and the templates in
  substrate_configs/).
- All lengths are in micrometers (µm). Meshes are built in µm and
  converted to meters only when saved, as the simulation expects.
- Common parameter names:
    name                      Substrate name (optional, built from parameters if omitted)
    seed                      Random seed (optional, random if omitted; always recorded)
    n_objects                 Number of objects to place
    domain_size               Side length of the placement area/volume (µm)
    min_gap                   Minimum gap between object surfaces (µm)
    radius_distribution       "gamma" or "fixed"
    gamma_shape, gamma_scale  Gamma distribution parameters (µm for scale)
    radius_min, radius_max    Gamma radii outside this range are redrawn (µm)
    radius_fixed              Radius used when radius_distribution = "fixed" (µm)
    max_attempts              Random positions tried per object before skipping it
    max_consecutive_failures  Stop placing after this many objects in a row fail
    periodic                  Build a periodic tile (optional, default false, see tile_periodic())

Outputs of save_substrate(), in substrate/<name>/:
    <name>_vertices.csv, <name>_faces.csv   Mesh in meters (simulation input)
    <name>_mesh.png                         3D preview (µm)
    <name>_radii.png                        Radius histogram (if radii are given)
    <name>_cross_section.png                2D cross-section (cylinders only)
    <name>_params.json                      Input parameters, achieved results and provenance
    <name>_objects.csv                      Object centers and radii in meters, in the frame
                                            of the mesh CSVs (if objects are given)
"""

import os
import sys
import json
import argparse
import tomllib
import subprocess
from datetime import datetime

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
from trimesh.intersections import slice_mesh_plane

# Optional keys of the random packing generators (spheres.py, cylinders.py)
PACKING_DEFAULTS = {
    "name": None,
    "seed": None,
    "max_attempts": 2000,
    "max_consecutive_failures": 100,
    "periodic": False,
}
RADIUS_KEYS = {
    "gamma": ["gamma_shape", "gamma_scale", "radius_min", "radius_max"],
    "fixed": ["radius_fixed"],
}


def read_config(description):
    """Read the substrate config TOML file given on the command line. Returns (config, config_path)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="Substrate config file (.toml), e.g. substrate_configs/<name>.toml")
    config_path = parser.parse_args().config

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config, config_path


def check_keys(config, config_path, required, optional):
    """Raise an error for missing required keys or unknown keys (e.g. typos)."""
    unknown = sorted(set(config) - set(required) - set(optional))
    if unknown:
        raise ValueError(f"{config_path}: unknown parameter(s) {unknown}")

    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_path}: missing parameter(s) {missing}")


def load_params(description, required, defaults):
    """
    Read and check the config given on the command line. `defaults` holds the
    optional keys and their default values. Returns (params, config_path).
    """
    config, config_path = read_config(description)
    check_keys(config, config_path, required, defaults)
    return {**defaults, **config}, config_path


def load_packing_params(description, required, optional=()):
    """
    Read and check the config of a random packing generator.

    `required` lists the keys the script needs besides the radius keys, which
    depend on radius_distribution (see RADIUS_KEYS). `optional` lists other
    keys the script accepts (None if not in the file). Optional keys not in
    the file take their value from PACKING_DEFAULTS. If no seed is given, a
    random one is generated.

    Returns (params, config_path).
    """
    config, config_path = read_config(description)

    distribution = config.get("radius_distribution")
    if distribution not in RADIUS_KEYS:
        raise ValueError(f"{config_path}: radius_distribution must be one of {list(RADIUS_KEYS)}, got {distribution!r}")

    # Radius keys of the other distribution are allowed (and ignored), so a
    # config can keep both sets of values
    all_radius_keys = [key for keys in RADIUS_KEYS.values() for key in keys]
    check_keys(config, config_path,
               required=list(required) + RADIUS_KEYS[distribution],
               optional=list(PACKING_DEFAULTS) + all_radius_keys + list(optional))

    if distribution == "gamma" and config["radius_min"] >= config["radius_max"]:
        raise ValueError(f"{config_path}: radius_min must be smaller than radius_max")

    params = {**PACKING_DEFAULTS, **{key: None for key in optional}, **config}

    if params["seed"] is None:
        params["seed"] = int(np.random.default_rng().integers(2**32))
        print(f"No seed given, using random seed {params['seed']}")

    return params, config_path


def sample_radii(rng, n, params):
    """Draw n radii (µm) according to params["radius_distribution"]."""
    distribution = params["radius_distribution"]

    if distribution == "fixed":
        return np.full(n, float(params["radius_fixed"]))

    if distribution == "gamma":
        # Truncated gamma: redraw any radius outside [radius_min, radius_max]
        radius_min = params["radius_min"]
        radius_max = params["radius_max"]
        radii = np.empty(0)
        while len(radii) < n:
            draws = rng.gamma(params["gamma_shape"], params["gamma_scale"], size=n)
            draws = draws[(draws >= radius_min) & (draws <= radius_max)]
            radii = np.concatenate([radii, draws])
        return radii[:n]

    raise ValueError(f"Unknown radius_distribution '{distribution}', use 'gamma' or 'fixed'")


def place_objects(rng, radii, dim, params, batch_size=100):
    """
    Randomly place non-overlapping circles (dim=2) or spheres (dim=3) with the
    given radii inside a box of side params["domain_size"] centered at 0.

    Two objects overlap if the distance between their centers is less than
    the sum of their radii plus params["min_gap"]. With params["periodic"],
    the distance is the minimum-image distance of the periodic box (the box
    repeats in every direction), so objects near opposite faces cannot
    overlap across the boundary. Objects that cannot be
    placed within params["max_attempts"] tries are skipped. Placement stops
    early after params["max_consecutive_failures"] skipped objects in a row,
    or on Ctrl+C, keeping the objects placed so far.

    Returns (centers, placed_radii) as arrays of shape (k, dim) and (k,).
    """
    size = params["domain_size"]
    half = size / 2
    periodic = params["periodic"]
    min_gap = params["min_gap"]
    max_attempts = params["max_attempts"]
    max_consecutive_failures = params["max_consecutive_failures"]

    centers = np.empty((len(radii), dim))
    placed_radii = np.empty(len(radii))
    n_placed = 0
    n_skipped = 0
    consecutive_failures = 0

    try:
        for i, r in enumerate(radii):
            placed = False
            attempts = 0

            while not placed and attempts < max_attempts:
                # Test a batch of candidate positions at once against all placed objects
                n_candidates = min(batch_size, max_attempts - attempts)
                candidates = rng.uniform(-half, half, size=(n_candidates, dim))
                attempts += n_candidates

                differences = candidates[:, None, :] - centers[None, :n_placed, :]
                if periodic:
                    differences -= size * np.round(differences / size)
                distances = np.linalg.norm(differences, axis=2)
                free = np.all(distances >= r + placed_radii[:n_placed] + min_gap, axis=1)

                if free.any():
                    centers[n_placed] = candidates[np.argmax(free)]
                    placed_radii[n_placed] = r
                    n_placed += 1
                    placed = True

            if placed:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                n_skipped += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Stopping: {max_consecutive_failures} objects in a row could not be placed")
                    break

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(radii)} objects, placed {n_placed}, skipped {n_skipped}")

    except KeyboardInterrupt:
        print("Manual stop. Keeping the objects placed so far.")

    print(f"Placed {n_placed} of {len(radii)} objects "
          f"({n_skipped} skipped after {max_attempts} attempts without a free position)")
    return centers[:n_placed], placed_radii[:n_placed]


def tile_periodic(objects, centers, radii, domain_size, axes):
    """
    Build a periodic tile of side domain_size centered at 0 from object meshes
    (each already at its center). Objects that cross a face of the tile along
    one of `axes` get shifted copies (by ±domain_size, up to 8 for a corner),
    then everything is cut at the six faces without closing the cuts, so the
    surfaces are open at the faces and continue in the next tile. The bounding
    box of the result is then the tile, which is what the simulation uses as
    the periodic voxel (checked). Objects along an axis not in `axes` (e.g.
    cylinders along z) must be longer than the tile, so they are cut at both
    faces. Returns the tile mesh (µm).
    """
    half = domain_size / 2
    inside, crossing = [], []
    for mesh, center, r in zip(objects, centers, radii):
        shifts = []
        for axis in range(3):
            axis_shifts = [0.0]
            if axis in axes:
                if center[axis] + r > half:
                    axis_shifts.append(-domain_size)
                if center[axis] - r < -half:
                    axis_shifts.append(domain_size)
            shifts.append(axis_shifts)
        if all(len(s) == 1 for s in shifts) and all(axis in axes for axis in range(3)):
            inside.append(mesh)
            continue
        for shift in itertools.product(*shifts):
            copy = mesh.copy()
            copy.apply_translation(shift)
            crossing.append(copy)

    # Cut the objects that cross the faces (the others are inside the tile)
    cut = trimesh.util.concatenate(crossing)
    for axis in range(3):
        for sign in [1, -1]:
            normal = np.zeros(3)
            normal[axis] = -sign                  # keep the side towards the center
            origin = np.zeros(3)
            origin[axis] = sign * half
            cut = slice_mesh_plane(cut, plane_normal=normal, plane_origin=origin, cap=False)
    tile = trimesh.util.concatenate(inside + [cut])
    tile.merge_vertices()                         # the cuts leave duplicate vertices

    extent = tile.bounds
    if not np.allclose(extent, [[-half] * 3, [half] * 3], rtol=0, atol=1e-6 * domain_size):
        raise ValueError(f"The periodic tile does not reach every face of the box: mesh bounds "
                         f"{extent.tolist()}, box ±{half}. Every face must be crossed by an object; "
                         f"use more objects or another seed.")
    return tile


def format_value(value):
    """Format a number for use in a substrate name, e.g. 1.5 -> '1p5', 2.0 -> '2'."""
    return f"{value:g}".replace(".", "p")


def radius_name(params):
    """Name fragment describing the radius distribution."""
    if params["radius_distribution"] == "fixed":
        return f"fixed_r{format_value(params['radius_fixed'])}"
    return f"gamma_shape{format_value(params['gamma_shape'])}_scale{format_value(params['gamma_scale'])}"


def git_commit():
    """Current git commit of the project, marked '-dirty' if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return None


def to_json(value):
    """Convert numpy types so they can be written to JSON."""
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def plot_mesh(mesh, path, title, equal_aspect=True, elev=None, azim=None):
    """
    Save a 3D preview of a mesh (in µm), with equal axis scaling unless
    equal_aspect=False. elev and azim set the viewing angles in degrees
    (matplotlib defaults if None).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )

    if equal_aspect:
        ax.set_box_aspect(np.ptp(mesh.vertices, axis=0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_title(title)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_section(centers, radii, domain_size, path, title, periodic=False):
    """
    Save a 2D view of circles (e.g. a cylinder cross-section) with the placement
    domain. With periodic=True, the circles are also drawn shifted by the domain
    size, and the view is limited to the domain (the periodic tile).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    shifts = [-domain_size, 0, domain_size] if periodic else [0]
    for (x, y), r in zip(centers, radii):
        for dx in shifts:
            for dy in shifts:
                ax.add_patch(plt.Circle((x + dx, y + dy), r, color="tab:blue"))

    half = domain_size / 2
    label = "Periodic tile" if periodic else "Placement domain"
    ax.add_patch(plt.Rectangle((-half, -half), domain_size, domain_size,
                               fill=False, linestyle="--", color="black", label=label))

    ax.set_aspect("equal")
    if periodic:
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    else:
        ax.autoscale_view()
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radii(radii, path, title):
    """Save a histogram of the placed radii (µm)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(radii, bins=50)
    ax.axvline(np.mean(radii), linestyle="--", linewidth=2, color="black",
               label=f"Mean = {np.mean(radii):.3f} µm")
    ax.set_xlabel("Radius (µm)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def save_substrate(mesh, name, params, results, title, config_path, radii=None, equal_aspect=True,
                   objects=None):
    """
    Save a substrate built in µm to substrate/<name>/: mesh CSVs (in meters),
    preview images, and a params JSON with inputs, results and provenance.
    If `radii` is given, a radius histogram and radius statistics are added.
    If `objects` (a DataFrame of object centers and radii in µm) is given, it
    is saved in meters as <name>_objects.csv, e.g. to sample walker positions
    inside or outside the objects. Returns the output directory.
    """
    output_dir = f"substrate/{name}"
    os.makedirs(output_dir, exist_ok=True)

    plot_mesh(mesh, f"{output_dir}/{name}_mesh.png", title, equal_aspect)

    # Convert from µm to meters for the simulation
    mesh_m = mesh.copy()
    mesh_m.apply_scale(1e-6)

    pd.DataFrame(mesh_m.vertices, columns=["x", "y", "z"]).to_csv(
        f"{output_dir}/{name}_vertices.csv", index=False
    )
    pd.DataFrame(mesh_m.faces, columns=["v1", "v2", "v3"]).to_csv(
        f"{output_dir}/{name}_faces.csv", index=False
    )

    if objects is not None:
        (objects * 1e-6).to_csv(f"{output_dir}/{name}_objects.csv", index=False)

    results = {**results, "mesh_watertight": bool(mesh.is_watertight)}

    if radii is not None:
        plot_radii(radii, f"{output_dir}/{name}_radii.png", f"{title}: radius distribution")
        results.update({
            "radius_mean": float(np.mean(radii)),
            "radius_std": float(np.std(radii)),
            "radius_min": float(np.min(radii)),
            "radius_max": float(np.max(radii)),
        })

    record = {
        "substrate_name": name,
        "script": os.path.basename(sys.argv[0]),
        "config_file": config_path,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "units": "Parameters and results in µm; mesh CSVs in meters",
        "params": params,
        "results": results,
    }

    with open(f"{output_dir}/{name}_params.json", "w", encoding="utf-8") as f:
        json.dump(to_json(record), f, indent=4, ensure_ascii=False)

    print(f"\nSaved substrate to {output_dir}/")
    for key, value in record["results"].items():
        print(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")

    return output_dir

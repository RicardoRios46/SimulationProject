"""
Randomly packed, non-overlapping parallel cylinders along z in a square domain.
With periodic = true, the substrate is a periodic tile of side domain_size in
x, y and z (see common.tile_periodic): the cylinders are open tubes cut at the
z faces, so they continue in the next tile, and cylinder_length is not used.

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/cylinders.py substrate_configs/<config>.toml

Start from substrate_configs/cylinders_template.toml. Outputs go to
substrate/<name>/ (see common.py), including <name>_params.json with the
parameters used.
"""

import numpy as np
import trimesh

import pandas as pd

from common import (load_packing_params, sample_radii, place_objects, tile_periodic, radius_name,
                    format_value, save_substrate, plot_cross_section)

params, config_path = load_packing_params(
    "Generate a substrate of randomly packed parallel cylinders.",
    required=["n_objects", "domain_size", "min_gap", "radius_distribution"],
    optional=["cylinder_length"],
)
periodic = params["periodic"]
if periodic and params["cylinder_length"] is not None:
    raise ValueError(f"{config_path}: cylinder_length is not used with periodic = true "
                     f"(the z period is domain_size); remove it")
if not periodic and params["cylinder_length"] is None:
    raise ValueError(f"{config_path}: missing parameter cylinder_length (needed with periodic = false)")

rng = np.random.default_rng(params["seed"])

radii = sample_radii(rng, params["n_objects"], params)
centers, radii = place_objects(rng, radii, dim=2, params=params)

# Periodic tiles: the cylinders are made longer than the tile and cut at the z
# faces (open tubes without end caps)
length = 1.1 * params["domain_size"] if periodic else params["cylinder_length"]
cylinders = []
for (x, y), r in zip(centers, radii):
    cylinder = trimesh.creation.cylinder(radius=r, height=length)
    cylinder.apply_translation([x, y, 0])
    cylinders.append(cylinder)

if periodic:
    centers_3d = np.column_stack([centers, np.zeros(len(centers))])
    mesh = tile_periodic(cylinders, centers_3d, radii, params["domain_size"], axes=[0, 1])
else:
    mesh = trimesh.util.concatenate(cylinders)

name = params["name"] or (f"cylinders_{len(radii)}_{radius_name(params)}_gap{format_value(params['min_gap'])}"
                          + ("_periodic" if periodic else ""))

results = {
    "n_placed": len(radii),
    "area_fraction": float(np.sum(np.pi * radii**2) / params["domain_size"] ** 2),
}

# Non-periodic cylinders are much longer than the domain is wide, so the 3D
# preview is not scaled equally; a cross-section shows the packing. The objects
# file has the cylinder axes (along z) and radii.
objects = pd.DataFrame({"x": centers[:, 0], "y": centers[:, 1], "radius": radii})
output_dir = save_substrate(mesh, name, params, results, "Random non-overlapping cylinders",
                            config_path, radii=radii, equal_aspect=periodic, objects=objects)
plot_cross_section(centers, radii, params["domain_size"], f"{output_dir}/{name}_cross_section.png",
                   "Random non-overlapping cylinders: cross-section", periodic=periodic)

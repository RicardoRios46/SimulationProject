"""
Randomly packed, non-overlapping parallel cylinders along z in a square domain.

Edit `params` below, then run from the project root:
    pixi run -e trimesh-env python src/substrate/cylinders.py

Outputs go to substrate/<name>/ (see common.py), including <name>_params.json
with the parameters used. The name is built from the parameters and the
number of cylinders actually placed, unless params["name"] is set.
"""

import numpy as np
import trimesh

from common import sample_radii, place_objects, radius_name, format_value, save_substrate, plot_cross_section

# All lengths in µm
params = {
    "name": None,                     # None = build name from parameters
    "seed": 123,
    "n_objects": 3000,                # Number of cylinders to place
    "domain_size": 120,               # Side of the square (xy) cylinder axes are placed in
    "min_gap": 0.45,                  # Minimum gap between cylinder surfaces
    "cylinder_length": 1000,          # Length along z
    "radius_distribution": "gamma",   # "gamma" or "fixed"
    "gamma_shape": 0.75,
    "gamma_scale": 0.55,
    "radius_min": 0.2,
    "radius_max": 3,
    "radius_fixed": 0.5,              # Only used if radius_distribution = "fixed"
    "max_attempts": 2000,
    "max_consecutive_failures": 100,
}

rng = np.random.default_rng(params["seed"])

radii = sample_radii(rng, params["n_objects"], params)
centers, radii = place_objects(rng, radii, dim=2, params=params)

cylinders = []
for (x, y), r in zip(centers, radii):
    cylinder = trimesh.creation.cylinder(radius=r, height=params["cylinder_length"])
    cylinder.apply_translation([x, y, 0])
    cylinders.append(cylinder)
mesh = trimesh.util.concatenate(cylinders)

name = params["name"] or f"cylinders_{len(radii)}_{radius_name(params)}_gap{format_value(params['min_gap'])}"

results = {
    "n_placed": len(radii),
    "area_fraction": float(np.sum(np.pi * radii**2) / params["domain_size"] ** 2),
}

# Cylinders are much longer than the domain is wide, so the 3D preview is not
# scaled equally, and a cross-section shows the packing
output_dir = save_substrate(mesh, name, params, results, radii, "Random non-overlapping cylinders", equal_aspect=False)
plot_cross_section(centers, radii, params["domain_size"], f"{output_dir}/{name}_cross_section.png",
                   "Random non-overlapping cylinders: cross-section")

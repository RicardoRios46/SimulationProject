"""
Randomly packed, non-overlapping spheres in a cubic domain.

Edit `params` below, then run from the project root:
    pixi run -e trimesh-env python src/substrate/spheres.py

Outputs go to substrate/<name>/ (see common.py), including <name>_params.json
with the parameters used. The name is built from the parameters and the
number of spheres actually placed, unless params["name"] is set.
"""

import numpy as np
import trimesh

from common import sample_radii, place_objects, radius_name, format_value, save_substrate

# All lengths in µm
params = {
    "name": None,                     # None = build name from parameters
    "seed": 123,
    "n_objects": 500,                 # Number of spheres to place
    "domain_size": 100,               # Side of the cube sphere centers are placed in
    "min_gap": 1.0,                   # Minimum gap between sphere surfaces
    "radius_distribution": "gamma",   # "gamma" or "fixed"
    "gamma_shape": 2,
    "gamma_scale": 1.5,
    "radius_min": 1,
    "radius_max": 12,
    "radius_fixed": 5,                # Only used if radius_distribution = "fixed"
    "max_attempts": 2000,
    "max_consecutive_failures": 100,
}

rng = np.random.default_rng(params["seed"])

radii = sample_radii(rng, params["n_objects"], params)
centers, radii = place_objects(rng, radii, dim=3, params=params)

spheres = []
for center, r in zip(centers, radii):
    sphere = trimesh.creation.uv_sphere(radius=r)
    sphere.apply_translation(center)
    spheres.append(sphere)
mesh = trimesh.util.concatenate(spheres)

name = params["name"] or f"spheres_{len(radii)}_{radius_name(params)}_gap{format_value(params['min_gap'])}"

results = {
    "n_placed": len(radii),
    "volume_fraction": float(np.sum(4 / 3 * np.pi * radii**3) / params["domain_size"] ** 3),
}

save_substrate(mesh, name, params, results, radii, "Random non-overlapping spheres")

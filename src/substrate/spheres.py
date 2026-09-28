"""
Randomly packed, non-overlapping spheres in a cubic domain.

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/spheres.py substrate_configs/<config>.toml

Start from substrate_configs/spheres_template.toml. Outputs go to
substrate/<name>/ (see common.py), including <name>_params.json with the
parameters used.
"""

import numpy as np
import trimesh

from common import load_params, sample_radii, place_objects, radius_name, format_value, save_substrate

params, config_path = load_params(
    "Generate a substrate of randomly packed spheres.",
    required=["n_objects", "domain_size", "min_gap", "radius_distribution"],
)

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

save_substrate(mesh, name, params, results, radii, "Random non-overlapping spheres", config_path)

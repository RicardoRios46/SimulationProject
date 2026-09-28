"""
Randomly packed, non-overlapping parallel cylinders along z in a square domain.

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/cylinders.py substrate_configs/<config>.toml

Start from substrate_configs/cylinders_template.toml. Outputs go to
substrate/<name>/ (see common.py), including <name>_params.json with the
parameters used.
"""

import numpy as np
import trimesh

from common import load_params, sample_radii, place_objects, radius_name, format_value, save_substrate, plot_cross_section

params, config_path = load_params(
    "Generate a substrate of randomly packed parallel cylinders.",
    required=["n_objects", "domain_size", "min_gap", "cylinder_length", "radius_distribution"],
)

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
output_dir = save_substrate(mesh, name, params, results, radii, "Random non-overlapping cylinders",
                            config_path, equal_aspect=False)
plot_cross_section(centers, radii, params["domain_size"], f"{output_dir}/{name}_cross_section.png",
                   "Random non-overlapping cylinders: cross-section")

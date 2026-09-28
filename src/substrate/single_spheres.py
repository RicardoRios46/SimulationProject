"""
Single spheres, one substrate per radius, e.g. for validating the simulation
against the analytical signal of restricted diffusion in a sphere.

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/single_spheres.py substrate_configs/<config>.toml

Start from substrate_configs/single_spheres_template.toml. Each sphere is
saved to substrate/<name_prefix>_r<radius>/ (see common.py), e.g.
substrate/sphere_r2p5/, including <name>_params.json with the parameters used.
"""

import numpy as np
import trimesh

from common import load_params, format_value, save_substrate

params, config_path = load_params(
    "Generate one single-sphere substrate per radius.",
    required=["radii"],
    defaults={"name_prefix": "sphere"},
)

if not params["radii"] or any(r <= 0 for r in params["radii"]):
    raise ValueError(f"{config_path}: radii must be a non-empty list of positive values")

for radius in params["radii"]:
    mesh = trimesh.creation.uv_sphere(radius=radius)

    name = f"{params['name_prefix']}_r{format_value(radius)}"

    sphere_volume = 4 / 3 * np.pi * radius**3
    results = {
        "radius": radius,
        "sphere_volume": sphere_volume,
        "mesh_volume": float(mesh.volume),
        "mesh_volume_error": float(mesh.volume / sphere_volume - 1),
    }

    save_substrate(mesh, name, params, results, f"Single sphere, radius {radius:g} µm", config_path)

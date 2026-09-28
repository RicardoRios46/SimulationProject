"""
Convert a CATERPillar output file into a substrate mesh.

CATERPillar describes each cell (e.g. an axon) as a chain of overlapping
spheres, one per row: cell_type cell_id component component_id X Y Z
inner_radius outer_radius (lengths in µm). The spheres of each cell are
merged into one surface, so every cell stays a separate compartment, and
the cells are then combined into one substrate.

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/caterpillar.py substrate_configs/<config>.toml

Start from substrate_configs/caterpillar_template.toml. Outputs go to
substrate/<name>/ (see common.py), including <name>_params.json with the
parameters used.
"""

import os

import pandas as pd
import trimesh

from common import load_params, save_substrate

RADIUS_COLUMNS = ["inner_radius", "outer_radius"]

params, config_path = load_params(
    "Convert a CATERPillar output file into a substrate mesh.",
    required=["input_file"],
    defaults={
        "name": None,
        "radius_column": "inner_radius",
        "cell_types": None,
        "sphere_subdivisions": 1,
        "smoothing_iterations": 1,
    },
)

if params["radius_column"] not in RADIUS_COLUMNS:
    raise ValueError(f"{config_path}: radius_column must be one of {RADIUS_COLUMNS}, got {params['radius_column']!r}")

data = pd.read_csv(params["input_file"], sep=r"\s+")

if params["cell_types"] is not None:
    data = data[data["cell_type"].isin(params["cell_types"])]
if data.empty:
    raise ValueError(f"No rows left in {params['input_file']} (check cell_types)")

cells = data.groupby(["cell_type", "cell_id"])
print(f"{len(data)} spheres in {len(cells)} cell(s)")

cell_meshes = []
for (cell_type, cell_id), cell in cells:
    spheres = []
    for x, y, z, r in cell[["X", "Y", "Z", params["radius_column"]]].to_numpy():
        sphere = trimesh.creation.icosphere(radius=r, subdivisions=params["sphere_subdivisions"])
        sphere.apply_translation([x, y, z])
        spheres.append(sphere)

    cell_mesh = trimesh.boolean.union(spheres, engine="manifold")
    print(f"  {cell_type} {cell_id}: {len(spheres)} spheres -> {len(cell_mesh.split())} volume(s)")
    cell_meshes.append(cell_mesh)

mesh = trimesh.util.concatenate(cell_meshes)

# Smooths the bumps between spheres; trimesh keeps the total volume constant
if params["smoothing_iterations"] > 0:
    trimesh.smoothing.filter_laplacian(mesh, iterations=params["smoothing_iterations"])

name = params["name"] or f"caterpillar_{os.path.splitext(os.path.basename(params['input_file']))[0]}"

results = {
    "n_cells": len(cells),
    "n_spheres": len(data),
    "n_volumes": len(mesh.split()),
    "mesh_volume": float(mesh.volume),
}

save_substrate(mesh, name, params, results, "CATERPillar substrate", config_path)

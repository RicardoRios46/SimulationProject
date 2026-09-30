"""
Convert a CATERPillar output file into a substrate mesh.

CATERPillar describes each cell (e.g. an axon) as a chain of overlapping
spheres, one per row: cell_type cell_id component component_id X Y Z
inner_radius outer_radius (lengths in µm). The spheres of each cell are
merged into one surface, so every cell stays a separate compartment, and
the cells are then combined into one substrate.

The CATERPillar output is kept with the substrate: everything for one
substrate is in substrate/<name>/, with CATERPillar's config, output CSV and
growth info in substrate/<name>/caterpillar/. Run CATERPillar with
"OutputDirectory": "substrate/caterpillar_<run>/caterpillar" (and
"Filename": "<run>"), then convert that CSV; a CSV from elsewhere is copied
there (with its <run>.json and <run>_growth_info.txt, if next to it).

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/caterpillar.py substrate/caterpillar_<run>/caterpillar/<run>.csv [options]

Options: --name (default caterpillar_<run>), --radius-column, --cell-types,
--sphere-subdivisions, --smoothing-iterations (see --help). Outputs go to
substrate/<name>/ (see common.py), including <name>_params.json with the
options used.
"""

import argparse
import os
import shutil

import pandas as pd
import trimesh

from common import save_substrate

RADIUS_COLUMNS = ["inner_radius", "outer_radius"]

parser = argparse.ArgumentParser(description="Convert a CATERPillar output file into a substrate mesh.")
parser.add_argument("input_file", help="CATERPillar output CSV, e.g. substrate/caterpillar_<run>/caterpillar/<run>.csv")
parser.add_argument("--name", help="Substrate name (default caterpillar_<run>, from the CSV file name)")
parser.add_argument("--radius-column", default="inner_radius", choices=RADIUS_COLUMNS,
                    help="inner_radius (the axon itself, default) or outer_radius (including the myelin sheath)")
parser.add_argument("--cell-types", nargs="+",
                    help="Only use these cell types, e.g. --cell-types axon (default: all rows)")
parser.add_argument("--sphere-subdivisions", type=int, default=1,
                    help="Icosphere resolution of each sphere: 1 = 80 faces, 2 = 320 faces (default 1)")
parser.add_argument("--smoothing-iterations", type=int, default=1,
                    help="Laplacian smoothing iterations after merging, 0 = off (default 1)")
params = vars(parser.parse_args())

run = os.path.splitext(os.path.basename(params["input_file"]))[0]
params["name"] = params["name"] or f"caterpillar_{run}"

#Keep CATERPillar's files with the substrate, in substrate/<name>/caterpillar/
caterpillar_dir = f"substrate/{params['name']}/caterpillar"
source_dir = os.path.dirname(os.path.abspath(params["input_file"]))
if source_dir != os.path.abspath(caterpillar_dir):
    os.makedirs(caterpillar_dir, exist_ok=True)
    for file in [f"{run}.csv", f"{run}.json", f"{run}_growth_info.txt"]:
        if os.path.exists(f"{source_dir}/{file}"):
            shutil.copy(f"{source_dir}/{file}", caterpillar_dir)
            print(f"Copied {file} to {caterpillar_dir}/")
params["input_file"] = f"{caterpillar_dir}/{run}.csv"

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

name = params["name"]

results = {
    "n_cells": len(cells),
    "n_spheres": len(data),
    "n_volumes": len(mesh.split()),
    "mesh_volume": float(mesh.volume),
}

save_substrate(mesh, name, params, results, "CATERPillar substrate", config_path=None)

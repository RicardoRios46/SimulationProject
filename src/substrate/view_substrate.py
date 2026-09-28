"""
Re-plot an existing substrate without regenerating it, e.g. to try other
viewing angles or to get an SVG for a figure.

Usage (from the project root):
    pixi run -e trimesh-env python src/substrate/view_substrate.py <name> [options]

Reads substrate/<name>/<name>_vertices.csv and _faces.csv (in meters) and
plots them in µm. By default the figure is saved to
substrate/<name>/<name>_view.svg.

Examples:
    python src/substrate/view_substrate.py sphere_r2p5
    python src/substrate/view_substrate.py sphere_r2p5 --elev 20 --azim 45 --output figures/sphere.png
    python src/substrate/view_substrate.py cylinders_2840_gamma_shape0p75_scale0p55_gap0p45 --no-equal-aspect
"""

import argparse
import os

import pandas as pd
import trimesh

from common import plot_mesh

parser = argparse.ArgumentParser(description="Re-plot an existing substrate.")
parser.add_argument("name", help="Substrate name (folder in substrate/)")
parser.add_argument("--output", help="Output file; the extension sets the format (default: substrate/<name>/<name>_view.svg)")
parser.add_argument("--title", help="Figure title (default: the substrate name)")
parser.add_argument("--elev", type=float, help="Elevation viewing angle in degrees")
parser.add_argument("--azim", type=float, help="Azimuth viewing angle in degrees")
parser.add_argument("--no-equal-aspect", action="store_true",
                    help="Do not scale axes equally (useful for long cylinders)")
args = parser.parse_args()

substrate_dir = f"substrate/{args.name}"
if not os.path.isdir(substrate_dir):
    raise SystemExit(f"Error: substrate folder '{substrate_dir}' not found")

vertices = pd.read_csv(f"{substrate_dir}/{args.name}_vertices.csv").to_numpy()
faces = pd.read_csv(f"{substrate_dir}/{args.name}_faces.csv").to_numpy()

# Substrates are saved in meters; plot in µm
mesh = trimesh.Trimesh(vertices * 1e6, faces, process=False)

output = args.output or f"{substrate_dir}/{args.name}_view.svg"
os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

plot_mesh(mesh, output, args.title or args.name,
          equal_aspect=not args.no_equal_aspect, elev=args.elev, azim=args.azim)
print(f"Saved {output}")

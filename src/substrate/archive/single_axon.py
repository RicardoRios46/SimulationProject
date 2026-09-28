import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_ba(length, radius_fiber, radius_bead, spacing):
    """
    Returns a beaded axon mesh (trimesh)
    """
    shaft = trimesh.creation.cylinder(radius=radius_fiber, height=length)

    beads = []
    z_positions = np.arange(-length / 2, length / 2, spacing)

    if radius_bead > 0:
        for z in z_positions:
            num = np.random.randint(0, spacing - radius_bead)
            bead = trimesh.creation.uv_sphere(radius=radius_bead)
            bead.apply_translation([0, 0, z + num])

            if -length / 2 < (z + num) < length / 2:
                beads.append(bead)

        all_parts = [shaft] + beads
        beaded_mesh = trimesh.boolean.union(all_parts, "manifold")
    else:
        beaded_mesh = shaft

    return beaded_mesh


axons = create_ba(60, 0.337, 1, 5)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(
    axons.vertices[:, 0],
    axons.vertices[:, 1],
    axons.vertices[:, 2],
    triangles=axons.faces,
    cmap="viridis",
    edgecolor="black",
    linewidth=0.01,
    shade=True,
)

# Compute limits
vertices = axons.vertices
mins = vertices.min(axis=0)
maxs = vertices.max(axis=0)

# Equal limits in all directions (keeps the plotting box cubic)
center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2

padding = 1.0  # μm

ax.set_xlim(center[0] - radius - padding, center[0] + radius + padding)
ax.set_ylim(center[1] - radius - padding, center[1] + radius + padding)
ax.set_zlim(center[2] - radius - padding, center[2] + radius + padding)

# Square plotting box
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")

ax.view_init(elev=20, azim=45)

axons.apply_scale(1e-6)

name = "axon_fiber0.337_bead1.0"

output_dir = f"substrate/{name}"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/{name}_mesh.png", dpi=300, bbox_inches="tight")
plt.close(fig)

data_vertices = pd.DataFrame(axons.vertices, columns=["x", "y", "z"])
data_vertices.to_csv(f"{output_dir}/{name}_vertices.csv", index=False)

data_faces = pd.DataFrame(axons.faces, columns=["v1", "v2", "v3"])
data_faces.to_csv(f"{output_dir}/{name}_faces.csv", index=False)

print("\nSuccess.")
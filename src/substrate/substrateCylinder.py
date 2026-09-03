import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spread = 0.45
n_cylinders =3000
cyl_height = 1000
domain_size = 120
gamma = True
gamma_shape = 0.75
gamma_scale = 0.55
max_radius = 3
min_radius = 0.2
cyl_radius = 0.5

if gamma:
    name = f"cylinders_gamma_shape{gamma_shape}_scale{gamma_scale}_spread{spread}"
else:
    name = f"cylinders_uniform_spread{spread}"
radii = []

cylinder_list = []
placed_positions = []  # (x, y, radius)


def is_overlapping(x, y, radius, existing_positions):
    for ex, ey, eradius in existing_positions:
        dist = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        if dist < (radius + eradius + spread):
            return True
    return False


consecutive = 0
try:
    for i in range(n_cylinders):
        if gamma:
            cyl_radius = np.random.gamma(gamma_shape, gamma_scale)
            while cyl_radius < min_radius or cyl_radius > max_radius:
                cyl_radius = np.random.gamma(gamma_shape, gamma_scale)
        else:
                cyl_radius = cyl_radius

        print(cyl_radius)
        placed = False
        attempts = 0
        max_attempts = 2000

        while not placed and attempts < max_attempts:

            x_pos = np.random.uniform(-domain_size / 2, domain_size / 2)
            y_pos = np.random.uniform(-domain_size / 2, domain_size / 2)

            if not is_overlapping(x_pos, y_pos, cyl_radius, placed_positions):

                cyl = trimesh.creation.cylinder(
                    radius=cyl_radius,
                    height=cyl_height
                )
                cyl.apply_translation([x_pos, y_pos, 0])

                cylinder_list.append(cyl)
                placed_positions.append((x_pos, y_pos, cyl_radius))
                placed = True
                print(f"placed cyl {i+1}")
                radii.append(cyl_radius)

            attempts += 1

        if not placed:
            consecutive += 1
            print(f"Warning: Could not place cylinder {i + 1}")
        else:
            consecutive = 0

        if consecutive == 100:
            print("Max consecutive failed attemps")
            break
except:
    print("Manual Stop. Finishing.")

print(f"Placed {len(cylinder_list)} cylinders.")

output_dir = f"substrate/{len(cylinder_list)}_{name}"
os.makedirs(output_dir, exist_ok=True)

combined_mesh = trimesh.util.concatenate(cylinder_list)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(
    combined_mesh.vertices[:, 0],
    combined_mesh.vertices[:, 1],
    combined_mesh.vertices[:, 2],
    triangles=combined_mesh.faces,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9,
)

ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")

plt.savefig(
    f"{output_dir}/substrate_mesh.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

combined_mesh.apply_scale(1e-6)

data_vertices = pd.DataFrame(
    combined_mesh.vertices,
    columns=["x", "y", "z"]
)
data_vertices.to_csv(
    f"{output_dir}/{len(cylinder_list)}_{name}_vertices.csv",
    index=False
)

data_faces = pd.DataFrame(
    combined_mesh.faces,
    columns=["v1", "v2", "v3"]
)
data_faces.to_csv(
    f"{output_dir}/{len(cylinder_list)}_{name}_faces.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(radii, bins=50, density=True)
ax.axvline(np.mean(radii), linestyle="--", linewidth=2, label=f"Mean = {np.mean(radii):.3f} µm")
ax.set_xlabel("Radius (µm)")
ax.set_ylabel("Amount")
ax.set_title("Cylinder Radius Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/radius_distribution.png", dpi=300)
plt.close(fig)

print("Mean radius:", np.mean(radii))
print("Maximum radius:", np.max(radii))
print("Minimum radius:", np.min(radii))

print("Done.")
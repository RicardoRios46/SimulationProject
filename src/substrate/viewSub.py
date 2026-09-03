import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

substrate_file = "axon_fiber0.337_bead1.0"

v_file = f"substrate/{substrate_file}/{substrate_file}_vertices.csv"
f_file = f"substrate/{substrate_file}/{substrate_file}_faces.csv"
output_png = f"substrate/{substrate_file}/{substrate_file}_views.svg"

vertices = np.loadtxt(v_file, delimiter=",",skiprows=1)
faces = np.loadtxt(f_file, delimiter=",",skiprows=1)

fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(
    vertices[:, 0], vertices[:, 1], vertices[:, 2], 
    triangles=faces, 
    cmap='viridis', 
    edgecolor='#88888822', 
    linewidth=0.1, 
    alpha=0.9
)

max_range = np.array([
    vertices[:, 0].max() - vertices[:, 0].min(),
    vertices[:, 1].max() - vertices[:, 1].min(),
    vertices[:, 2].max() - vertices[:, 2].min()
]).max() / 2.0

mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Clean labels
ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")
ax.set_title("Beaded Axon Mesh Substrate")

plt.savefig(output_png, bbox_inches='tight')
print(f"done, file saved to {output_png}")
plt.close()

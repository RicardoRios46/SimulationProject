import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for radius in np.arange(0.5,10.1,0.5):

    mesh = trimesh.creation.uv_sphere(radius=radius)

    name = f"sphereRadius_{radius}"
    output_dir = f"substrate/{name}"
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        mesh.vertices[:, 0], 
        mesh.vertices[:, 1], 
        mesh.vertices[:, 2], 
        triangles=mesh.faces, 
        cmap='viridis', 
        edgecolor='#88888822', 
        linewidth=0.1, 
        alpha=0.9
    )

    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_zlabel("Z (μm)")

    mesh.apply_scale(1e-6)

    ax.axis('equal')
    plt.savefig(f'{output_dir}/{name}_mesh.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    data_vertices = pd.DataFrame(mesh.vertices, columns=['x', 'y', 'z'])
    data_vertices.to_csv(f'{output_dir}/{name}_vertices.csv', index=False)

    data_faces = pd.DataFrame(mesh.faces, columns=['v1', 'v2', 'v3'])
    data_faces.to_csv(f'{output_dir}/{name}_faces.csv', index=False)

    print("\nSucess.")
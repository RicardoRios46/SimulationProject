import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os

#Name mesh here
name = 'CATERPIllar_single_axon_run3'

#make new directory for all outputs
os.makedirs(f'substrate/{name}', exist_ok=True)


#Create mesh here
meshs = []

with open("CATERPillar_inputs/single_axon_run3.csv") as f:
    next(f)  # Skip the header line

    for line in f:
        vals = line.strip().split()
        if not vals:
            continue

        x = float(vals[4])
        y = float(vals[5])
        z = float(vals[6])
        r = float(vals[7])

        # Create the sphere primitive
        mesh = trimesh.creation.icosphere(radius=r,subdivisions=1)

        # Apply the translation in-place
        mesh.apply_translation([x, y, z])
        meshs.append(mesh)

print(f"Number of spheres: {len(meshs)}")
mesh = trimesh.boolean.union(meshs, engine="manifold")
independent_volumes = mesh.split()
num_volumes = len(independent_volumes)
print(f"Union completed. Number of volumes: {num_volumes}")
trimesh.smoothing.filter_laplacian(mesh, iterations=1)
print(f"Smoothing completed")

mesh.apply_scale(1e-6)

#csv export
data_verticies = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
data_verticies.to_csv(f'substrate/{name}/{name}_vertices.csv', index=False)

data_faces = pd.DataFrame(mesh.faces, columns=['v1','v2','v3'])
data_faces.to_csv(f'substrate/{name}/{name}_faces.csv', index=False)




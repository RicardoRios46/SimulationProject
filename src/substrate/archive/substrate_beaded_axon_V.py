import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

length = 60
radius_fiberOG = 0.6
spacing = 5
n = int(length / spacing)

def create_ba(length, radius_fiber, radius_bead, spacing):
    """
    Returns a beaded axon mesh (trimesh)
    """
    shaft = trimesh.creation.cylinder(radius=radius_fiber, height=length)
    
    beads = []
    z_positions = np.arange(-length/2, length/2, spacing)
    
    if radius_bead > 0:
        for z in z_positions:
            num = np.random.randint(0, spacing)
            bead = trimesh.creation.uv_sphere(radius=radius_bead)
            bead.apply_translation([0, 0, z + num])
            if -length/2 < (z + num) < length/2:
                beads.append(bead)

        all_parts = [shaft] + beads
        beaded_mesh = trimesh.boolean.union(all_parts, 'manifold')
    else:
        beaded_mesh = shaft
          
    return beaded_mesh


# Size down values for analytical calculations
lengthS = length * 1e-6
spacingS = spacing * 1e-6

# Find baseline volume (straight cylinder, no beading)
baselineMesh = create_ba(length, radius_fiberOG, 0, spacing)
baselineMesh.apply_scale(1e-6)
baseline = baselineMesh.volume

# Calculate max bead radius allowed under constant volume constraints
max_beading = ((3 * (radius_fiberOG * 1e-6)**2 * lengthS) / (4 * n)) ** (1/3)

print(f"Baseline Volume: {baseline:.3e}")
print(f"Number of beads: {n}")
print(f"Maximum bead radius allowed: {max_beading:.3e}\n")

# Range of bead radii to evaluate (using 0.95 scaling factor to avoid limit limits)
bead_radii_range = np.arange(radius_fiberOG * 1e-6, max_beading * 0.95, 1e-7)

# --- Substrate Generation Loop ---
for idx, radius_bead in enumerate(bead_radii_range):
    
    # Calculate matching fiber radius to preserve uniform volume
    if radius_bead > radius_fiberOG * 1e-6:
        numerator = baseline - (4/3) * n * np.pi * radius_bead**3
        denominator = np.pi * lengthS - np.pi * n * radius_bead
        
        # Guard against minor mesh discretization artifacts yielding negative sqrt numbers
        if numerator / denominator < 0:
            radius_fiber = 0.0
        else:
            radius_fiber = np.sqrt(numerator / denominator)
    else:
        radius_fiber = radius_fiberOG * 1e-6

    print(f"Config {idx+1}/{len(bead_radii_range)} | Fiber Radius: {radius_fiber:.3e} | Bead Radius: {radius_bead:.3e}")

    # Generate the 3x3 Bundled Geometry
    axons = []
    for i in range(3):
        for j in range(3):
            x = j * (2 * radius_fiberOG + 0.5) + (i % 2) * radius_fiberOG 
            y = i * (2 * radius_fiberOG + 0.5)

            # Generate mesh in micron-scale units natively
            axon = create_ba(length, radius_fiber * 1e6, radius_bead * 1e6, spacing)
            axon.apply_translation([x, y, 0])
            axons.append(axon)

    # Combine into a unified substrate mesh object
    axons = trimesh.util.concatenate(axons)

    # Scale to SI meters
    axons.apply_scale(1e-6)

    # --- Directory Export & File Setup ---
    name = f"beaded_bundle_rfiber_{radius_fiber*1e6:.3f}_rbead_{radius_bead*1e6:.3f}_VOL"
    output_dir = f"substrate/{name}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Render and Save Visualizations ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(
        axons.vertices[:, 0], 
        axons.vertices[:, 1], 
        axons.vertices[:, 2], 
        triangles=axons.faces, 
        cmap='viridis',
        edgecolor='k', 
        linewidth=0.2
    )

    ax.axis('equal')
    plt.savefig(f'{output_dir}/{name}_mesh.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Export Data Geometries to CSV ---
    data_vertices = pd.DataFrame(axons.vertices, columns=['x', 'y', 'z'])
    data_vertices.to_csv(f'{output_dir}/{name}_vertices.csv', index=False)

    data_faces = pd.DataFrame(axons.faces, columns=['v1', 'v2', 'v3'])
    data_faces.to_csv(f'{output_dir}/{name}_faces.csv', index=False)

print("\nSuccess.")
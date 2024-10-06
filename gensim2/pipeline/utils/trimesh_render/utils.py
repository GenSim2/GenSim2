from .src import lookAt
import trimesh
import numpy as np


# Get the sphere vertices + center as the potential position for the camera
def getSpherePositions(center, radius, subdivisions=1, num_samples=-1):
    # Create a mesh sphere with the defined radius
    sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    vertices = sphere.vertices
    positions = np.array(vertices + center)

    if num_samples > len(positions):
        raise ValueError("Number of samples is greater than the number of positions")

    if num_samples > 0:
        # Randomly sample some points
        positions = np.random.choice(positions, num_samples, replace=False)

    return positions

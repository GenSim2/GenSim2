import open3d as o3d
import numpy as np

# Step 1: Load the OBJ file
mesh = o3d.io.read_triangle_mesh(
    "assets/articulated_objs/refrigerator/11231/textured_objs/new-4.obj"
)

# Step 2: Define a transformation matrix
# For example, to flip the Z-axis (common when switching coordinate systems)
transformation_matrix = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

# Step 3: Apply the transformation
mesh.transform(transformation_matrix)

# Step 4: Save the transformed mesh
o3d.io.write_triangle_mesh(
    "assets/articulated_objs/refrigerator/11231/textured_objs/new-4.obj", mesh
)

import numpy as np
import open3d as o3d
import copy


# Convert the trimesh scene into open3d geometry
def getOpen3DFromTrimeshScene(
    trimesh_scene, random_color=True, color=np.array([1, 0, 0])
):
    mesh = o3d.geometry.TriangleMesh()
    geo_trans_mapping = {}
    # Get the mapping between the geometry key and its corresponding transformation
    for key in trimesh_scene.graph.nodes_geometry:
        geo_trans_mapping[trimesh_scene.graph[key][1]] = trimesh_scene.graph[key][0]

    for key, geometry in trimesh_scene.geometry.items():
        # Need to deepcopy, or it will influence the original mesh
        geometry = copy.deepcopy(geometry)
        # Take the scene transformation into account
        geometry.apply_transform(
            np.dot(trimesh_scene.graph["world"][0], geo_trans_mapping[key])
        )
        # Convert the mesh into open3d
        temp_mesh = geometry.as_open3d
        if random_color:
            temp_mesh.paint_uniform_color(np.random.rand(3))
        mesh += temp_mesh
    if not random_color:
        mesh.paint_uniform_color(color)
    return mesh


def getConventionTransform(source):
    transformation = np.matrix(np.eye(4))
    if source == "partnetsim" or source == "sapien" or source == "SAPIEN":
        transformation[0, 0] = -1
        transformation[2, 2] = -1
    elif source == "shape2motion":
        transformation[0:3, 0:3] = np.matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    return transformation.I

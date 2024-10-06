import trimesh


# This function doesn't copy the geometry. Modify based on https://github.com/mikedh/trimesh/blob/main/trimesh/scene/scene.py#L1194
# The original function has something wrong with the geometry copy
def scene_transform_copy(mesh):
    """
    Return a deep copy of the current scene
    Returns
    ----------
    copied : trimesh.Scene
        Copy of the current scene
    """
    # Doesn't copy the geometry
    geometry = mesh.geometry

    if not hasattr(mesh, "_camera") or mesh._camera is None:
        # if no camera set don't include it
        camera = None
    else:
        # otherwise get a copy of the camera
        camera = mesh.camera.copy()
    # create a new scene with copied geometry and graph
    copied = trimesh.Scene(
        geometry=geometry,
        graph=mesh.graph.copy(),
        metadata=mesh.metadata.copy(),
        camera=camera,
    )
    return copied


class MeshNode:
    def __init__(self):
        self.mesh = None

    def addMesh(self, mesh):
        if self.mesh == None:
            self.mesh = mesh
        else:
            self.mesh = trimesh.scene.scene.append_scenes([self.mesh, mesh])

    def addMeshFile(self, mesh_file):
        # Read the mesh from obj file
        mesh = trimesh.load(mesh_file)
        if not isinstance(mesh, trimesh.Scene):
            scene = trimesh.Scene()
            scene.add_geometry(mesh)
            mesh = scene
        self.addMesh(mesh)

    def getMesh(self, worldMatrix):
        if self.mesh == None:
            return None
        new_mesh = scene_transform_copy(self.mesh)
        new_mesh.apply_transform(worldMatrix)
        return new_mesh

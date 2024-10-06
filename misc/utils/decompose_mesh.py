import coacd
import trimesh

input_file = "assets/articulated_objs/dishwasher/12085/textured_objs/original-4.obj"

mesh = trimesh.load(input_file, force="mesh")
mesh = coacd.Mesh(mesh.vertices, mesh.faces)
parts = coacd.run_coacd(mesh, threshold=0.2)  # a list of convex hulls.

# save parts into new obj files
for i, part in enumerate(parts):
    # import ipdb; ipdb.set_trace()
    part_mesh = trimesh.Trimesh(part[0], part[1])
    output_file = input_file.replace(".obj", f"-part-{i}.obj")
    part_mesh.export(output_file)

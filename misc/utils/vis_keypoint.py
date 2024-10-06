import os
import glob
import argparse

# from colmap_depth import extract_all_depths
import numpy as np
import cv2

import open3d as o3d
import os
import IPython
import json
import yaml

from gensim2.pipeline.utils.open3d_RGBD import getOpen3DFromTrimeshScene
from gensim2.pipeline.utils.trimesh_render import lookAt
from gensim2.pipeline.utils.trimesh_URDF import getURDF

parser = argparse.ArgumentParser()

parser.add_argument("--mesh_file", "-m", type=str, default="")  # .obj
parser.add_argument("--urdf_file", "-u", type=str, default="")  # .urdf
parser.add_argument("--ply_file", "-p", type=str, default="")  # .ply
parser.add_argument("--json_file", "-j", type=str, default="")  # ga_partnet yaml file

args = parser.parse_args()

data_loaded = False
meshes = []

if len(args.mesh_file) > 0:
    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    pcd = mesh.sample_points_uniformly(number_of_points=20000)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw_geometries([pcd])
    meshes.append(pcd)
    data_loaded = True

if len(args.urdf_file) > 0:
    if data_loaded:
        print("Please only specify one of ply_file and mesh_file")
        exit()
    urdf, controller = getURDF(args.urdf_file)
    trimesh_scene = urdf.getMesh()
    mesh = getOpen3DFromTrimeshScene(trimesh_scene)

    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw_geometries([pcd])
    meshes.append(pcd)
    data_loaded = True

if len(args.ply_file) > 0:
    if data_loaded:
        print("Please only specify one of ply_file and mesh_file")
        exit()
    mesh = o3d.io.read_point_cloud(args.ply_file)
    meshes.append(mesh)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
# o3d.visualization.draw_geometries([mesh])

if len(args.json_file) > 0:  # GA PartNet
    keypoint_dict = yaml.safe_load(open(args.json_file))
    # get the directory of the args.json_file
    pose = os.path.dirname(args.json_file) + "/pose.yaml"
    pose_dict = yaml.safe_load(open(pose))
    tool_keypoints = [
        [
            keypoint_dict["pos"]["x"] * pose_dict["scale"],
            keypoint_dict["pos"]["y"] * pose_dict["scale"],
            keypoint_dict["pos"]["z"] * pose_dict["scale"],
        ]
    ]
    # tool_keypoints = [[(keypoint_dict["pos"]["x"] + pose_dict["pos"][0])*pose_dict["scale"] , (keypoint_dict["pos"]["y"] + pose_dict["pos"][1]) * pose_dict["scale"], (keypoint_dict["pos"]["z"] + pose_dict["pos"][2]) * pose_dict["scale"]]]
    # tool_keypoints = [[keypoint_dict["pos"]["x"], keypoint_dict["pos"]["y"], keypoint_dict["pos"]["z"]]]
    print(tool_keypoints)
    # tool_keypoints = tool_keypoints[0] * pose_dict["scale"]
    radius = 0.05
    # radius = 0.1
else:
    if len(args.urdf_file) > 0:
        keypoint_info = json.load(
            open(os.path.dirname(os.path.dirname(args.urdf_file)) + "/info.json")
        )
    elif len(args.mesh_file) > 0:
        keypoint_info = json.load(open(os.path.dirname(args.mesh_file) + "/info.json"))
        # keypoint_info = json.load(open(os.path.dirname(os.path.dirname(args.mesh_file)) + "/info.json"))
    elif len(args.ply_file) > 0:
        keypoint_info = json.load(
            open(os.path.dirname(os.path.dirname(args.ply_file)) + "/info.json")
        )

    tool_keypoints = keypoint_info["keypoints"]
    tool_keypoints = list(tool_keypoints.values())
    print(tool_keypoints)
    radius = 0.01

colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
]  # red green blue

for idx in range(len(tool_keypoints)):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(tool_keypoints[idx])
    mesh_sphere.paint_uniform_color(colors[idx])
    meshes.append(mesh_sphere)

# visualizer with editing

# viewer = o3d.visualization.VisualizerWithEditing()
viewer = o3d.visualization.Visualizer()
viewer.create_window()
print("meshes:", len(meshes))

for m in meshes:
    viewer.add_geometry(m)

# viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()

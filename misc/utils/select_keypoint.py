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

from gensim2.pipeline.utils.trimesh_URDF import getURDF
from gensim2.pipeline.utils.open3d_RGBD import getOpen3DFromTrimeshScene


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--urdf_dir", type=str)

# parser.add_argument('--write', required=True)
args = parser.parse_args()
datapath = args.dir

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

urdf, controller = getURDF(os.path.join(args.urdf_dir, "mobility.urdf"))
trimesh_scene = urdf.getMesh()
# mesh = o3d.io.read_triangle_mesh(args.mesh_file)
mesh = getOpen3DFromTrimeshScene(trimesh_scene)
pcd = mesh.sample_points_uniformly(number_of_points=500000)
# o3d.visualization.draw_geometries([pcd])

viewer = o3d.visualization.VisualizerWithEditing()
viewer.create_window()
viewer.add_geometry(pcd)

opt = viewer.get_render_option()
opt.show_coordinate_frame = True


# opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()

print("saving picked points")
picked_points = viewer.get_picked_points()

if len(picked_points) == 0:
    print("No points were picked")
    exit()

xyz = np.asarray(pcd.points)
print(picked_points)
picked_points = xyz[picked_points]
print(picked_points)
color_lists = ["red", "yellow", "blue", "green", "magenta", "purple", "orange"]

dir_name = args.urdf_dir
keypoint_description_file = os.path.join(dir_name, "keypoints.json")

keypoint_info = {
    "keypoints": {c: p.tolist() for c, p in zip(color_lists, picked_points)},
}

with open(keypoint_description_file, "w") as f:
    json.dump(keypoint_info, f, indent=4, sort_keys=True)

print("keypoint_info saved to", keypoint_description_file)
# visualize and also generate masks
# o3d.visualization.draw_geometries_with_editing([pcd])

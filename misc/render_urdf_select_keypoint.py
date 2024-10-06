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
from gensim2.pipeline.utils import render_urdf_sapien


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--mesh_file", type=str)

# parser.add_argument('--write', required=True)
args = parser.parse_args()
datapath = args.dir
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)


# instead of reading triangle mesh. render depth images in sapien for multiple views
# fuse them together to form a open3d point cloud
# this is not done
IPython.embed()
depth_img = render_urdf_sapien(args.mesh_file, args.mesh_file, return_depth=True)
pcd = o3d.geometry.PointCloud()
xyz = depth_img[..., :3].reshape(-1, 3)
xyz = xyz[xyz.sum(-1) != 0]  # nonzero mask
pcd.points = o3d.utility.Vector3dVector(xyz)
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
color_lists = ["red", "yellow", "blue", "green"]

keypoint_description_file = args.mesh_file.replace(".obj", "_all_keypoints.json")
keypoint_info = {
    "keypoints": {c: p.tolist() for c, p in zip(color_lists, picked_points)},
}

with open(keypoint_description_file, "w") as f:
    json.dump(keypoint_info, f, indent=4, sort_keys=True)

print("keypoint_info saved to", keypoint_description_file)

# visualize and also generate masks
# o3d.visualization.draw_geometries_with_editing([pcd])

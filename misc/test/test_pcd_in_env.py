import icecream
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim
from gensim2.agent.utils.pcd_utils import *
from gensim2.env.utils.pcd_utils import *

import numpy as np
import sapien.core as sapien

from common_parser import parser
from PIL import Image
import ipdb
import open3d as o3d
import matplotlib.pyplot as plt

from datetime import datetime

if __name__ == "__main__":
    args = parser.parse_args()

    env = create_gensim(
        task_name=args.env,
        asset_id=args.asset_id,
        use_gui=False,
        use_ray_tracing=args.rt,
        eval=False,
        obs_mode="pointcloud",
        headless=False,
        cam=args.cam,
    )

    icecream.ic(env.horizon)
    builder = env.scene.create_actor_builder()

    obs = env.reset(args.random)
    action = np.zeros(env.action_space.shape)
    #     pos=array([[0.43722507],
    #        [0.03720725],
    #        [0.33280694]], dtype=float32)
    # orientation=array([[-0.01035694],
    #        [ 0.98963939],
    #        [ 0.09988713],
    #        [ 0.10261173]])
    print(action.shape)
    action[:3] = np.array([0.43722507, 0.03720725, 0.33280694])
    action[3:-1] = np.array([-0.01035694, 0.98963939, 0.09988713, 0.10261173])
    obs, reward, done, info = env.step(action=action)

    joint_pose = [
        0.00000000e00,
        -3.19999993e-01,
        0.00000000e00,
        -2.61799383e00,
        0.00000000e00,
        2.23000002e00,
        7.85398185e-01,
    ]

    # env.task.env.agent.set_qpos(joint_pose)

    # print(env.task.env.agent.get_qpos())
    pcds = obs["pointcloud"]

    pcd_o3d = o3d.geometry.PointCloud()

    pcd = pcds["pos"]

    print(pcd)

    # pcd = pcd[uniform_sampling(pcd, npoints=8192)]
    # pcd = pcd[fps_sampling(pcd, npoints=4096)]

    # save pcd
    # o3d.io.write_point_cloud(f"pcd_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pcd", pcd_o3d)

    # print(pcd.shape)
    # voxel, voxel_center = voxelize_point_cloud(pcd, voxel_size=0.05)
    # import ipdb

    # ipdb.set_trace()
    # o3d.visualization.draw_geometries([voxel])
    # pcds["pos"] = open3d_pcd_outlier_removal(pcd)  # should use in real
    # pcds["pos"] = add_crop_noise_to_points_data(pcd)
    # pcds["pos"] = add_pointoutlier_aug(pcd)
    pcds["pos"] = add_gaussian_noise(pcd, np.random)  # Should use in sim
    pcds["pos"] = randomly_drop_point(pcd)  # Should use in sim
    # pcds["pos"] = cutplane_pointcloud_aug(pcd)

    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pcd_o3d])

    # ================ render pcds
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcds["pos"])
    # # pcd_o3d.colors = o3d.utility.Vector3dVector(pcds["colors"]/255)

    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # # Initialize the visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # # Add the point cloud to the visualizer
    # vis.add_geometry(pcd_o3d)
    # vis.add_geometry(coordinate_frame)

    # # Get the render options and set the point size
    # render_option = vis.get_render_option()
    # render_option.point_size = 5.0  # Set the point size to your desired value

    # # Run the visualizer
    # vis.run()
    # vis.destroy_window()

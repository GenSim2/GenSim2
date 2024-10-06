import open3d as o3d


def vis_pcd(pcds, is_dict=False):
    geom_list = []

    front_cam_pcd = o3d.geometry.PointCloud()

    if is_dict:
        front_cam_pcd.points = o3d.utility.Vector3dVector(pcds["pos"])
        front_cam_pcd.colors = o3d.utility.Vector3dVector(pcds["colors"])
    else:
        front_cam_pcd.points = o3d.utility.Vector3dVector(pcds[..., :3])
        front_cam_pcd.colors = o3d.utility.Vector3dVector(pcds[..., 3:])
    # front_cam_pcd.points = o3d.utility.Vector3dVector(points)
    # front_cam_pcd.colors = o3d.utility.Vector3dVector(colors)

    # save the point cloud
    # o3d.io.write_point_cloud("combined_pcd.ply", front_cam_pcd)
    geom_list.append(front_cam_pcd)
    geom_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    o3d.visualization.draw_geometries(geometry_list=geom_list)

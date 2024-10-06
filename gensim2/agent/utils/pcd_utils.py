import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import os


def rand_dist(size, min=-1.0, max=1.0):
    return (max - min) * torch.rand(size) + min


def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max + 1, size)


def voxelize_point_cloud(points, voxel_size=0.1, colors=None):
    """
    Voxelize a single point cloud.

    Parameters:
    points (np.ndarray): Input point cloud as a (N, 3) array.
    colors (np.ndarray): Input point cloud color as a (N, 3) array.
    voxel_size (float): Size of the voxel grid.

    Returns:
    np.ndarray: Voxelized point cloud.
    """
    # Convert numpy array to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Voxelize the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )

    # Extract voxel centers
    voxels = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    voxel_centers = voxels * voxel_size + voxel_grid.origin

    return voxel_grid, voxel_centers


def visualize_point_cloud(points):
    # Convert the points to Open3D's PointCloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the PointCloud
    o3d.visualization.draw_geometries([pcd])


def update_openpoint_cfgs(cfg):
    pass


def vis_attention(pc1, pc2, attention_score):
    """
    src_emb: B * 3 * N_src
    score: B * N_src * N_dst
    """
    # TODO(lirui): build interactive feature visualization of the attention
    # http://www.open3d.org/docs/0.9.0/tutorial/Advanced/interactive_visualization.html

    for idx, (tool_point, obj_point) in enumerate(zip(pc1, pc2)):
        tool_pcd = o3d.geometry.PointCloud()
        tool_pcd.points = o3d.utility.Vector3dVector(tool_point.detach().cpu().numpy())
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_point.detach().cpu().numpy())
        tool_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        obj_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(tool_pcd)
        vis.run()  # user picks points

        attn = attention_score[idx].detach().cpu().numpy()
        picked_points = vis.get_picked_points()
        print("picked_points:", picked_points)

        if len(picked_points) > 0:
            corrsponding_score = attn[picked_points[0]]
            colors = plt.cm.jet(corrsponding_score)[:, :3]
            obj_pcd.colors = o3d.utility.Vector3dVector(colors)

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(obj_pcd)
            vis.run()  # user picks points
            vis.destroy_window()


def dbscan_outlier_removal(pcd):  # (N, 3)
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(pcd)
    labels = clustering.labels_
    print("Number of clusters: ", len(set(labels)))
    # max_label = max(set(labels), key=labels) # only keep the cluster with the most points

    return np.array(pcd)[labels != -1]


def dbscan_outlier_removal_idx(pcd, eps=0.1, min_samples=300):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pcd)
    labels = clustering.labels_
    # print("Number of clusters: ", len(set(labels)))
    non_outlier_indices = np.where(labels != -1)[0]

    return non_outlier_indices


def dbscan_outlier_removal_idx(pcd, eps=0.1, min_samples=300):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pcd)
    labels = clustering.labels_
    # print("Number of clusters: ", len(set(labels)))
    non_outlier_indices = np.where(labels != -1)[0]

    return non_outlier_indices


def open3d_pcd_outlier_removal(
    pointcloud, radius_nb_num=300, radius=0.08, std_nb_num=300, vis=False
):
    """N x 3 or N x 6"""
    print("running outlier removal")
    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(pointcloud[..., :3])
    model_pcd.colors = o3d.utility.Vector3dVector(pointcloud[..., 3:])
    # prior: it's a single rigid object
    model_pcd.remove_duplicated_points()
    model_pcd.remove_non_finite_points()
    print("finished removing duplicated and non-finite points")

    cl, ind = model_pcd.remove_radius_outlier(
        nb_points=int(radius_nb_num), radius=radius
    )
    model_pcd.points = o3d.utility.Vector3dVector(np.array(model_pcd.points)[ind, :3])
    model_pcd.colors = o3d.utility.Vector3dVector(np.array(model_pcd.colors)[ind, :3])
    print("finished removing radius outliers")

    cl, ind = model_pcd.remove_statistical_outlier(
        nb_neighbors=std_nb_num, std_ratio=2.0
    )
    print("finished removing statistical outliers")
    if vis:
        display_inlier_outlier(model_pcd, ind)
    # return pointcloud[ind] # No remove, not sure why
    return np.array(model_pcd.select_by_index(ind).points), np.array(
        model_pcd.select_by_index(ind).colors
    )


def display_inlier_outlier(pcd, ind):
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    # Display inlier and outlier point clouds
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def add_crop_noise_to_points_data(point_cloud, radius_min=0.015, radius_max=0.04):
    """
    select an anchor point, and remove all points inside radius of size 0.02
    """
    # should be independent of tool or object
    CROP_REMAINING = np.random.uniform() < 0.5
    radius = radius_max
    if not CROP_REMAINING:
        radius = radius_min
        radius = np.random.uniform(0.005, radius)  # further reduce

    CROP_TOOL = True
    point_num = point_cloud.shape[0]
    point_cloud = (
        point_cloud[: point_num // 2] if CROP_TOOL else point_cloud[point_num // 2 :]
    )

    select_anchor_index = np.random.randint(point_num // 2)
    point_center = point_cloud[select_anchor_index]
    close_dist = np.linalg.norm(point_cloud[:, :3] - point_center[None, :3], axis=1)
    close_mask = close_dist < radius if CROP_REMAINING else close_dist >= radius
    if close_mask.sum() != 0 and close_mask.sum() != point_num // 2:
        # in case it crops out entire object or tool
        masked_point = point_cloud[close_mask]
        random_index = np.random.choice(masked_point.shape[0], (~close_mask).sum())
        masked_point = np.concatenate(
            (masked_point, masked_point[random_index]), axis=0
        )
        point_cloud[:] = masked_point  # modify in place

    return point_cloud.astype(np.float32)


def randomly_drop_point(point_clouds, drop_point_min=0.1, drop_point_max=0.5):
    """
    randomly drop points such that the pointcloud can work in the real world
    """
    orig_shape = point_clouds.shape
    if len(orig_shape) == 2:  # (N, 3)
        point_clouds = point_clouds[None, :, :]  # B, N, 3

    remained_point_ratio = np.random.uniform(drop_point_min, drop_point_max)
    remained_point_num = int(point_clouds.shape[1] * remained_point_ratio)

    for idx, point_cloud in enumerate(point_clouds):
        try:
            random_index = np.random.choice(point_cloud.shape[0], remained_point_num)
            resampled_index = np.random.choice(random_index, point_cloud.shape[0])
            point_cloud = point_cloud[resampled_index]
        except:
            pass
        point_clouds[idx] = point_cloud  # in-place

    if len(orig_shape) == 2:
        point_clouds = point_clouds.squeeze(0)

    return point_clouds.astype(np.float32)


def add_gaussian_noise(clouds: np.ndarray, noise_level=1):
    # cloud should be (B, n, 3)
    orig_shape = clouds.shape
    if len(orig_shape) == 3:
        B, N, _ = clouds.shape
        clouds = clouds.reshape(-1, 3)
    num_points = clouds.shape[0]
    multiplicative_noise = (
        1 + np.random.randn(num_points)[:, None] * 0.01 * noise_level
    )  # (n, 1)
    clouds = clouds * multiplicative_noise
    if len(orig_shape) == 3:
        clouds = clouds.reshape(B, N, 3).astype(np.float32)
    return clouds


def add_pointoutlier_aug(point_cloud, outlier_point_num=20):
    """
    add outlier points to the pointcloud
    """

    # completely random points to increate robustness
    outlier_points = np.random.uniform(-1, 1, size=(outlier_point_num, 3))
    random_index = np.random.choice(point_cloud.shape[0], outlier_point_num)
    point_cloud[random_index] = outlier_points

    # point_clouds[:,:3] = point_cloud.T
    return point_cloud.astype(np.float32)


def cutplane_pointcloud_aug(point_cloud, action):
    # print("in cutplane")
    N = len(point_cloud)

    for b in range(N):

        if np.random.uniform() < 0.5:
            # tool
            cut_tool = True
            pcd = point_cloud[b].T[:512, :3]
        else:
            # object
            cut_tool = False
            pcd = point_cloud[b].T[512:, :3]

        bounding_box = trimesh.PointCloud(pcd).bounding_box
        # bounding_box = bounding_box.to_mesh()
        random_pts = bounding_box.sample_volume(2)

        # first point will be the point, second point will be used for the normal
        pt = random_pts[0]
        vec = random_pts[1] - pt
        normal = vec / np.linalg.norm(vec)
        normal = normal.reshape(3, 1)

        # get points that are on one side of this plane
        shifted_pcd = pcd - pt
        dots = np.matmul(shifted_pcd, normal)
        pos_inds = np.where(dots > 0)[0]
        neg_inds = np.where(dots < 0)[0]
        # pick the majority and then oversample it to match the pointcloud size

        # print(f"pos: {len(pos_inds)} neg: {len(neg_inds)}")
        if len(pos_inds) > len(neg_inds):
            keep_pts = pcd[pos_inds]
        else:
            keep_pts = pcd[neg_inds]

        if pcd.shape[0] > keep_pts.shape[0]:
            random_index = np.random.choice(
                keep_pts.shape[0], pcd.shape[0] - keep_pts.shape[0], replace=True
            )
        keep_pts = np.concatenate((keep_pts, keep_pts[random_index]), axis=0)
        if cut_tool:
            point_cloud[b].T[:512, :3] = keep_pts  # in-place should be
        else:
            point_cloud[b].T[512:, :3] = keep_pts  # in-place should be

    return point_cloud.astype(np.float32), action.astype(np.float32)


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    return Rz, Ry


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1, color=(1.0, 0.0, 0.0)):
    assert not np.all(end == origin)
    import open3d as o3d

    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size / 17.5 * scale,
        cone_height=size * 0.2 * scale,
        cylinder_radius=size / 30 * scale,
        cylinder_height=size * (1 - 0.2 * scale),
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return mesh


def rhlb(bounds):
    val = np.random.random() * (max(bounds) - min(bounds)) + min(bounds)
    return val


def transform_pcd(pcd, transform):
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


def vis_pcd_html(pcds, rgbs, name, gt_traj=None):
    rgb_strings = [f"rgb{rgbs[i][0],rgbs[i][1],rgbs[i][2]}" for i in range(len(rgbs))]

    if gt_traj is not None:
        gx, gy, gz = gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2]

    pcd_plots = [
        go.Scatter3d(
            x=pcds[:, 0],
            y=pcds[:, 1],
            z=pcds[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=rgb_strings,
            ),
        )
    ]

    if gt_traj is not None:
        gt_plot = [
            go.Scatter3d(
                x=gx,
                y=gy,
                z=gz,
                mode="markers",
                marker=dict(size=10, color="red"),
            )
        ]
        pcd_plots += gt_plot

    fig = go.Figure(pcd_plots)
    path = f"./plots"
    os.makedirs(path, exist_ok=True)
    fig.write_html(os.path.join(path, f"vis_{name}.html"))


def simulate_deform_contact_point(point_cloud, uniform=False):
    # pcd: N x 4 x 1024
    # high, low = 1.5, 0.5
    from scipy.spatial.transform import Rotation as R

    high, low = 2.0, 0.4
    for idx, pcd_ in enumerate(point_cloud):
        pcd = pcd_.T[:, :3]
        deform_about_point = np.random.randint(len(pcd))
        deform_about_point = pcd[deform_about_point]

        # scale up the points about this specific location
        # scale_x, scale_y, scale_z = rhlb((1.5, 0.5)), rhlb((1.5, 0.5)), rhlb((1.5, 0.5))
        scale_x, scale_y, scale_z = (
            rhlb((high, low)),
            rhlb((high, low)),
            rhlb((high, low)),
        )

        # apply the scaling to the place pcd
        pcd_contact_cent = pcd - deform_about_point
        if uniform:
            pcd_contact_cent = pcd_contact_cent * scale_x
            pcd_aug = pcd_contact_cent + deform_about_point
        else:
            # apply a random rotation, scale, and then unrotate
            # rot_grid = R.random().as_matrix()
            # rot_idx = np.random.randint(rot_grid.shape[0], size=1)
            rnd_rot = R.random().as_matrix()  # rot_grid[rot_idx]
            rnd_rot_T = np.eye(4)
            rnd_rot_T[:-1, :-1] = rnd_rot

            pcd_contact_cent = transform_pcd(pcd_contact_cent, rnd_rot_T)
            pcd_contact_cent[:, 0] *= scale_x
            pcd_contact_cent[:, 1] *= scale_y
            pcd_contact_cent[:, 2] *= scale_z

            pcd_contact_cent = transform_pcd(pcd_contact_cent, np.linalg.inv(rnd_rot_T))

            pcd_aug = pcd_contact_cent + deform_about_point
        pcd_[:3, :] = pcd_aug.T
    return point_cloud


def se3_augmentation(gripper_poses_euler, point_clouds, bounds, rgbs=None):
    """
    Apply SE(3) augmentation to batches of gripper poses (represented in Euler angles) and point clouds,
    ensuring each gripper pose remains within specified bounds.

    Parameters:
    gripper_poses_euler (np.ndarray): Batch of gripper poses as a Bx6 matrix [x, y, z, roll, pitch, yaw].
    point_clouds (np.ndarray): Batch of point clouds as a BxNx3 matrix.
    bounds (list or np.ndarray): The scene bounds [x0, x1, y0, y1, z0, z1].

    Returns:
    np.ndarray: Batch of augmented gripper poses in Euler angles.
    np.ndarray: Batch of augmented point clouds.
    """
    # Generate random rotation
    # random_rotation = R.random().as_matrix()  # Random rotation matrix

    ## We limit the rotation to yaw and +-15 degrees
    # Generate a random angle between -15 and 5 degrees
    angle_rad = np.radians(np.random.uniform(-15, 15))

    # Create the yaw rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    random_rotation = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )

    # Generate random translation within bounds
    translation = np.zeros(3)
    perturb_attempts = 0
    while True:
        perturb_attempts += 1
        if perturb_attempts > 100:
            print(
                "Failing to perturb action and keep it within bounds. Use the original one."
            )
            return gripper_poses_euler, point_clouds

        trans_range = (0.125, 0.125, 0.125)  # Adjust range as needed
        translation = trans_range * np.random.uniform(-1, 1, size=3)
        new_positions = gripper_poses_euler[:, :3] + translation

        # Check if all new positions are within bounds
        if np.all(
            (bounds[0] <= new_positions[:, 0])
            & (new_positions[:, 0] <= bounds[1])
            & (bounds[2] <= new_positions[:, 1])
            & (new_positions[:, 1] <= bounds[3])
            & (bounds[4] <= new_positions[:, 2])
            & (new_positions[:, 2] <= bounds[5])
        ):
            break

    new_euler_angles = gripper_poses_euler[:, 3:]
    # # Convert Euler angles to rotation matrices
    # rotation_matrices = R.from_euler("xyz", gripper_poses_euler[:, 3:]).as_matrix()

    # # Calculate new rotation matrices
    # new_rotation_matrices = np.einsum("ij,bjk->bik", random_rotation, rotation_matrices)

    # # Convert new rotation matrices back to Euler angles
    # new_euler_angles = R.from_matrix(new_rotation_matrices).as_euler("xyz")

    # Apply the translation to the gripper pose positions
    augmented_positions = gripper_poses_euler[:, :3] + translation

    # Create the augmented gripper poses
    augmented_gripper_poses_euler = np.hstack((augmented_positions, new_euler_angles))

    # shift points to have action_gripper pose as the origin
    gripper_pos = np.tile(
        gripper_poses_euler[:, :3], (point_clouds.shape[1], 1, 1)
    ).transpose(1, 0, 2)
    augmented_point_clouds = point_clouds - gripper_pos
    # Apply the SE(3) transformation to each point cloud in the batch
    # augmented_point_clouds = np.einsum("ij,bnj->bni", random_rotation, augmented_point_clouds)
    # Shift the point clouds back to the original position
    augmented_point_clouds = augmented_point_clouds + gripper_pos + translation

    # vis_pcd_html(augmented_point_clouds[0], rgbs[0], "augmented", augmented_gripper_poses_euler)
    # vis_pcd_html(point_clouds[0], rgbs[0], "origin", gripper_poses_euler)

    return augmented_gripper_poses_euler, augmented_point_clouds

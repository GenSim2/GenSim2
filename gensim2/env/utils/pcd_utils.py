import torch
import noise

try:
    import pytorch3d.ops as torch3d_ops
except:
    print("pytorch3d not installed")

try:
    from openpoints.models.layers import furthest_point_sample
except:
    print("openpoints not installed")
import numpy as np

# DESK2ROBOT_Z_AXIS = 0.02 # open laptop
DESK2ROBOT_Z_AXIS = 0.0  # close laptop
# DESK2ROBOT_Z_AXIS = -0.04  # open / close safe
# DESK2ROBOT_Z_AXIS = 0.07
# DESK2ROBOT_Z_AXIS = 0.035
# DESK2ROBOT_Z_AXIS = 0.040
# DESK2ROBOT_Z_AXIS = 0.01
# DESK2ROBOT_Z_AXIS = -0.03

BOUND = [0.15, 0.8, -0.6, 0.6, DESK2ROBOT_Z_AXIS + 0.005, 0.8]


def vis_pcd(pcd):
    import open3d as o3d

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd["pos"])
    if "colors" in pcd:
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd["colors"] / 255)
    o3d.visualization.draw_geometries([pcd_o3d])


def select_mask(obs, key, mask):
    if key in obs:
        obs[key] = obs[key][mask]


def pcd_filter_bound(cloud, eps=1e-3, max_dis=1.5, bound=BOUND):
    # return (
    #     (pcd["pos"][..., 2] > eps)
    #     & (pcd["pos"][..., 1] < max_dis)
    #     & (pcd["pos"][..., 0] < max_dis)
    #     & (pcd["pos"][..., 2] < max_dis)
    # )
    if isinstance(cloud, dict):
        pc = cloud["pos"]  # (n, 3)
    else:
        assert isinstance(cloud, np.ndarray), f"{type(cloud)}"
        assert cloud.shape[1] == 3, f"{cloud.shape}"
        pc = cloud

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(
        np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z))
    )[0]

    return within_bound


def pcd_filter_with_mask(obs, mask, env=None):
    assert isinstance(obs, dict), f"{type(obs)}"
    for key in ["pos", "colors", "seg", "visual_seg", "robot_seg"]:
        select_mask(obs, key, mask)


def pcd_downsample(
    obs,
    env=None,
    bound_clip=False,
    ground_eps=-1e-3,
    max_dis=15,
    num=1200,
    method="fps",
    bound=BOUND,
):
    assert method in [
        "fps",
        "uniform",
    ], "expected method to be 'fps' or 'uniform', got {method}"

    sample_mehod = uniform_sampling if method == "uniform" else fps_sampling
    # import ipdb; ipdb.set_trace()
    if bound_clip:
        pcd_filter_with_mask(
            obs,
            pcd_filter_bound(obs, eps=ground_eps, max_dis=max_dis, bound=bound),
            env,
        )
    pcd_filter_with_mask(obs, sample_mehod(obs["pos"], num), env)
    return obs


def fps_sampling(points, npoints=1200):
    num_curr_pts = points.shape[0]
    if num_curr_pts < npoints:
        return np.random.choice(num_curr_pts, npoints, replace=True)
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    try:
        fps_idx = furthest_point_sample(points[..., :3], npoints)
    except:
        npoints = torch.tensor([npoints]).cuda()
        _, fps_idx = torch3d_ops.sample_farthest_points(points[..., :3], K=npoints)

    return fps_idx.squeeze(0).cpu().numpy()


def uniform_sampling(points, npoints=1200):
    n = points.shape[0]
    index = np.arange(n)
    if n == 0:
        return np.zeros(npoints, dtype=np.int64)
    if index.shape[0] > npoints:
        np.random.shuffle(index)
        index = index[:npoints]
    elif index.shape[0] < npoints:
        num_repeat = npoints // index.shape[0]
        index = np.concatenate([index for i in range(num_repeat)])
        index = np.concatenate([index, index[: npoints - index.shape[0]]])
    return index


def add_gaussian_noise(
    cloud: np.ndarray, np_random: np.random.RandomState, noise_level=1
):
    # cloud is (n, 3)
    num_points = cloud.shape[0]
    multiplicative_noise = (
        1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level
    )  # (n, 1)
    return cloud * multiplicative_noise


def add_perlin_noise(
    points, scale=0.1, octaves=1, persistence=0.5, lacunarity=2.0, amplitude=1.0
):
    """
    Adds Perlin noise to a point cloud.

    :param points: A numpy array of shape (n, 3) representing the point cloud.
    :param scale: Scale of the Perlin noise.
    :param octaves: Number of octaves for the Perlin noise.
    :param persistence: Persistence of the Perlin noise.
    :param lacunarity: Lacunarity of the Perlin noise.
    :param amplitude: Amplitude of the noise to make the effect more noticeable.
    :return: A numpy array of the same shape as points with added Perlin noise.
    """
    noisy_points = np.zeros_like(points)

    for i, point in enumerate(points):
        x, y, z = point
        noise_x = (
            noise.pnoise3(
                x * scale,
                y * scale,
                z * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noise_y = (
            noise.pnoise3(
                y * scale,
                z * scale,
                x * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noise_z = (
            noise.pnoise3(
                z * scale,
                x * scale,
                y * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noisy_points[i] = point + np.array([noise_x, noise_y, noise_z])

    return noisy_points

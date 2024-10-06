import numpy as np
import open3d as o3d


# Get the point cloud from rgb and depth numpy array
def getPcdFromRgbd(
    rgb,
    depth,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    intrinsic=None,
    depth_scale=1,
    alpha_filter=False,
):
    # Make the rgb go to 0-1
    if rgb.max() > 1:
        rgb /= 255.0
    # Convert the unit to meter
    depth /= depth_scale

    height, width = np.shape(depth)
    points = []
    colors = []

    for y in range(height):
        for x in range(width):
            if alpha_filter:
                # Filter the background based on the alpha channel
                if rgb[y][x][3] != 1:
                    continue
            colors.append(rgb[y][x][:3])
            if fx != None:
                points.append(
                    [
                        (x - cx) * (depth[y][x] / fx),
                        -(y - cy) * (depth[y][x] / fy),
                        -depth[y][x],
                    ]
                )
            else:
                depth[y][x] *= -1
                old_point = np.array(
                    [(width - x) * depth[y][x], y * depth[y][x], depth[y][x], 1]
                )
                point = np.dot(np.linalg.inv(intrinsic), old_point)
                points.append(point[:3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)[:, 0:3])

    return pcd

from functools import partial
import os
import sys
import time

from IPython.display import display, HTML, Javascript
import numpy as np

from pydrake.geometry import Cylinder, Rgba, Sphere
from pydrake.perception import PointCloud, Fields, BaseField
from pydrake.all import BoundingBoxConstraint

# imports for the pose sliders
from collections import namedtuple
from pydrake.all import *

# Some GUI code that will be moved into Drake.
import IPython


def AddMeshcatVector(
    meshcat, path, color, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()
):
    meshcat.SetTransform(path, X_PT)
    meshcat.SetObject(path, Cylinder(radius, length), color)


def AddMeshcatTriad(
    meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()
):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(
        path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(
        path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(
        path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )


def draw_point_cloud(
    meshcat, path, pts, colors, normals_scale=0.0, point_size=0.001, sphere_vis=True
):

    if sphere_vis:
        for idx, pt in enumerate(pts):
            meshcat_geom = Sphere(point_size)
            color = colors[idx]
            point_path = path + f"_point_{idx}"
            meshcat.SetObject(
                point_path, Sphere(point_size), Rgba(color[0], color[1], color[2], 1.0)
            )
            transform = np.eye(4)
            transform[:3, 3] = pt
            meshcat.SetTransform(point_path, RigidTransform(pt))

    else:
        cloud = PointCloud(pts.shape[0], Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_xyzs()[:] = pts.T
        cloud.mutable_rgbs()[:] = 255 * colors.T
        meshcat.SetObject(path, cloud, point_size=point_size)


def plot_surface(
    meshcat,
    path,
    X,
    Y,
    Z,
    rgba=Rgba(0.87, 0.6, 0.6, 1.0),
    wireframe=False,
    wireframe_line_width=1.0,
):
    (rows, cols) = Z.shape
    assert np.array_equal(X.shape, Y.shape)
    assert np.array_equal(X.shape, Z.shape)

    vertices = np.empty((rows * cols, 3), dtype=np.float32)
    vertices[:, 0] = X.reshape((-1))
    vertices[:, 1] = Y.reshape((-1))
    vertices[:, 2] = Z.reshape((-1))

    # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy  # noqa
    faces = np.empty((rows - 1, cols - 1, 2, 3), dtype=np.uint32)
    r = np.arange(rows * cols).reshape(rows, cols)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces.shape = (-1, 3)

    # TODO(Russ): support per vertex / Colormap colors.
    meshcat.SetTriangleMesh(
        path, vertices.T, faces.T, rgba, wireframe, wireframe_line_width
    )


def plot_mathematical_program(meshcat, path, prog, X, Y, result=None, point_size=0.05):
    assert prog.num_vars() == 2
    assert X.size == Y.size

    N = X.size
    values = np.vstack((X.reshape(-1), Y.reshape(-1)))
    costs = prog.GetAllCosts()

    # Vectorized multiply for the quadratic form.
    # Z = (D*np.matmul(Q,D)).sum(0).reshape(nx, ny)

    if costs:
        Z = prog.EvalBindingVectorized(costs[0], values)
        for b in costs[1:]:
            Z = Z + prog.EvalBindingVectorized(b, values)

    cv = f"{path}/constraints"
    for binding in prog.GetAllConstraints():
        if isinstance(binding.evaluator(), BoundingBoxConstraint):
            c = binding.evaluator()
            var_indices = [
                int(prog.decision_variable_index()[v.get_id()])
                for v in binding.variables()
            ]
            satisfied = np.array(
                c.CheckSatisfiedVectorized(values[var_indices, :], 0.001)
            ).reshape(1, -1)
            if costs:
                Z[~satisfied] = np.nan

            v = f"{cv}/{type(c).__name__}"
            Zc = np.zeros(Z.shape)
            Zc[satisfied] = np.nan
            plot_surface(
                meshcat,
                v,
                X,
                Y,
                Zc.reshape(X.shape),
                rgba=Rgba(1.0, 0.2, 0.2, 1.0),
                wireframe=True,
            )
        else:
            Zc = prog.EvalBindingVectorized(binding, values)
            evaluator = binding.evaluator()
            low = evaluator.lower_bound()
            up = evaluator.upper_bound()
            cvb = f"{cv}/{type(evaluator).__name__}"
            for index in range(Zc.shape[0]):
                # TODO(russt): Plot infeasible points in a different color.
                infeasible = np.logical_or(
                    Zc[index, :] < low[index], Zc[index, :] > up[index]
                )
                plot_surface(
                    meshcat,
                    f"{cvb}/{index}",
                    X,
                    Y,
                    Zc[index, :].reshape(X.shape),
                    rgba=Rgba(1.0, 0.3, 1.0, 1.0),
                    wireframe=True,
                )

    if costs:
        plot_surface(
            meshcat,
            f"{path}/objective",
            X,
            Y,
            Z.reshape(X.shape),
            rgba=Rgba(0.3, 1.0, 0.3, 1.0),
            wireframe=True,
        )

    if result:
        v = f"{path}/solution"
        meshcat.SetObject(v, Sphere(point_size), Rgba(0.3, 1.0, 0.3, 1.0))
        x_solution = result.get_x_val()
        meshcat.SetTransform(
            v, RigidTransform([x_solution[0], x_solution[1], result.get_optimal_cost()])
        )

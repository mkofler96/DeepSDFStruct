import matplotlib.pyplot as plt
import numpy as np
from typing import TypedDict
import torch


class custom_zoom(TypedDict):
    x: list[float]
    y: list[float]


def scatter_contour_at_z_level(
    fun,
    z_level=0,
    res=100,
    custom_axis=None,
    eval_area=(-1, 1),
    scale=(1, 1),
    custom_zoom: custom_zoom = {"x": [0.25, 0.75], "y": [-0.25, -0.75]},
    clim=None,
    flip_axes=False,
):
    if custom_axis:
        ax = [custom_axis]
        plt_show = False
    elif custom_zoom is not None:
        _, ax = plt.subplots(1, 2)
    else:
        _, ax = plt.subplots(1, 1)
        ax = [ax]

    x = np.linspace(eval_area[0], eval_area[1], num=res)
    y = np.linspace(eval_area[0], eval_area[1], num=res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) + z_level
    sdf = fun(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T).reshape(X.shape)
    if flip_axes:
        tmp = Y
        Y = X
        X = tmp

    # cbar = ax[0].scatter(X, Y, c=sdf, cmap="seismic")c
    cbar = ax[0].contourf(X * scale[0], Y * scale[1], sdf, cmap="seismic")
    ax[0].contour(
        X * scale[0], Y * scale[1], sdf, levels=[0], color="black", linewidths=0.5
    )
    if clim:
        cbar.set_clim(clim[0], clim[1])
    else:
        cbar.set_clim(-1, 1)
    ax[0].set_aspect(1)

    if custom_zoom is not None and not custom_axis:
        x2 = np.linspace(custom_zoom["x"][0], custom_zoom["x"][1], num=res)
        y2 = np.linspace(custom_zoom["y"][0], custom_zoom["y"][1], num=res)
        X2, Y2 = np.meshgrid(x2, y2)
        Z2 = np.zeros_like(X2) + 0
        sdf2 = fun(np.vstack([X2.flatten(), Y2.flatten(), Z2.flatten()]).T).reshape(
            X2.shape
        )

        # cbar = ax[1].scatter(X2, Y2, c=sdf2, cmap="seismic")
        cbar = ax[1].contour(
            X2 * scale[0], Y2 * scale[0], sdf2, levels=[0], colors="seismic"
        )
        if clim:
            cbar.set_clim(clim[0], clim[1])
        else:
            cbar.set_clim(-1, 1)
        ax[1].set_aspect(1)
    if plt_show:
        plt.show()


def scatter_contour_at_origin(
    fun,
    origin=(0, 0, 0),
    normal=(0, 0, 1),
    res=100,
    custom_axis=None,
    eval_area=(-1, 1),
    scale=1,
    custom_zoom=None,
    clim=None,
    flip_axes=False,
    cmap="seismic",
    show_zero_level=True,
):
    """
    example for custom zoom:
        custom_zoom={"x": [0.25, 0.75],
                        "y": [-0.25, -0.75]}
    """
    plt_show = True
    if custom_axis:
        ax = [custom_axis]
        plt_show = False
    elif custom_zoom is not None:
        _, ax = plt.subplots(1, 2)
    else:
        _, ax = plt.subplots(1, 1)
        ax = [ax]

    spacing = (eval_area[1] - eval_area[0]) / res  # Distance between points

    points = generate_plane_points(origin, normal, res, res, spacing)
    #     x = np.linspace(eval_area[0], eval_area[1], num=res)
    # y = np.linspace(eval_area[0], eval_area[1], num=res)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X) + z_level
    points = torch.from_numpy(points)
    sdf = fun(points).reshape((res, res))
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    # sdf = fun(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T).reshape(X.shape)
    # X = points[:,0].reshape((num_points, num_points))
    # Y = points[:,1].reshape((num_points, num_points))
    if flip_axes:
        sdf = sdf.T

    # cbar = ax[0].scatter(X, Y, c=sdf, cmap="seismic")c
    cbar = ax[0].contourf(sdf, cmap=cmap, levels=10)
    if show_zero_level:
        ax[0].contour(sdf, levels=[0], colors="black", linewidths=0.5)
    if clim:
        cbar.set_clim(clim[0], clim[1])
    else:
        cbar.set_clim(-1, 1)
    ax[0].set_aspect(scale)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    if custom_zoom is not None and not custom_axis:
        x2 = np.linspace(custom_zoom["x"][0], custom_zoom["x"][1], num=res)
        y2 = np.linspace(custom_zoom["y"][0], custom_zoom["y"][1], num=res)
        X2, Y2 = np.meshgrid(x2, y2)
        Z2 = np.zeros_like(X2) + 0
        sdf2 = fun(np.vstack([X2.flatten(), Y2.flatten(), Z2.flatten()]).T).reshape(
            X2.shape
        )

        # cbar = ax[1].scatter(X2, Y2, c=sdf2, cmap="seismic")
        cbar = ax[1].contour(X2, Y2, sdf2, levels=[0], colors="black")
        if clim:
            cbar.set_clim(clim[0], clim[1])
        else:
            cbar.set_clim(-1, 1)
        ax[1].set_aspect(scale)

    if plt_show:
        plt.show()


def generate_plane_points(origin, normal, num_points_u, num_points_v, spacing):
    """
    Generates evenly spaced points on a plane.

    Parameters:
    origin (array-like): A point on the plane (3D vector).
    normal (array-like): Normal vector of the plane (3D vector).
    num_points_u (int): Number of points along the first direction (u-axis).
    num_points_v (int): Number of points along the second direction (v-axis).
    spacing (float): Distance between adjacent points.

    Returns:
    points (numpy.ndarray): Array of points on the plane of shape (num_points_u * num_points_v, 3).
    """

    # Ensure the normal is a unit vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    # Find two orthogonal vectors to the normal that lie on the plane (u and v axes)
    if np.allclose(normal, [0, 0, 1]):  # Special case when the normal is along z-axis
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    else:
        u = np.cross(
            [0, 0, 1], normal
        )  # Cross product to get a vector orthogonal to normal
        u = u / np.linalg.norm(u)  # Normalize the vector
        v = np.cross(normal, u)  # v is orthogonal to both normal and u

    # Create grid points in 2D space (u-v plane)
    u_coords = (
        np.linspace(-num_points_u // 2, num_points_u // 2, num_points_u) * spacing
    )
    v_coords = (
        np.linspace(-num_points_v // 2, num_points_v // 2, num_points_v) * spacing
    )

    points = []

    for u_val in u_coords:
        for v_val in v_coords:
            point = origin + u_val * u + v_val * v
            points.append(point)

    return np.array(points)

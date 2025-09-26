import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_slice(
    fun,
    origin=(0, 0, 0),
    normal=(0, 0, 1),
    res=(100, 100),
    ax=None,
    xlim=(-1, 1),
    ylim=(-1, 1),
    clim=(-1, 1),
    cmap="seismic",
    show_zero_level=True,
):
    plt_show = False
    if ax is None:
        fig, ax = plt.subplots()
        plt_show = True

    points, u, v = generate_plane_points(origin, normal, res, xlim, ylim)

    points = torch.from_numpy(points).to(torch.float32)
    sdf = fun(points).reshape((res[0], res[1]))
    X = u.reshape((res[0], res[1]))
    Y = v.reshape((res[0], res[1]))
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()

    # cbar = ax[0].scatter(X, Y, c=sdf, cmap="seismic")c
    cbar = ax.contourf(X, Y, sdf, cmap=cmap, levels=10)
    if show_zero_level:
        ax.contour(X, Y, sdf, levels=[0], colors="black", linewidths=0.5)
    cbar.set_clim(clim[0], clim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    if plt_show:
        plt.show()
        return fig, ax


def generate_plane_points(origin, normal, res, xlim, ylim):
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
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    origin = np.array(origin)
    # Find two orthogonal vectors to the normal that lie on the plane (u and v axes)
    if np.allclose(normal, [0, 0, 1]):  # Special case when the normal is along z-axis
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    elif np.allclose(normal, [0, 1, 0]):  # Special case when the normal is along z-axis
        u = np.array([1, 0, 0])
        v = np.array([0, 0, 1])
    elif np.allclose(normal, [1, 0, 0]):  # Special case when the normal is along z-axis
        u = np.array([0, 1, 0])
        v = np.array([0, 0, 1])
    else:
        raise NotImplementedError(
            "Normal vector other than [1,0,0], [0,1,0] and [0,0,1] not supported yet."
        )

    # Create grid points in 2D space (u-v plane)
    u_coords = np.linspace(xlim[0], xlim[1], res[0])
    v_coords = np.linspace(ylim[0], ylim[1], res[1])

    points = []
    u_exp = []
    v_exp = []
    for u_val in u_coords:
        for v_val in v_coords:
            point = origin + u_val * u + v_val * v
            u_exp.append(u_val)
            v_exp.append(v_val)
            points.append(point)

    return np.array(points), np.array(u_exp), np.array(v_exp)

"""
Visualization and Plotting Utilities
=====================================

This module provides utilities for visualizing SDF representations, particularly
for creating 2D cross-sectional views of 3D signed distance functions.

Functions
---------
plot_slice
    Create a contour plot of an SDF on a 2D plane slice.
generate_plane_points
    Generate a regular grid of points on a plane in 3D space.

The primary use case is visualizing SDF values on axis-aligned planes
to understand the geometry and verify correctness during development
and debugging.
"""

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
    """Plot a 2D slice through an SDF as a contour plot.
    
    This function evaluates an SDF on a planar grid and visualizes the
    signed distance values using a color map. The zero level set (the
    actual surface) can be highlighted with a contour line.
    
    Parameters
    ----------
    fun : callable
        The SDF function to visualize. Should accept a torch.Tensor
        of shape (N, 3) and return distances of shape (N, 1).
    origin : tuple of float, default (0, 0, 0)
        A point on the slice plane.
    normal : tuple of float, default (0, 0, 1)
        Normal vector of the slice plane. Currently supports only
        axis-aligned planes: (1,0,0), (0,1,0), or (0,0,1).
    res : tuple of int, default (100, 100)
        Resolution of the slice grid (num_points_u, num_points_v).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    xlim : tuple of float, default (-1, 1)
        Range along the first plane axis.
    ylim : tuple of float, default (-1, 1)
        Range along the second plane axis.
    clim : tuple of float, default (-1, 1)
        Color map limits for distance values.
    cmap : str, default 'seismic'
        Matplotlib colormap name.
    show_zero_level : bool, default True
        If True, draws a black contour line at distance=0 (the surface).
        
    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Only returned if ax was None (i.e., a new figure was created).
        
    Examples
    --------
    >>> from DeepSDFStruct.sdf_primitives import SphereSDF
    >>> from DeepSDFStruct.plotting import plot_slice
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Create a sphere
    >>> sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
    >>> 
    >>> # Plot XY slice at z=0
    >>> fig, ax = plot_slice(
    ...     sphere,
    ...     origin=(0, 0, 0),
    ...     normal=(0, 0, 1),
    ...     res=(200, 200)
    ... )
    >>> plt.title("XY Slice of Sphere")
    >>> plt.show()
    
    Notes
    -----
    The 'seismic' colormap is well-suited for SDFs as it uses blue for
    negative (inside) and red for positive (outside), with white near zero.
    """
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
    """Generate evenly spaced points on a plane in 3D space.
    
    Creates a regular grid of points on a plane defined by a point and normal
    vector. The grid is axis-aligned in the plane's local coordinate system.
    
    Parameters
    ----------
    origin : array-like of shape (3,)
        A point on the plane (3D vector).
    normal : array-like of shape (3,)
        Normal vector of the plane (3D vector). Currently supports only
        axis-aligned normals: [1,0,0], [0,1,0], or [0,0,1].
    res : tuple of int
        Grid resolution (num_points_u, num_points_v).
    xlim : tuple of float
        Range along the first plane axis (umin, umax).
    ylim : tuple of float
        Range along the second plane axis (vmin, vmax).
        
    Returns
    -------
    points : np.ndarray of shape (num_points_u * num_points_v, 3)
        3D coordinates of grid points.
    u : np.ndarray of shape (num_points_u * num_points_v,)
        First plane coordinate for each point.
    v : np.ndarray of shape (num_points_u * num_points_v,)
        Second plane coordinate for each point.
        
    Raises
    ------
    NotImplementedError
        If normal is not axis-aligned.
        
    Examples
    --------
    >>> from DeepSDFStruct.plotting import generate_plane_points
    >>> import numpy as np
    >>> 
    >>> # Generate points on XY plane at z=0.5
    >>> points, u, v = generate_plane_points(
    ...     origin=[0, 0, 0.5],
    ...     normal=[0, 0, 1],
    ...     res=(10, 10),
    ...     xlim=(-1, 1),
    ...     ylim=(-1, 1)
    ... )
    >>> print(points.shape)  # (100, 3)
    >>> print(np.allclose(points[:, 2], 0.5))  # True (all on z=0.5 plane)
    
    Notes
    -----
    The function determines two orthogonal axes (u and v) in the plane
    based on the normal vector. For axis-aligned planes:
    - Normal [0,0,1] (XY plane): u=[1,0,0], v=[0,1,0]
    - Normal [0,1,0] (XZ plane): u=[1,0,0], v=[0,0,1]
    - Normal [1,0,0] (YZ plane): u=[0,1,0], v=[0,0,1]
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

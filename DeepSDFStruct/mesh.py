"""
Mesh Generation and Processing
===============================

This module provides comprehensive tools for generating, processing, and exporting
meshes from SDF representations. It supports both surface (triangle) meshes and
volume (tetrahedral) meshes, with advanced algorithms for high-quality mesh extraction.

Key Capabilities
----------------

Mesh Extraction
    - FlexiCubes: State-of-the-art dual contouring for smooth, feature-preserving 3D meshes
    - FlexiSquares: 2D mesh extraction for cross-sections and planar geometries
    - Marching Cubes: Traditional isosurface extraction (via skimage)

Mesh Processing
    - Tetrahedral meshing for finite element analysis
    - Mesh cleanup and repair (disconnected regions, degenerate elements)
    - Mesh decimation and simplification
    - Normal computation and smoothing

Export Formats
    - VTK (.vtk) for visualization in ParaView
    - Abaqus (.inp) for finite element analysis
    - PLY (.ply) for general 3D interchange
    - MFEM format for MFEM solvers

The module provides both PyTorch-based mesh representations (torchLineMesh,
torchSurfMesh, torchVolumeMesh) for differentiable operations and conversion
to standard formats (gustaf, trimesh) for I/O and visualization.
"""

import logging
import torch as _torch
import torch.autograd.functional
from torch.func import functional_call, jacrev, jacfwd
import tetgenpy
import numpy as np
import napf
import gustaf as gus
import pathlib
import os
import skimage
import triangle
import trimesh
import vtk
from typing import Optional, Tuple, Union
import scipy
import scipy.sparse.csgraph

from functools import partial

from DeepSDFStruct.flexicubes.flexicubes import FlexiCubes
from DeepSDFStruct.flexisquares.flexisquares import FlexiSquares
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.SDF import SDFBase
import DeepSDFStruct

logger = logging.getLogger(DeepSDFStruct.__name__)


class torchLineMesh:
    """PyTorch-based line mesh representation for differentiable operations.

    Stores line segments with vertices and connectivity, supporting
    gradient propagation for optimization tasks.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex coordinates of shape (N, 3).
    lines : torch.Tensor
        Line connectivity of shape (M, 2), where each row contains
        indices into the vertices array.

    Methods
    -------
    to_gus()
        Convert to gustaf Edges format for visualization and I/O.
    """

    def __init__(self, vertices: _torch.Tensor, lines: _torch.Tensor):
        self.vertices = vertices
        self.lines = lines

    def to_gus(self):
        return gus.Edges(self.vertices.detach().cpu(), self.lines.detach().cpu())

    def triangulate(self, x_nx2, s_n, bbox_vertices=None, tolerance=0.05):
        holes = x_nx2[torch.where(s_n > tolerance)[0], :]
        # N_elements = 1000 * np.prod(tiling)
        # surf_area = 2
        # max_a = round(surf_area / N_elements, 5)
        # max_a = 1e-4

        # Compute bounding box
        xmin, ymin = x_nx2.min(dim=0).values.detach()
        xmax, ymax = x_nx2.max(dim=0).values.detach()

        # Add 4 corner vertices for the bounding rectangle
        bbox_vertices = torch.tensor(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            dtype=self.vertices.dtype,
            device=self.vertices.device,
        )

        # Offset for new vertex indices
        offset = self.vertices.shape[0]
        vertices_with_boundary = torch.cat([self.vertices, bbox_vertices], dim=0)

        # Create segments for the rectangle boundary
        bbox_segments = torch.tensor(
            [
                [offset + 0, offset + 1],
                [offset + 1, offset + 2],
                [offset + 2, offset + 3],
                [offset + 3, offset + 0],
            ],
            dtype=self.lines.dtype,
            device=self.lines.device,
        )

        # Combine with original segments
        segments_with_boundary = torch.cat([self.lines, bbox_segments], dim=0)

        # compute the mean side length of the first 100 line segments
        first_100_lines = self.lines[: min(100, self.lines.shape[0]), :]
        first_100_line_vertices = self.vertices[first_100_lines]
        mean_side_length = torch.norm(
            first_100_line_vertices[:, 0, :] - first_100_line_vertices[:, 1, :], dim=1
        ).mean()

        a = (np.sqrt(3) / 4) * mean_side_length.item() ** 2
        triangle_string = f"pqa{a}"

        A = dict(
            vertices=vertices_with_boundary.detach().cpu().numpy(),
            segments=segments_with_boundary.detach().cpu().numpy(),
            holes=holes.detach().cpu().numpy(),
        )

        logger.info("Calling triangulate with " + triangle_string)
        B = triangle.triangulate(A, triangle_string)

        return torchSurfMesh(
            torch.tensor(
                B["vertices"], dtype=self.vertices.dtype, device=self.vertices.device
            ),
            torch.tensor(
                B["triangles"], device=self.vertices.device, dtype=self.lines.dtype
            ),
        )


class torchSurfMesh:
    """PyTorch-based surface mesh representation for differentiable operations.

    Stores triangle mesh with vertices and face connectivity, supporting
    gradient propagation for shape optimization and learning tasks.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex coordinates of shape (N, 3).
    faces : torch.Tensor
        Triangle face connectivity of shape (M, 3), where each row
        contains indices into the vertices array.

    Methods
    -------
    to_gus()
        Convert to gustaf Faces format for visualization and I/O.
    """

    def __init__(self, vertices: _torch.Tensor, faces: _torch.Tensor):
        self.vertices = vertices
        self.faces = faces

    def to_gus(self):
        return gus.Faces(self.vertices.detach().cpu(), self.faces.detach().cpu())

    def to_trimesh(self):
        gus_mesh = self.to_gus()
        return trimesh.Trimesh(gus_mesh.vertices, gus_mesh.faces)

    def clean(self, clean_jacobian=True):
        """
        Remove unused vertices and remap face indices.

        Returns
        -------
        torchSurfMesh
            A new cleaned mesh with only the vertices used by faces.
        """
        n_elements_orig = self.faces.shape[0]
        n_vertices_orig = self.vertices.shape[0]
        if clean_jacobian:
            v0 = self.vertices[self.faces[:, 0]]
            v1 = self.vertices[self.faces[:, 1]]
            v2 = self.vertices[self.faces[:, 2]]

            if self.vertices.shape[1] == 2:
                # 2D Jacobian: signed area of triangle
                # det( [v1-v0, v2-v0] ) = e1.x*e2.y - e1.y*e2.x
                e1 = v1 - v0
                e2 = v2 - v0
                jac_det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            elif self.vertices.shape[1] == 3:
                # 3D Jakob: sign from oriented area using cross product direction
                e1 = v1 - v0
                e2 = v2 - v0
                cross = torch.cross(e1, e2, dim=-1)  # (F,3)
                # Signed Jacobian: oriented area relative to a stable axis
                # Use the largest-magnitude axis to avoid degeneracy
                abs_cross = cross.abs()
                axis = abs_cross.argmax(dim=1)  # (F,)

                signed_area = cross[torch.arange(cross.size(0)), axis]
                jac_det = signed_area  # sign preserved

            # remove elements with zero surface area
            valid_mask = jac_det > 0
            self.faces = self.faces[valid_mask]

        # Find unique vertices referenced by faces
        used_idx = _torch.unique(self.faces.reshape(-1))

        # Build mapping from old indices → new compacted indices
        new_index = -_torch.ones(
            self.vertices.shape[0], dtype=_torch.long, device=self.vertices.device
        )
        new_index[used_idx] = _torch.arange(
            used_idx.numel(), device=self.vertices.device
        )

        # Compact vertex array
        new_vertices = self.vertices[used_idx]

        # Remap face indices
        new_faces = new_index[self.faces]
        n_elements_after = new_faces.shape[0]
        n_vertices_after = new_vertices.shape[0]
        info_string = ""
        if n_elements_after < n_elements_orig:
            info_string += f"removed {n_elements_orig-n_elements_after} elements"
        if n_vertices_after < n_vertices_orig:
            info_string += f" removed {n_vertices_orig-n_vertices_after} vertices"
        if info_string != "":
            logger.info(info_string)
        return torchSurfMesh(new_vertices, new_faces)


class torchVolumeMesh:
    """PyTorch-based volume mesh representation for differentiable operations.

    Stores tetrahedral mesh with vertices and element connectivity, supporting
    gradient propagation through finite element analysis and other volume-based
    operations.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex coordinates of shape (N, 3).
    volumes : torch.Tensor
        Tetrahedral element connectivity of shape (M, 4), where each row
        contains indices into the vertices array.

    Methods
    -------
    to_gus()
        Convert to gustaf Volumes format for visualization and I/O.
    remove_disconnected_regions(support_node, clear_unused)
        Remove disconnected mesh components, optionally keeping only
        the region connected to a specific node.
    """

    def __init__(self, vertices: _torch.Tensor, volumes: _torch.Tensor):
        self.vertices = vertices
        self.volumes = volumes

    def to_gus(self):
        return gus.Volumes(self.vertices.detach().cpu(), self.volumes.detach().cpu())

    def to_trimesh(self):
        gus_mesh = self.to_gus()
        return trimesh.Trimesh(gus_mesh.vertices, gus_mesh.volumes)

    def remove_disconnected_regions(
        self, support_node: int | None = None, clear_unused=True
    ):
        """Remove disconnected parts from the mesh.

        Uses graph connectivity analysis to identify and remove mesh regions
        that are not connected to the main component or a specified support node.
        Useful for cleaning up meshes after optimization or level set extraction.

        Parameters
        ----------
        support_node : int, optional
            If specified, keeps only the connected component containing this
            vertex index. If None, keeps the largest component.
        clear_unused : bool, default True
            If True, removes unreferenced vertices after pruning elements.

        Raises
        ------
        NotImplementedError
            If mesh contains non-tetrahedral elements.
        ValueError
            If support_node is out of range.

        Notes
        -----
        Currently only supports tetrahedral elements (4 nodes per element).
        For now, unreferenced vertices are kept even when clear_unused=True.
        """
        if self.volumes.shape[1] != 4:
            raise NotImplementedError("Cleanup only supports tetrahedral elements yet.")
        if support_node is not None:
            if support_node > self.vertices.shape[0]:
                raise ValueError(
                    "Support node must be part of vertices. "
                    "Support Node: {support_node}, N Vertices: {self.vertices.shape[0]}"
                )
        edges = torch.cat(
            [
                self.volumes[:, [0, 1]],
                self.volumes[:, [1, 2]],
                self.volumes[:, [2, 3]],
                self.volumes[:, [3, 0]],
            ],
            dim=0,
        )

        # Make it undirected
        edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
        num_nodes = self.vertices.shape[0]
        row, col = edges.T.cpu().numpy()
        data = np.ones(len(row), dtype=np.int8)
        adj = scipy.sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        # Compute connected components
        num_comp, component_labels = scipy.sparse.csgraph.connected_components(
            adj, directed=False
        )
        if support_node is None:
            values, counts = np.unique(component_labels, return_counts=True)
            sorted_idx = np.argsort(-counts)  # negative for descending

            most_frequent = values[sorted_idx[0]]
            if values.shape[0] < 2:
                logger.info("No disconnected regions found")
                return
            second_most_frequent = values[sorted_idx[1]]

            # safety check that the second most frequent is not considerably large
            if counts[most_frequent] < (counts[second_most_frequent] * 1.5):
                logger.warning(
                    f"Main body contains {counts[most_frequent]} nodes and the "
                    f"second largest {counts[second_most_frequent]}. "
                    "Conside Adding a support node."
                )
            remaining_body = most_frequent
        else:
            remaining_body = component_labels[support_node]

        valid_node_ids = torch.arange(self.vertices.shape[0])[
            component_labels == remaining_body
        ]
        mask = torch.isin(self.volumes, valid_node_ids)
        row_mask = mask.all(dim=1)
        removed = (~row_mask).sum()
        logger.info(f"Removed {removed} elements from disconnected regions.")
        self.volumes = self.volumes[row_mask]
        if clear_unused:
            self.clear_unreferenced_nodes()

    def clear_unreferenced_nodes(self):
        used_nodes, inverse = torch.unique(self.volumes, return_inverse=True)
        remapped_volumes = inverse.view(self.volumes.shape)
        self.vertices = self.vertices[used_nodes]
        self.volumes = remapped_volumes


def tetrahedralize_surface(surface_mesh: gus.Faces) -> tuple[gus.Volumes, np.ndarray]:
    logger.debug("Tetrahedralizing surface mesh")
    t_in = tetgenpy.TetgenIO()
    t_in.setup_plc(surface_mesh.vertices, surface_mesh.faces.tolist())
    # gus.show(dmesh)
    switch_command = "pYq"
    if logging.DEBUG <= logging.root.level:
        switch_command += "Q"
    t_out = tetgenpy.tetrahedralize(switch_command, t_in)  # pqa

    tets = np.vstack(t_out.tetrahedra())
    verts = t_out.points()

    kdt = napf.KDT(tree_data=verts, metric=1)

    distances, face_indices = kdt.knn_search(
        queries=surface_mesh.vertices, kneighbors=1, nthread=4
    )
    tol = 1e-6
    if distances.max() > tol:
        Warning("Not all surface nodes as included in the volumetric mesh.")
    volumes = gus.Volumes(verts, tets)
    surface_mesh_indices = face_indices
    return volumes, surface_mesh_indices


def export_volume_mesh(volume_mesh: gus.Volumes, filename: str, export_abaqus=False):
    """
    export a mesh and adds corresponding boundary conditions
    """
    filepath = pathlib.Path(filename)
    if not os.path.isdir(filepath.parent):
        os.makedirs(filepath.parent)
    logger.debug(
        f"Exporting mesh with {len(volume_mesh.volumes)} elements, {len(volume_mesh.vertices)} vertices to {filepath}"
    )
    gus.io.mfem.export(str(filepath), volume_mesh)


def export_abaqus_surf_mesh(surf_mesh: gus.Faces, filename: str):
    """
    export a mesh and adds corresponding boundary conditions
    """
    filepath = pathlib.Path(filename)
    if not os.path.isdir(filepath.parent):
        os.makedirs(filepath.parent)
    gus.io.meshio.export(str(filepath.with_suffix(".inp")), surf_mesh)


def generate_2D_surf_mesh(
    sdf: SDFBase, n_squares: int, n_elements: int = 50000, bounds=None
):
    n_points = 1000
    if bounds is None:
        bounds = sdf._get_domain_bounds()
    x = np.linspace(bounds[0, 0], bounds[1, 0], n_points)
    y = np.linspace(bounds[0, 1], bounds[1, 1], n_points)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    queries = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    evaluated_sdf = sdf(queries)
    evaluated_sdf_orig_shape = evaluated_sdf.reshape(xx.shape)
    A = sdf_to_triangle_dict(x, y, evaluated_sdf_orig_shape, level=0)
    domain_area = (bounds[1, 0] - bounds[0, 0]) * (bounds[1, 1] - bounds[0, 1])
    min_a = domain_area / n_elements
    triangle_string = f"pqa{min_a}"
    logger.info("Calling triangulate with " + triangle_string)
    B = triangle.triangulate(A, triangle_string)
    mesh = gus.Faces(vertices=B["vertices"], faces=B["triangles"])
    return mesh


def sdf_to_triangle_dict(x, y, sdf, level=0.0, pad_value=1.0, collinear_tol=1e-8):
    """
    Convert an SDF array to a PSLG dict for triangle.triangulate.
    Includes domain boundary and hole polygons.

    Args:
        sdf (np.ndarray): 2D signed distance field.
        level (float): Contour level (usually 0 for boundary).
        pad_value (float): Value outside domain to close boundaries.

    Returns:
        dict: {"vertices": ..., "segments": ..., "holes": ...}
    """
    H, W = sdf.shape

    # Pad so contours touching domain edge get closed
    padded = np.pad(sdf, 1, mode="constant", constant_values=pad_value)
    contours = skimage.measure.find_contours(padded, level)
    contours = [c - 1 for c in contours]  # shift back after padding

    vertices = []
    segments = []
    holes = []

    v_offset = 0

    for contour in contours:
        # contour[:, 0] = row indices in [0, H), map to y
        # contour[:, 1] = col indices in [0, W), map to x
        poly_x = np.interp(contour[:, 1], np.arange(W), x)
        poly_y = np.interp(contour[:, 0], np.arange(H), y)
        poly = np.column_stack([poly_x, poly_y])
        poly = prune_collinear(poly, tol=collinear_tol)

        n = len(poly)
        vertices.extend(poly.tolist())
        segments.extend([[v_offset + i, v_offset + (i + 1) % n] for i in range(n)])

        # Orientation heuristic
        area = 0.5 * np.sum(
            poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1)
        )
        if area > 0:
            centroid = poly.mean(axis=0)
            holes.append(centroid.tolist())

        v_offset += n

    _, unique_indices, counts = np.unique(
        vertices, axis=0, return_index=True, return_counts=True
    )
    duplicate_indices = np.where(counts > 1)[0]

    if len(duplicate_indices) > 0:
        logger.info(f"Found {len(duplicate_indices)} duplicate vertices:")
        for idx in duplicate_indices:
            logger.debug(vertices[idx])
    else:
        print("No duplicate vertices found.")

    vertices, idx = np.unique(vertices, axis=0, return_inverse=True)
    segments = np.array([[idx[s[0]], idx[s[1]]] for s in segments])

    return dict(
        vertices=np.array(vertices), segments=np.array(segments), holes=np.array(holes)
    )


def prune_collinear(points, tol=1e-9):
    """
    Remove nearly collinear points from a polyline (closed loop).
    Args:
        points (ndarray): Nx2 array of (x,y) vertices.
        tol (float): area tolerance. Smaller = stricter (keep more points).
    Returns:
        ndarray: pruned Nx2 array of vertices.
    """
    if len(points) <= 3:
        return points  # nothing to prune

    pruned = [points[0]]
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        # Compute twice the triangle area formed by (a,b,c)
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area > tol:
            pruned.append(b)
    pruned.append(points[-1])
    return np.array(pruned)


def process_N_base_input(N, tiling, dim=3):
    if isinstance(N, list):
        if len(N) != dim:
            raise ValueError("Number of grid points must be a list of 3 integers")
        N = _torch.tensor(N)
    elif isinstance(N, int):
        N = _torch.tensor([N] * dim)
    else:
        raise ValueError("Number of grid points must be a list or an integer")
    # add 1 on each side to slightly include the border
    N_mod = N * tiling + 1
    return N_mod


def _prepare_flexicubes_querypoints(
    N, device=None, bounds=None, extr_type="flexicubes"
):
    """
    takes the tiling and a resolution as input
    output: DeepSDFStruct.flexicubes constructor, samples and cube indices
            the points are located in the region [0,1] with a margin of 0.025
            -> [-0.025, 1.025]
    """
    # check_tiling_input(tiling)
    if extr_type not in ["flexicubes", "flexisquares"]:
        raise TypeError(
            f"Argument extr_type must be either flexicubes or flexisquares, not {extr_type}"
        )
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    if extr_type == "flexicubes":
        flexi_cubes_constructor = FlexiCubes(device=device)
    else:
        flexi_cubes_constructor = FlexiSquares(device=device)

    samples, cube_idx = flexi_cubes_constructor.construct_voxel_grid(
        resolution=tuple(N), bounds=bounds
    )

    return flexi_cubes_constructor, samples, cube_idx


def create_3D_mesh(
    sdf: SDFBase,
    N_base,
    mesh_type: str,
    differentiate=False,
    device="cpu",
    bounds=None,
    diffmode="fwd",
) -> Tuple[Union[torchSurfMesh, torchVolumeMesh], Optional[_torch.Tensor]]:
    """Generate a 3D mesh from an SDF using FlexiCubes dual contouring.

    This is the main entry point for extracting high-quality meshes from SDF
    representations. It uses the FlexiCubes algorithm, which produces smooth,
    feature-preserving meshes that are superior to marching cubes.

    The function supports both surface meshes (triangles) and volume meshes
    (tetrahedra) and can compute gradients for optimization if requested.

    Parameters
    ----------
    sdf : SDFBase
        The signed distance function to mesh. Can be any SDFBase subclass
        including primitives, lattice structures, or learned representations.
    N_base : int or list of int
        Base resolution per unit cell. If an int, uses the same resolution
        in all dimensions. If a list, specifies resolution [nx, ny, nz].
        For lattice structures, the total resolution is N_base * tiling.
    mesh_type : {'surface', 'volume'}
        Type of mesh to generate:
        - 'surface': Triangle surface mesh (torchSurfMesh)
        - 'volume': Tetrahedral volume mesh (torchVolumeMesh)
    differentiate : bool, default False
        If True, computes and returns gradients of vertex positions with
        respect to SDF parameters, enabling gradient-based optimization.
    device : str, default 'cpu'
        Device for computation ('cpu' or 'cuda').
    bounds : array-like of shape (2, 3), optional
        Spatial bounds [[xmin, ymin, zmin], [xmax, ymax, zmax]].
        If None, uses the SDF's domain bounds.

    Returns
    -------
    mesh : torchSurfMesh or torchVolumeMesh
        The extracted mesh with vertices and connectivity.
    dVerts_dParams : torch.Tensor or None
        If differentiate=True, returns gradients of vertex positions with
        respect to parameters. Otherwise returns None.

    Examples
    --------
    >>> from DeepSDFStruct.sdf_primitives import SphereSDF
    >>> from DeepSDFStruct.mesh import create_3D_mesh
    >>>
    >>> # Create a sphere SDF
    >>> sphere = SphereSDF(center=[0, 0, 0], radius=1.0)
    >>>
    >>> # Extract surface mesh
    >>> mesh, _ = create_3D_mesh(sphere, N_base=64, mesh_type='surface')
    >>> print(f"Vertices: {mesh.vertices.shape}")
    >>> print(f"Faces: {mesh.faces.shape}")
    >>>
    >>> # Extract volume mesh for FEA
    >>> vol_mesh, _ = create_3D_mesh(sphere, N_base=32, mesh_type='volume')
    >>> print(f"Tetrahedra: {vol_mesh.volumes.shape}")

    Notes
    -----
    FlexiCubes produces higher quality meshes than marching cubes by:
    - Using flexible vertex positions within each cube
    - Preserving sharp features and corners
    - Minimizing mesh artifacts and degeneracies

    For lattice structures, the resolution is automatically scaled by the
    tiling factor to maintain consistent resolution per unit cell.
    """
    lattice = find_lattice_sdf(sdf)
    if lattice is None:
        tiling = _torch.tensor([1, 1, 1])
    else:
        tiling = _torch.tensor(lattice.tiling)

    if mesh_type == "surface":
        output_tetmesh = False
    elif mesh_type == "volume":
        output_tetmesh = True
    else:
        raise RuntimeError(
            f"mesh_type {mesh_type} unavailable. Must be either surface or volume"
        )

    if bounds is None:
        bounds = sdf._get_domain_bounds()
    extended_bounds = bounds.clone()
    off = (extended_bounds[1] - extended_bounds[0]) * 0.05
    extended_bounds[0] -= off
    extended_bounds[1] += off

    N = process_N_base_input(N_base, tiling)

    constructor, samples, cube_idx = _prepare_flexicubes_querypoints(
        N, device=device, bounds=extended_bounds
    )
    dVerts_dParams = None

    buffers = dict(sdf.named_buffers())
    verts_fn = partial(
        _verts_from_params,
        buffers=buffers,
        sdf=sdf,
        samples=samples,
        constructor=constructor,
        cube_idx=cube_idx,
        N=N,
        return_faces=False,
        output_tetmesh=output_tetmesh,
    )
    # returns faces or volumes depending on the output_tetmesh flag
    # if output_tetmesh -> returns volumes
    # if not output_tetmesh -> returns faces
    verts, faces_or_volumes = get_verts(
        sdf=sdf,
        samples=samples,
        constructor=constructor,
        cube_idx=cube_idx,
        N=N,
        return_faces=True,
        output_tetmesh=output_tetmesh,
    )
    if differentiate:
        if diffmode == "rev":
            dVerts_dParams = jacrev(verts_fn)(dict(sdf.named_parameters()))
        elif diffmode == "fwd":
            dVerts_dParams = jacfwd(verts_fn)(dict(sdf.named_parameters()))
        else:
            raise NotImplementedError("diffmode must be either fwd or rev")
    if output_tetmesh:
        return torchVolumeMesh(verts, faces_or_volumes), dVerts_dParams
    else:
        return torchSurfMesh(verts, faces_or_volumes), dVerts_dParams


def find_lattice_sdf(module) -> LatticeSDFStruct | None:
    for m in module.modules():
        if isinstance(m, LatticeSDFStruct):
            return m
    return None


def create_2D_mesh(
    sdf: SDFBase,
    N_base,
    mesh_type: str,
    differentiate=False,
    device=None,
    bounds=None,
    diffmode="fwd",
) -> Tuple[Union[torchLineMesh, torchSurfMesh], Optional[torch.Tensor]]:

    if device is not None:
        print("Warning: the argument device is no longer used.")

    lattice = find_lattice_sdf(sdf)
    if lattice is None:
        tiling = _torch.tensor([1, 1])
    else:
        tiling = _torch.tensor(lattice.tiling)

    if mesh_type in ["line", "surface_triangle"]:
        output_tetmesh = False
    elif mesh_type == "surface":
        output_tetmesh = True
    else:
        raise RuntimeError(
            f"mesh_type {mesh_type} unavailable. Must be either line or surface"
        )

    N = process_N_base_input(N_base, tiling, dim=2)

    constructor, samples, cube_idx = _prepare_flexicubes_querypoints(
        N, device=sdf.get_device(), bounds=bounds, extr_type="flexisquares"
    )
    dVerts_dParams = None

    buffers = dict(sdf.named_buffers())
    verts_fn = partial(
        _verts_from_params,
        buffers=buffers,
        sdf=sdf,
        samples=samples,
        constructor=constructor,
        cube_idx=cube_idx,
        N=N,
        return_faces=False,
        output_tetmesh=output_tetmesh,
    )
    # returns faces or volumes depending on the output_tetmesh flag
    # if output_tetmesh -> returns volumes
    # if not output_tetmesh -> returns faces
    sdf_values = sdf(samples)

    verts_local, faces_or_volumes, _ = constructor(
        voxelgrid_vertices=samples,
        scalar_field=sdf_values.reshape(-1),
        cube_idx=cube_idx,
        resolution=tuple(N),
        output_tetmesh=output_tetmesh,
    )

    if sdf.deformation_spline is not None:
        verts_local = sdf.deformation_spline.forward(verts_local)
        with torch.no_grad():
            samples_deformed = sdf.deformation_spline.forward(samples)

    if differentiate:
        if diffmode == "rev":
            dVerts_dParams = jacrev(verts_fn)(dict(sdf.named_parameters()))
        elif diffmode == "fwd":
            dVerts_dParams = jacfwd(verts_fn)(dict(sdf.named_parameters()))
        else:
            raise NotImplementedError("diffmode must be either fwd or rev")

    if mesh_type == "line":
        return torchLineMesh(verts_local, faces_or_volumes), dVerts_dParams
    elif mesh_type == "surface":
        return torchSurfMesh(verts_local, faces_or_volumes), dVerts_dParams
    elif mesh_type == "surface_triangle":
        line_mesh = torchLineMesh(verts_local, faces_or_volumes)
        gus.io.meshio.export("lines.inp", line_mesh.to_gus())
        surf_mesh = line_mesh.triangulate(samples_deformed, sdf_values.reshape(-1))
        gus.io.meshio.export(
            "surfs_straight_after_triangulation.inp", surf_mesh.to_gus()
        )
        surf_mesh.vertices.requires_grad = True
        dist = torch.cdist(
            surf_mesh.vertices.double(),
            line_mesh.vertices.double(),
            compute_mode="use_mm_for_euclid_dist",
        )  # (N, M)

        # find nearest original vertex index
        min_dist, min_idx = dist.min(dim=1)
        # mask where surf vertex matches original vertex
        mask = min_dist < 1e-5  # or some tolerance
        if mask.sum() != line_mesh.vertices.shape[0]:
            raise ValueError("Not all boundary vertices were found in the surface mesh")
        orig_idx = min_idx
        surf_mesh.vertices = torch.where(
            mask[:, None], line_mesh.vertices[orig_idx], surf_mesh.vertices
        )
        gus.io.meshio.export("surfs_straight_after_replacement.inp", surf_mesh.to_gus())
        return surf_mesh, dVerts_dParams
    else:
        raise RuntimeError(
            f"mesh_type {mesh_type} unavailable. Must be either line or surface"
        )


def _verts_from_params(
    parameters: _torch.Tensor,
    buffers: _torch.Tensor,
    sdf: SDFBase,
    samples: _torch.Tensor,
    constructor: FlexiCubes | FlexiSquares,
    cube_idx: _torch.Tensor,
    N,
    return_faces=False,
    output_tetmesh=False,
):

    sdf_values = functional_call(sdf, (parameters, buffers), (samples,))

    verts_local, faces, _ = constructor(
        voxelgrid_vertices=samples,
        scalar_field=sdf_values.reshape(-1),
        cube_idx=cube_idx,
        resolution=tuple(N),
        output_tetmesh=output_tetmesh,
    )

    if sdf.deformation_spline is not None:
        verts_local = sdf.deformation_spline.forward(verts_local)
    if return_faces:
        return verts_local, faces
    else:
        return verts_local


def get_verts(
    sdf: SDFBase,
    samples: _torch.Tensor,
    constructor: FlexiCubes,
    cube_idx: _torch.Tensor,
    N,
    return_faces=False,
    output_tetmesh=False,
):
    sdf_values = sdf(samples)

    verts_local, faces, _ = constructor(
        voxelgrid_vertices=samples,
        scalar_field=sdf_values.reshape(-1),
        cube_idx=cube_idx,
        resolution=tuple(N),
        output_tetmesh=output_tetmesh,
    )

    if sdf.deformation_spline is not None:
        verts_local = sdf.deformation_spline.forward(verts_local)
    if return_faces:
        return verts_local, faces
    else:
        return verts_local


def export_sdf_grid_vtk(
    sdf: SDFBase, filename, N=64, bounds=None, device="cpu", dtype=torch.float32
):
    if bounds is None:
        bounds = sdf._get_domain_bounds()
    if bounds is None:
        bounds = np.array([[0, 0, 0], [1, 1, 1]])
    if isinstance(bounds, torch.Tensor):
        bounds = bounds.detach().cpu().numpy()
    # Generate grid points
    x = np.linspace(bounds[0, 0], bounds[1, 0], N)
    y = np.linspace(bounds[0, 1], bounds[1, 1], N)
    z = np.linspace(bounds[0, 2], bounds[1, 2], N)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Evaluate SDF
    with _torch.no_grad():
        if sdf.geometric_dim == 2:
            input_points = points[:, :2]
        elif sdf.geometric_dim == 3:
            input_points = points
        sdf_vals = sdf(_torch.tensor(input_points, device=device, dtype=dtype))
    sdf_vals = sdf_vals.detach().cpu().numpy().reshape(-1)

    # Create vtkPoints
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt.tolist())

    # Create structured grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(N, N, N)
    grid.SetPoints(vtk_points)

    # Add SDF scalar field
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName("SDF")
    vtk_array.SetNumberOfValues(len(sdf_vals))
    for i, val in enumerate(sdf_vals):
        vtk_array.SetValue(i, val)
    grid.GetPointData().SetScalars(vtk_array)

    # Write to file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
    logger.info(f"SDF structured grid saved to {filename}")


def _export_surface_mesh_vtk(verts, faces, filename, dSurf: dict = None):
    """
    verts: (N, 3) torch tensor
    faces: (M, 3) torch tensor
    dSurf: (N, 3, ..,..) torch tensor
    """
    vtk_points = vtk.vtkPoints()
    for v in verts:
        vert = v.tolist()
        if len(vert) == 2:
            vert.append(0.0)
        vtk_points.InsertNextPoint(vert)

    vtk_cells = vtk.vtkCellArray()
    for f in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, f[0])
        triangle.GetPointIds().SetId(1, f[1])
        triangle.GetPointIds().SetId(2, f[2])
        vtk_cells.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    if dSurf is not None:
        for param_name, dSurf_dparam in dSurf.items():
            # Add derivative vectors as a vector field
            extra_dims = dSurf_dparam.shape[2:]
            N = int(np.prod(extra_dims))

            # reshape to [n_nodes, N_derivatives, 3]
            dSurf_flat = dSurf_dparam.flatten(2)
            indices = [np.unravel_index(i, extra_dims) for i in range(N)]
            assert dSurf_flat.shape[2] == len(indices)
            for j, idx in enumerate(indices):
                idx_str = "".join(str(i + 1) for i in idx)
                name = f"derivative_{param_name}_[{idx_str}]"

                vectors = vtk.vtkDoubleArray()
                vectors.SetNumberOfComponents(3)
                vectors.SetName(name)

                for vec in dSurf_flat[:, :, j]:
                    vecND = vec.tolist()
                    if len(vecND) == 2:
                        vecND.append(0.0)
                    vectors.InsertNextTuple(vecND)

                polydata.GetPointData().AddArray(vectors)
                polydata.GetPointData().SetActiveVectors(name)
    # Write to file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    print(f"Mesh saved to {filename}")


def export_surface_mesh(
    filename: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    mesh: gus.Faces | torchSurfMesh,
    dSurf=None,
):
    export_filename = pathlib.Path(filename)
    ext = export_filename.suffix.lower()
    if isinstance(mesh, torchSurfMesh):
        mesh = mesh.to_gus()
    match ext:
        case ".vtk":
            _export_surface_mesh_vtk(mesh.vertices, mesh.faces, export_filename, dSurf)
        case _:
            gus.io.meshio.export(export_filename, mesh)


def mergeMeshs(mesh1, mesh2, tol=1e-10):
    vertices1 = mesh1.vertices
    vertices2 = mesh2.vertices

    diff = vertices2.unsqueeze(1) - vertices1.unsqueeze(0)
    dist2 = (diff**2).sum(dim=-1)
    min_dist, nearest_idx = dist2.min(dim=1)

    duplicates = min_dist < tol

    mapping = torch.empty(vertices2.shape[0], dtype=torch.long)
    mapping[duplicates] = nearest_idx[duplicates]

    unique_vertices_mask = ~duplicates
    mapping[unique_vertices_mask] = torch.arange(
        vertices1.shape[0], vertices1.shape[0] + unique_vertices_mask.sum()
    )

    merged_vertices = torch.cat([vertices1, vertices2[unique_vertices_mask]], dim=0)

    if isinstance(mesh1, torchLineMesh):
        merged_lines = mapping[mesh2.lines]
        merged_lines = torch.cat([mesh1.lines, merged_lines], dim=0)
        return torchLineMesh(merged_vertices, merged_lines)

    elif isinstance(mesh1, torchSurfMesh):
        merged_faces = mapping[mesh2.faces]
        merged_faces = torch.cat([mesh1.faces, merged_faces], dim=0)
        return torchSurfMesh(merged_vertices, merged_faces)

    elif isinstance(mesh1, torchVolumeMesh):
        merged_volumes = mapping[mesh2.volumes]
        merged_volumes = torch.cat([mesh1.volumes, merged_volumes], dim=0)
        return torchVolumeMesh(merged_vertices, merged_volumes)

    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh1)}")

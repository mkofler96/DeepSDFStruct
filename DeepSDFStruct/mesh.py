import logging
import torch as _torch
import tetgenpy
import numpy as np
import napf
import gustaf as gus
import pathlib
import os
import skimage
import triangle
from DeepSDFStruct.SDF import SDFBase

logger = logging.getLogger(__name__)


class torchSurfMesh:
    def __init__(self, vertices: _torch.tensor, faces: _torch.tensor):
        self.vertices = vertices
        self.faces = faces

    def to_gus(self):
        return gus.Faces(self.vertices.detach().cpu(), self.faces.detach().cpu())

    def to_trimesh():
        raise NotImplementedError("To trimesh functionality not implemented yet.")
        pass


def tetrahedralize_surface(surface_mesh: gus.Faces):
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

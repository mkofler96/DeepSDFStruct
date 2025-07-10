import logging
import torch as _torch
import tetgenpy
import numpy as _np
import napf
import gustaf as gus
import pathlib
import os

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

    tets = _np.vstack(t_out.tetrahedra())
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

import trimesh
from DeepSDFStruct.mesh import torchSurfMesh
from DeepSDFStruct.SDF import SDFBase


def mesh_to_analytical(gt_sdf: SDFBase, gen_mesh: torchSurfMesh) -> float:
    """
    Calculates the SDF for every vertex on the mesh
    """

    return gt_sdf.forward(gen_mesh.vertices).abs().mean().item()

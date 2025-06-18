import trimesh


def mesh_to_analytical(gt_sdf, gen_mesh):
    """
    Calculates the SDF for every vertex on the mesh
    """
    mesh = trimesh.Trimesh(gen_mesh)

    return gt_sdf(mesh.vertices)

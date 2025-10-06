from DeepSDFStruct.mesh import torchVolumeMesh
import gustaf as gus
import torch


def test_mesh_clean() -> None:
    gus_mesh = gus.io.meshio.load("tests/data/example_disconnectd_mesh.inp")
    verts_tensor = torch.tensor(gus_mesh.vertices, requires_grad=True)
    volumes_tensor = torch.tensor(gus_mesh.volumes)
    torch_vol_mesh = torchVolumeMesh(verts_tensor, volumes_tensor)
    torch_vol_mesh.remove_disconnected_regions()
    vols = torch_vol_mesh.to_gus()
    assert (
        vols.vertices.shape[0] == 194
    ), f"number of clean vertices should be 194, but is {vols.vertices.shape[0]}"
    gus.io.meshio.export("tests/tmp_outputs/cleaned_mesh.inp", vols)


if __name__ == "__main__":
    test_mesh_clean()

from DeepSDFStruct.mesh import torchVolumeMesh, mergeMeshs, torchLineMesh
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


def test_mesh_merge():
    # Simple line mesh 1: two vertices, one line
    vertices1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lines1 = torch.tensor([[0, 1]])
    mesh1 = torchLineMesh(vertices1, lines1)

    # Simple line mesh 2: one duplicate vertex and one new vertex
    vertices2 = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    lines2 = torch.tensor([[0, 1]])
    mesh2 = torchLineMesh(vertices2, lines2)

    merged_mesh = mergeMeshs(mesh1, mesh2, tol=1e-8)
    expected_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )

    assert torch.allclose(merged_mesh.vertices, expected_vertices), "Vertices mismatch"
    # Check merged connectivity
    expected_lines = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(
        merged_mesh.lines, expected_lines
    ), f"Line connectivity mismatch. Expected\n {expected_lines}\n got \n{merged_mesh.lines}"
    # Check no indices exceed number of vertices
    assert (
        merged_mesh.lines.max().item() < merged_mesh.vertices.shape[0]
    ), "Index out of range"


if __name__ == "__main__":
    test_mesh_merge()
    test_mesh_clean()

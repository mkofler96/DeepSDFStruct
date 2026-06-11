"""Regression tests for tetrahedron orientation in FlexiCubes output.

FlexiCubes' ``_tetrahedralize`` builds tets from two sub-procedures (surface
pyramids and interior edges) whose vertex orderings do not share a consistent
winding, so a large fraction of the elements used to come out inverted
(negative signed volume). That breaks FEA solvers, which require a positive
signed volume / Jacobian on every element. The interior sub-procedure can
additionally emit degenerate (exactly coplanar, zero-volume) tets, which are
removed during extraction. These tests pin both fixes: every tet returned by
FlexiCubes must have a strictly positive signed volume.
"""

import torch

from DeepSDFStruct.flexicubes.flexicubes import FlexiCubes
from DeepSDFStruct.optimization import tet_signed_vol


def _sphere_volume_mesh(res=16, radius=0.7, center=(0.0, 0.0, 0.0)):
    fc = FlexiCubes(device="cpu")
    x_nx3, cube_fx8 = fc.construct_voxel_grid(res)
    # Spread the unit grid over [-1, 1] so the sphere sits inside the domain.
    x_nx3 = x_nx3 * 2.0
    c = torch.tensor(center, dtype=x_nx3.dtype)
    s_n = torch.linalg.norm(x_nx3 - c, dim=1) - radius
    verts, tets, _ = fc(x_nx3, s_n, cube_fx8, res, output_tetmesh=True)
    return verts, tets


def test_orient_tets_flips_inverted_element():
    """A single inverted tet is flipped to positive signed volume."""
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # [0, 1, 2, 3] is positively oriented; [0, 1, 3, 2] is its inversion.
    tets = torch.tensor([[0, 1, 2, 3], [0, 1, 3, 2]])

    assert (
        tet_signed_vol(verts, tets) < 0
    ).any(), "fixture should contain an inversion"

    oriented = FlexiCubes._orient_tets(verts, tets)
    vols = tet_signed_vol(verts, oriented)
    assert (vols > 0).all(), f"expected all positive volumes, got {vols.tolist()}"
    # Orientation is fixed by reordering indices only: |volume| is preserved.
    assert torch.allclose(vols.abs(), tet_signed_vol(verts, tets).abs())


def test_orient_tets_handles_empty():
    verts = torch.zeros((0, 3))
    tets = torch.zeros((0, 4), dtype=torch.long)
    assert FlexiCubes._orient_tets(verts, tets).shape == (0, 4)


def test_orient_tets_removes_degenerate_elements():
    """Zero-volume (coplanar) tets are dropped."""
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],  # coplanar with the first three (z = 0 plane)
        ]
    )
    tets = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])

    oriented = FlexiCubes._orient_tets(verts, tets)

    assert oriented.shape == (1, 4)
    assert (tet_signed_vol(verts, oriented) > 0).all()


def test_flexicubes_volume_mesh_has_only_positive_tets():
    """End-to-end: extracted volume mesh has strictly positive signed volumes."""
    for center in [(0.0, 0.0, 0.0), (0.13, 0.07, 0.21)]:
        verts, tets = _sphere_volume_mesh(center=center)
        assert tets.shape[0] > 0
        vols = tet_signed_vol(verts, tets)
        n_bad = int((vols <= 0).sum())
        assert n_bad == 0, f"{n_bad} non-positive tets for center {center}"


if __name__ == "__main__":
    test_orient_tets_flips_inverted_element()
    test_orient_tets_handles_empty()
    test_flexicubes_volume_mesh_has_only_positive_tets()
    print("ok")

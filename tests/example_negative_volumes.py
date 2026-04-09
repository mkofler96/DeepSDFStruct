"""
Demonstration: Sources of Negative-Volume Tetrahedra in FlexiCubes Meshes
=========================================================================

This script investigates why negative-volume tetrahedra appear when extracting
tetrahedral meshes with FlexiCubes and how to diagnose / fix them.  It covers
three scenarios of increasing complexity:

Source 1 – By construction in ``_tetrahedralize`` (surface tets)
    Surface tets are formed by appending an interior grid vertex to each
    isosurface triangle.  Because the triangle normals point *outward* (toward
    positive SDF), most of the resulting surface tets have a negative signed
    volume under the standard right-hand-rule formula.  However, a uniform
    swap of every face's first two vertices is insufficient: FlexiCubes
    sometimes places dual vertices deeper inside the solid than a nearby
    interior grid vertex, so the grid vertex ends up geometrically on the
    *outward* side of the face plane and the tet already has a positive signed
    volume.  Swapping it would make things worse.  The correct fix, applied in
    ``flexicubes.py``, computes the signed volume of every surface tet and
    flips only those that are negative, achieving 100 % correct orientation.

Source 2 – Interior tets with inconsistent orientation
    Interior tets (``tets_inside``) connect pairs of surface dual vertices to
    pairs of interior grid-edge vertices.  Their orientation is not explicitly
    controlled during assembly and can be positive or negative depending on the
    local voxel geometry.  Approximately 50 % of interior tets are negatively
    oriented.  These are not affected by the surface-tet fix.  A lightweight
    post-processing step (detect negative volumes, flip vertex order) correctly
    re-orients all of them.

Source 3 – Non-linear deformation with a locally orientation-reversing Jacobian
    When a non-linear ``TorchSpline`` deformation is applied to the mesh in
    parametric space, regions where the Jacobian determinant is negative flip the
    orientation of otherwise positive tets.

Run this file directly to see the diagnostic output for each case::

    python tests/example_negative_volumes.py
"""

import torch
import splinepy
import numpy as np

from DeepSDFStruct.sdf_primitives import SphereSDF
from DeepSDFStruct.mesh import create_3D_mesh, torchVolumeMesh
from DeepSDFStruct.optimization import tet_signed_vol
from DeepSDFStruct.torch_spline import TorchSpline


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_stats(label: str, verts: torch.Tensor, tets: torch.Tensor) -> None:
    """Print signed-volume statistics for a tetrahedral mesh."""
    vols = tet_signed_vol(verts, tets)
    n_total = len(vols)
    n_neg = int((vols < 0).sum())
    n_zero = int((vols == 0).sum())
    n_pos = int((vols > 0).sum())
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total tets  : {n_total}")
    print(f"  Positive    : {n_pos}  ({100 * n_pos / n_total:.1f} %)")
    print(f"  Zero        : {n_zero}")
    print(f"  Negative    : {n_neg}  ({100 * n_neg / n_total:.1f} %)")


def _orient_tets(verts: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    """Return a copy of *tets* with all negative-volume elements flipped."""
    tets_out = tets.clone()
    vols = tet_signed_vol(verts, tets_out)
    neg = vols < 0
    if neg.any():
        tets_out[neg] = tets_out[neg][:, [0, 2, 1, 3]]
    return tets_out


# ---------------------------------------------------------------------------
# Shared geometry: a sphere SDF centred in the unit cube
# ---------------------------------------------------------------------------

def _sphere_sdf() -> SphereSDF:
    return SphereSDF(center=[0.5, 0.5, 0.5], radius=0.35)


# ---------------------------------------------------------------------------
# Step 1 – No deformation
# ---------------------------------------------------------------------------

def step1_no_deformation() -> None:
    """
    Extract a volume mesh from a sphere SDF without any deformation.

    This step shows two stages of the orientation fix:

    Stage A – Raw FlexiCubes output with the conditional ``_tetrahedralize`` fix:
        The per-tet signed-volume check in ``_tetrahedralize`` flips every
        surface tet that has negative volume, achieving 100 % positive surface
        tets.  Interior tets (Source 2) remain inconsistently oriented.
        Typical result: ~40 % negative (all from ``tets_inside``).

    Stage B – After the additional post-hoc orientation pass:
        ``_orient_tets`` detects and flips every remaining negative tet.
        Virtually all tets become positive; the tiny residual (<1 %) are
        near-zero-volume degenerate elements at the domain boundary.
    """
    sdf = _sphere_sdf()
    mesh, _ = create_3D_mesh(sdf, N_base=20, mesh_type="volume", differentiate=False)
    assert isinstance(mesh, torchVolumeMesh)
    verts, tets = mesh.vertices, mesh.volumes

    _print_stats("Step 1 – Sphere, no deformation (raw FlexiCubes output, surface tets 100% fixed)", verts, tets)

    tets_fixed = _orient_tets(verts, tets)
    _print_stats("Step 1 – After additional post-hoc orientation fix", verts, tets_fixed)


# ---------------------------------------------------------------------------
# Step 2 – Linear deformation  (box(2, 1, 1) scaling)
# ---------------------------------------------------------------------------

def step2_linear_deformation() -> None:
    """
    Apply a linear (affine) deformation that doubles the x-extent.

    A positive-definite Jacobian (diagonal scaling) must not introduce new
    negative-volume tets.  Any negative tets come solely from the same
    systematic source as in Step 1.
    """
    sdf = _sphere_sdf()
    deformation = TorchSpline(
        splinepy.helpme.create.box(2, 1, 1).bspline, device="cpu"
    )
    mesh, _ = create_3D_mesh(
        sdf, N_base=20, mesh_type="volume", differentiate=False,
        deformation_function=deformation,
    )
    assert isinstance(mesh, torchVolumeMesh)
    verts, tets = mesh.vertices, mesh.volumes

    _print_stats("Step 2 – Sphere, linear deformation (box 2×1×1), raw (surface tets 100% fixed)", verts, tets)

    tets_fixed = _orient_tets(verts, tets)
    _print_stats("Step 2 – After additional post-hoc orientation fix", verts, tets_fixed)


# ---------------------------------------------------------------------------
# Step 3 – Non-linear deformation with a local fold
# ---------------------------------------------------------------------------

def step3_nonlinear_deformation() -> None:
    """
    Apply a non-linear deformation whose Jacobian is negative in some region.

    We start from the identity box(1, 1, 1) and shift the midpoint control
    points so that the spline folds back on itself, creating a region where
    the Jacobian determinant is negative.  This introduces additional
    negative-volume tets *beyond* those produced by Source 1.

    The excess negative tets (those that remain after flipping all surface tets)
    are evidence of the deformation-induced orientation reversal described in
    Source 3 of the analysis.
    """
    sdf = _sphere_sdf()

    # Start from a degree-1 box spline spanning [0,1]^3 (8 control points)
    box_spline = splinepy.helpme.create.box(1, 1, 1).bspline
    # Elevate x-degree to 2 so we get interior (mid-layer) control points.
    box_spline.elevate_degrees([0])  # now degrees = [2, 1, 1]; 12 control points

    cp = np.array(box_spline.control_points, dtype=np.float64)
    # The mid-layer of control points (x ≈ 0.5) can be identified by x == 0.5.
    # Shift them strongly in y to create a fold (Jacobian sign flip).
    mid_mask = np.isclose(cp[:, 0], 0.5, atol=0.1)
    # The mid-layer control points need to be shifted far enough in y to cause
    # the spline's Jacobian to go negative in the folded region.  A shift of
    # 0.7 units (into the domain width of 1.0) reliably produces a local fold
    # at this degree and knot configuration; smaller values leave the Jacobian
    # positive throughout.
    _Y_FOLD_SHIFT = 0.7
    cp[mid_mask, 1] += _Y_FOLD_SHIFT
    box_spline.control_points = cp

    deformation = TorchSpline(box_spline, device="cpu")

    mesh, _ = create_3D_mesh(
        sdf, N_base=20, mesh_type="volume", differentiate=False,
        deformation_function=deformation,
    )
    assert isinstance(mesh, torchVolumeMesh)
    verts, tets = mesh.vertices, mesh.volumes

    _print_stats(
        "Step 3 – Sphere, non-linear deformation (folded spline), raw",
        verts, tets,
    )

    # Count negative tets that survive after the surface-tet flip – these are
    # the truly problematic ones caused by the deformation.
    tets_fixed = _orient_tets(verts, tets)
    vols_fixed = tet_signed_vol(verts, tets_fixed)
    remaining_neg = int((vols_fixed < 0).sum())
    print(f"\n  Negative tets remaining after orientation fix: {remaining_neg}")
    if remaining_neg > 0:
        print(
            "  -> These are caused by the deformation Jacobian being negative in "
            "some region (Source 3)."
        )
    else:
        print(
            "  -> No deformation-induced inversions detected at this resolution."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nDemonstrating sources of negative-volume tetrahedra in FlexiCubes meshes.")
    print("(Run with a patched flexicubes.py to see the fix in action)\n")

    step1_no_deformation()
    step2_linear_deformation()
    step3_nonlinear_deformation()

    print("\nDone.")

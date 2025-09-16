import numpy as np
import pytest
from DeepSDFStruct.splinepy_unitcells.chi_3D import Chi3D
from DeepSDFStruct.splinepy_unitcells.cross_lattice import CrossLattice
from DeepSDFStruct.splinepy_unitcells.double_lattice_extruded import (
    DoubleLatticeExtruded,
)
from DeepSDFStruct.splinepy_unitcells.snappy_3d import Snappy3D

# Tolerance value for checking control points
EPS = 1e-8

all_tile_classes = [Chi3D, CrossLattice, DoubleLatticeExtruded, Snappy3D]


def check_control_points(tile_patches):
    """Helper function. Check if all of tile's control points all lie within unit
    square/cube. The tolerance is defined by EPS"""
    # Go through all patches
    for tile_patch in tile_patches:
        cps = tile_patch.control_points
        valid_cp_indices = (cps >= 0.0 - EPS) & (cps <= 1.0 + EPS)
        assert np.all(valid_cp_indices), (
            "Control points of tile must lie inside the unit square/cube. "
            + f"Found points outside bounds: {cps[~(valid_cp_indices)]}"
        )


@pytest.mark.parametrize("tile_class", all_tile_classes)
def test_tile_bounds(tile_class):
    """Test if tile is still in unit cube at the bounds. Checks default and also
    non-default parameter values.

    Parameters
    ---------
    tile_class: tile class in splinepy.microstructure.tiles
        Microtile
    """
    tile_creator = tile_class()
    # Create tile with default parameters
    tile_patches, _ = tile_creator.create_tile()
    check_control_points(tile_patches)


@pytest.mark.parametrize("tile_class", all_tile_classes)
def test_tile_evaluation(tile_class):
    tile_creator = tile_class()
    n_test_points = 10

    # Evaluate tile with given parameter and closure configuration
    splines_orig, _ = tile_creator.create_tile()
    # Set evaluation points as random spots in the parametric space
    rand = np.random.default_rng(seed=0)
    for patch in splines_orig:
        eval_points = rand.random((n_test_points, patch.para_dim))
        res = patch.evaluate(eval_points)
        np.testing.assert_array_compare(np.greater_equal, res.min(), 0)
        np.testing.assert_array_compare(np.less_equal, res.max(), 1)


if __name__ == "__main__":
    for tile in all_tile_classes:
        test_tile_bounds(tile)
        test_tile_evaluation(tile)

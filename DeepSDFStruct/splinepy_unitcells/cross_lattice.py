"""
Cross Lattice Unit Cell
=======================

This module implements a 2D cross-shaped lattice unit cell using splinepy.
The lattice consists of a rectangle with two diagonal struts in the center.

The unit cell is parametrized by strut thickness and can be used as a
building block for larger lattice structures.
"""

import numpy as _np

from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase
import splinepy as sp


class CrossLattice(_TileBase):
    """
    Lattice base cell, consisting of a rectangle with two diagonals in the
    center.

    .. raw:: html

        <p><a href="../_static/DoubleLattice.html">Fullscreen</a>.</p>
        <embed type="text/html" width="100%" height="400" src="../_static/DoubleLattice.html" />

    """  # noqa: E501

    _dim = 2
    _para_dim = 2
    # tile is evaluated at each corner
    _evaluation_points = _np.array([[0.5, 0.5, 0.5]])
    _n_info_per_eval_point = 1

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        make3D=True,
        **kwargs,  # noqa ARG002
    ):
        """Create a microtile based on the parameters that describe the branch
        thicknesses.

        Thickness parameters are used to describe the inner radius of the
        outward facing branches

        Parameters
        ----------
        parameters : np.array
          first entry defines the thickness of the vertical and horizontal
          branches
          second entry defines the thickness of the diagonal branches
        parameter_sensitivities: np.ndarray
          correlates with thickness of branches and entouring wall
        contact_length : double
          required for conformity between tiles, sets the length of the center
          block on the tiles boundary

        Returns
        -------
        microtile_list : list(splines)
        """

        # set to default if nothing is given
        if parameters is None:
            self._logd("Tile request is not parametrized, setting default 0.2")
            parameters = _np.ones((1, 1)) * 0.1
        if not (_np.all(parameters > 0) and _np.all(parameters < 0.5)):
            raise ValueError(
                "Parameters must be between 0.01 and 0.5/(1+sqrt(2))=0.207"
            )

        self.check_params(parameters)

        # Check if user requests derivative splines
        if self.check_param_derivatives(parameter_sensitivities):
            n_derivatives = parameter_sensitivities.shape[2]
            derivatives = []
        else:
            n_derivatives = 0
            derivatives = None

        splines = []
        for i_derivative in range(n_derivatives + 1):
            # Constant auxiliary values
            if i_derivative == 0:
                t = parameters[0, 0]
                v_one_half = 0.5
                v_one = 1.0
                v_zero = 0.0
            else:
                cl = 0.0
                t = parameter_sensitivities[0, 0, i_derivative - 1]
                v_one_half = 0.0
                v_one = 0.0
                v_zero = 0.0

            # Init return value
            spline_list = []

            # 1
            spline_list.append(
                _Bezier(
                    degrees=[1, 1],
                    control_points=[
                        [t, 0],
                        [v_one_half, v_one_half - t],
                        [0, 0],
                        [v_one_half, v_one_half],
                    ],
                )
            )

            # 2
            spline_list.append(
                _Bezier(
                    degrees=[1, 1],
                    control_points=[
                        [0, 0],
                        [v_one_half, v_one_half],
                        [0, t],
                        [v_one_half - t, v_one_half],
                    ],
                )
            )
            hor_reflected_spline_list = []
            for spline in spline_list:
                reflected_spline = spline.copy()
                reflected_spline.control_points[:, 0] = (
                    v_one - spline.control_points[:, 0]
                )
                hor_reflected_spline_list.append(reflected_spline)

            vert_reflected_splines = []
            for spline in spline_list + hor_reflected_spline_list:
                reflected_spline = spline.copy()
                reflected_spline.control_points[:, 1] = (
                    v_one - spline.control_points[:, 1]
                )
                vert_reflected_splines.append(reflected_spline)
            spline_list = (
                spline_list + hor_reflected_spline_list + vert_reflected_splines
            )
            # Pass to output
            if i_derivative == 0:
                splines = spline_list.copy()
            else:
                derivatives.append(spline_list)
        extr_spline_list = []
        for spline in spline_list:
            if make3D:
                extr_spline = sp.helpme.create.extruded(spline, (0, 0, 1))
                temp_pts = extr_spline.control_points.copy()
                extr_spline.control_points[:, 0] = temp_pts[:, 1]
                extr_spline.control_points[:, 1] = temp_pts[:, 2]
                extr_spline.control_points[:, 2] = temp_pts[:, 0]
            else:
                extr_spline = spline
            extr_spline_list.append(extr_spline)

        # Return results
        return (extr_spline_list, None)

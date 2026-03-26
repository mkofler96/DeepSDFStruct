"""
Chi-3D Lattice Unit Cell
========================

This module implements a 3D chi-shaped lattice unit cell using splinepy.
The chi structure is characterized by intersecting diagonal struts forming
an X or chi (χ) pattern, commonly used in mechanical metamaterials.

The unit cell provides parametric control over:
- Strut thickness
- Arm position
- Rounding radius
- Angular configuration
"""

import splinepy
import numpy as _np
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


class Chi3D(_TileBase):

    _dim = 2  # Dimension der Einheitszelle
    _para_dim = 2  # Dimension der Spline Struktur?
    _evaluation_points = _np.array(
        [[0.5, 0], [0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.5, 1]]
    )
    _n_info_per_eval_point = 5

    def create_tile(
        self, parameters=None, make3D=True, parameters_sensitivitW=None, **kwargs
    ):
        """
        [Winkel, Dicke, Armposition, Armpositionsverschiebung, Rundungradius]
        """
        Edge = 0.5
        if parameters is None:
            self._logd("Tile request is not parametrized, setting default Pi/8")
            # Parameter wie folgt [Winkel, Dicke, Armposition, Armpositionsverschiebung, Rundungradius]
            parameters = _np.array(
                [
                    [_np.pi / 8, Edge * 0.1, Edge / 1.4, Edge / 3, Edge / 8],
                    [_np.pi / 8, Edge * 0.1, Edge / 1.4, Edge / 3, Edge / 8],
                    [_np.pi / 8, Edge * 0.1, Edge / 1.4, Edge / 3, Edge / 8],
                    [_np.pi / 8, Edge * 0.1, Edge / 1.4, Edge / 3, Edge / 8],
                    [_np.pi / 8, Edge * 0.1, Edge / 1.4, Edge / 3, Edge / 8],
                ]
            )
        else:
            if not (
                _np.all(parameters >= -_np.pi * 0.5)
                and _np.all(parameters <= _np.pi * 0.5)
            ):
                raise ValueError("The parameter must be in -Pi/2 and Pi/2")
            pass
        self.check_params(parameters)

        # radius = Edge/8
        # arm_pos = Edge/1.4
        # delta_d = Edge/3

        arm_pos = parameters[2, 2]
        delta_d = parameters[2, 3]
        radius = parameters[2, 4]

        beta = parameters[2, 0]

        t = parameters[2, 1]
        # t = 0.15
        tO = parameters[3, 1]
        tN = parameters[4, 1]
        tW = parameters[1, 1]
        tS = parameters[0, 1]

        lges1 = arm_pos / _np.cos(beta) - t / 2 + Edge + arm_pos * _np.tan(beta)

        delta_tS_knick = (
            (tS - t) * ((arm_pos / _np.cos(beta) - t / 2) / lges1) + t
        ) / t
        delta_tS_gerade = (
            (tS - t) * ((arm_pos / _np.cos(beta) - t / 2) * 2 / lges1) + t
        ) / t
        delta_tW_knick = (
            (tW - t) * ((arm_pos / _np.cos(beta) - t / 2) / lges1) + t
        ) / t
        delta_tW_gerade = (
            (tW - t) * ((arm_pos / _np.cos(beta) - t / 2) * 2 / lges1) + t
        ) / t

        delta_tO_knick = (
            (tO - t) * ((arm_pos / _np.cos(beta) - t / 2) / lges1) + t
        ) / t
        delta_tO_gerade = (
            (tO - t) * ((arm_pos / _np.cos(beta) - t / 2) * 2 / lges1) + t
        ) / t
        delta_tN_knick = (
            (tN - t) * ((arm_pos / _np.cos(beta) - t / 2) / lges1) + t
        ) / t
        delta_tN_gerade = (
            (tN - t) * ((arm_pos / _np.cos(beta) - t / 2) * 2 / lges1) + t
        ) / t

        spline_list = []

        #  Chi links unten
        m1 = 1
        m2 = 1
        translationx = Edge
        translationy = Edge

        # Spline 1.1 Rundung des Armes innen
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                        [
                            -arm_pos - delta_tS_knick * t / 2,
                            (arm_pos + t / 2) * _np.tan(beta)
                            + delta_tS_knick * t / (2 * _np.sin(_np.pi / 2 - beta)),
                        ],
                        [
                            -arm_pos - delta_tS_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                        [
                            -arm_pos + delta_tS_knick * t / 2,
                            (arm_pos - t / 2) * _np.tan(beta)
                            - delta_tS_knick * t / (2 * _np.sin(_np.pi / 2 - beta)),
                        ],
                        [
                            -arm_pos + delta_tS_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                    ]
                )
                * _np.array(
                    [[m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 1.2 Rundung des Armes außen
        spline_list.append(
            splinepy.Bezier(
                degrees=[3, 1],
                control_points=_np.array(
                    [
                        [
                            -arm_pos - delta_tS_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                        [
                            -arm_pos - delta_tS_gerade * t / 6 * 2 - tS / 6,
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tS
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [
                            -arm_pos - delta_tS_gerade * t / 6 - tS / 6 * 2 + delta_d,
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tS
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [-arm_pos - tS / 2 + delta_d, -Edge],
                        [
                            -arm_pos + delta_tS_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                        [
                            -arm_pos + delta_tS_gerade * t / 6 * 2 + tS / 6,
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tS
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [
                            -arm_pos + delta_tS_gerade * t / 6 + tS / 6 * 2 + delta_d,
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tS
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [-arm_pos + tS / 2 + delta_d, -Edge],
                    ]
                )
                * _np.array(
                    [
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                    ]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 1.3 innerer Radius
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(_np.pi / 4 + beta),
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(_np.pi / 4 + beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                        [
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(-_np.pi / 4 + beta),
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(-_np.pi / 4 + beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                    ]
                )
                * _np.array(
                    [[m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 2.1
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                        [
                            (arm_pos + t / 2) * _np.tan(beta)
                            + delta_tO_gerade * t / (2 * _np.sin(_np.pi / 2 - beta)),
                            -arm_pos - delta_tO_gerade * t / 2,
                        ],
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos - delta_tO_gerade * t / 2,
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                        [
                            (arm_pos - t / 2) * _np.tan(beta)
                            - delta_tO_gerade * t / (2 * _np.sin(_np.pi / 2 - beta)),
                            -arm_pos + delta_tO_knick * t / 2,
                        ],
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos + delta_tO_gerade * t / 2,
                        ],
                    ]
                )
                * _np.array(
                    [[-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 2.2 Rundung des Armes außen
        spline_list.append(
            splinepy.Bezier(
                degrees=[3, 1],
                control_points=_np.array(
                    [
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos - delta_tO_gerade * t / 2,
                        ],
                        [
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tO
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos - delta_tO_gerade * t / 6 * 2 - tO / 6,
                        ],
                        [
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tO
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos - delta_tO_gerade * t / 6 - tO / 6 * 2 + delta_d,
                        ],
                        [-Edge, -arm_pos - tO / 2 + delta_d],
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos + delta_tO_gerade * t / 2,
                        ],
                        [
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tO
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos + delta_tO_gerade * t / 6 * 2 + tO / 6,
                        ],
                        [
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tO
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos + delta_tO_gerade * t / 6 + tO / 6 * 2 + delta_d,
                        ],
                        [-Edge, -arm_pos + tO / 2 + delta_d],
                    ]
                )
                * _np.array(
                    [
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                    ]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 2.3
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(_np.pi / 4 + beta),
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(_np.pi / 4 + beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                        [
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(-_np.pi / 4 + beta),
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(-_np.pi / 4 + beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                    ]
                )
                * _np.array(
                    [[-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 3.1
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                        [
                            -arm_pos - delta_tN_knick * t / 2,
                            (arm_pos + t / 2) * _np.tan(beta)
                            + delta_tN_knick * t / (2 * _np.sin(_np.pi / 2 - beta)),
                        ],
                        [
                            -arm_pos - delta_tN_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                        [
                            -arm_pos + delta_tN_knick * t / 2,
                            (arm_pos - t / 2) * _np.tan(beta)
                            - delta_tN_knick * t / (2 * _np.sin(_np.pi / 2 - beta)),
                        ],
                        [
                            -arm_pos + delta_tN_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                    ]
                )
                * -1
                * _np.array(
                    [[m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 3.2 Rundung des Armes außen
        spline_list.append(
            splinepy.Bezier(
                degrees=[3, 1],
                control_points=_np.array(
                    [
                        [
                            -arm_pos - delta_tN_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                        [
                            -arm_pos - delta_tN_gerade * t / 6 * 2 - tN / 6,
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tN
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [
                            -arm_pos - delta_tN_gerade * t / 6 - tN / 6 * 2 + delta_d,
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tN
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [-arm_pos - tN / 2 + delta_d, -Edge],
                        [
                            -arm_pos + delta_tN_gerade * t / 2,
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                        ],
                        [
                            -arm_pos + delta_tN_gerade * t / 6 * 2 + tN / 6,
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tN
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [
                            -arm_pos + delta_tN_gerade * t / 6 + tN / 6 * 2 + delta_d,
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tN
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                        ],
                        [-arm_pos + tN / 2 + delta_d, -Edge],
                    ]
                )
                * -1
                * _np.array(
                    [
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                        [m1, m2],
                    ]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 3.3
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(_np.pi / 4 + beta),
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(_np.pi / 4 + beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                        [
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(-_np.pi / 4 + beta),
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(-_np.pi / 4 + beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                        ],
                        [
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                        ],
                    ]
                )
                * -1
                * _np.array(
                    [[m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2], [m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 4.1
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                        [
                            (arm_pos + t / 2) * _np.tan(beta)
                            + delta_tW_knick * t / (2 * _np.sin(_np.pi / 2 - beta)),
                            -arm_pos - delta_tW_knick * t / 2,
                        ],
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos - delta_tW_gerade * t / 2,
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                        [
                            (arm_pos - t / 2) * _np.tan(beta)
                            - delta_tW_knick * t / (2 * _np.sin(_np.pi / 2 - beta)),
                            -arm_pos + delta_tW_knick * t / 2,
                        ],
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos + delta_tW_gerade * t / 2,
                        ],
                    ]
                )
                * -1
                * _np.array(
                    [[-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 4.2 Rundung des Armes außen
        spline_list.append(
            splinepy.Bezier(
                degrees=[3, 1],
                control_points=_np.array(
                    [
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos - delta_tW_gerade * t / 2,
                        ],
                        [
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tW
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos - delta_tW_gerade * t / 6 * 2 - tW / 6,
                        ],
                        [
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            - (
                                0.5
                                * tW
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos - delta_tW_gerade * t / 6 - tW / 6 * 2 + delta_d,
                        ],
                        [-Edge, -arm_pos - tW / 2 + delta_d],
                        [
                            (arm_pos * _np.tan(beta) - arm_pos / _np.cos(beta) + t / 2)
                            + radius,
                            -arm_pos + delta_tW_gerade * t / 2,
                        ],
                        [
                            -(Edge / 3)
                            + 2
                            * (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tW
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos + delta_tW_gerade * t / 6 * 2 + tW / 6,
                        ],
                        [
                            -(2 * Edge / 3)
                            + (
                                (
                                    arm_pos * _np.tan(beta)
                                    - arm_pos / _np.cos(beta)
                                    + t / 2
                                )
                                + radius
                            )
                            / 3
                            + (
                                0.5
                                * tW
                                * (
                                    -1
                                    + 1
                                    / (
                                        _np.cos(
                                            _np.arctan(
                                                (3 * delta_d)
                                                / (
                                                    Edge
                                                    + (
                                                        arm_pos * _np.tan(beta)
                                                        - arm_pos / _np.cos(beta)
                                                        + t / 2
                                                    )
                                                    + radius
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (
                                (3 * delta_d)
                                / (
                                    Edge
                                    + (
                                        arm_pos * _np.tan(beta)
                                        - arm_pos / _np.cos(beta)
                                        + t / 2
                                    )
                                    + radius
                                )
                            ),
                            -arm_pos + delta_tW_gerade * t / 6 + tW / 6 * 2 + delta_d,
                        ],
                        [-Edge, -arm_pos + tW / 2 + delta_d],
                    ]
                )
                * -1
                * _np.array(
                    [
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                        [-m1, m2],
                    ]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 4.3
        spline_list.append(
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(_np.pi / 4 + beta),
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(_np.pi / 4 + beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                        [
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(-_np.pi / 4 + beta),
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(-_np.pi / 4 + beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + radius * (2 - _np.sqrt(2)) * _np.sin(beta),
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - radius * (2 - _np.sqrt(2)) * _np.cos(beta),
                        ],
                        [
                            t / _np.sqrt(2) * _np.sin(-_np.pi / 4 + beta)
                            + (radius * _np.sin(beta)),
                            -t / _np.sqrt(2) * _np.cos(-_np.pi / 4 + beta)
                            - (radius * _np.cos(beta)),
                        ],
                    ]
                )
                * -1
                * _np.array(
                    [[-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2], [-m1, m2]]
                )
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        # Spline 5 Middle
        spline_list.append(
            splinepy.Bezier(
                degrees=[1, 1],
                control_points=_np.array(
                    [
                        [
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(_np.pi / 4 + beta),
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(_np.pi / 4 + beta),
                        ],
                        [
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(-_np.pi / 4 + beta),
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(-_np.pi / 4 + beta),
                        ],
                        [
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(-_np.pi / 4 + beta),
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(-_np.pi / 4 + beta),
                        ],
                        [
                            (t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.cos(_np.pi / 4 + beta),
                            -(t / _np.sqrt(2) + radius * (_np.sqrt(2) - 1))
                            * _np.sin(_np.pi / 4 + beta),
                        ],
                    ]
                )
                * _np.array([[m1, m2], [m1, m2], [m1, m2], [m1, m2]])
                + _np.array(
                    [
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                        [translationx, translationy],
                    ]
                ),
            )
        )

        extr_spline_list = []
        for spline in spline_list:
            if make3D:
                extr_spline = splinepy.helpme.create.extruded(spline, (0, 0, 1))
                temp_pts = extr_spline.control_points.copy()
                extr_spline.control_points[:, 0] = temp_pts[:, 1]
                extr_spline.control_points[:, 1] = temp_pts[:, 2]
                extr_spline.control_points[:, 2] = temp_pts[:, 0]
            else:
                extr_spline = spline
            extr_spline_list.append(extr_spline)

        return (extr_spline_list, None)

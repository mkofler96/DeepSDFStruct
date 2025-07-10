import numpy as _np
import splinepy
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase

"""
Chi Tile written by Michael Giritsch

"""


class Chi3D(_TileBase):
    _dim = 2  # Dimension der Einheitszelle
    _para_dim = 2  # Dimension der Spline Struktur?
    _evaluation_points = _np.array([[0.5, 0.5], [0, 0], [1, 0], [0, 1], [1, 1]])  # ??
    _n_info_per_eval_point = 2  # ??

    def create_tile(
        self, parameters=None, make3D=True, parameters_sensitivity=None, **kwargs
    ):
        if parameters is None:
            self._logd("Tile request is not parametrized, setting default Pi/8")
            parameters = _np.array(
                [
                    [_np.pi / 8, 0.05],
                    [_np.pi / 8, 0.05],
                    [_np.pi / 8, 0.05],
                    [_np.pi / 8, 0.05],
                    [_np.pi / 8, 0.05],
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

        # beta = _np.pi/4
        beta = parameters[0, 0]
        Base = 0
        Edge = 0.5
        r = _np.sqrt(0.125)
        a = r * _np.cos(beta + _np.pi / 4)
        b = r * _np.sin(beta + _np.pi / 4)

        gamma = (
            _np.arctan(
                (Edge - r * _np.cos(beta + _np.pi / 4))
                / (Edge - r * _np.sin(beta + _np.pi / 4))
            )
            - _np.pi / 4
        )

        delta = _np.pi - gamma - beta

        t = parameters[0, 1]
        t1 = parameters[3, 1]
        t2 = parameters[4, 1]
        t3 = parameters[2, 1]
        t4 = parameters[1, 1]

        # short = t/_np.sqrt(2)*_np.sin(_np.pi/4-gamma)
        # long = t/_np.sqrt(2)*_np.sin(_np.pi/4+gamma)

        Length_outer_part = (Edge - r * _np.sin(beta + _np.pi / 4)) / _np.cos(
            gamma + _np.pi / 4
        )

        Length_ratio = _np.sqrt(0.125) / (Length_outer_part + _np.sqrt(0.125))
        t1_ratio_middle_CP = ((t1 - t) * Length_ratio + t) / t
        t2_ratio_middle_CP = ((t2 - t) * Length_ratio + t) / t
        t3_ratio_middle_CP = ((t3 - t) * Length_ratio + t) / t
        t4_ratio_middle_CP = ((t4 - t) * Length_ratio + t) / t

        spline_list = []
        # spline 1
        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 1.1 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        # [-Edge, Edge - t/(2*_np.sin(gamma+_np.pi/4))],
                        [-Edge, Edge - t1 / _np.sqrt(2)],
                        [
                            -a
                            - t1_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            b
                            - t1_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base - (t * _np.cos(beta) / _np.sqrt(2)),
                            Base + (t * _np.sin(beta) / _np.sqrt(2)),
                        ],
                        [-Edge, Edge],
                        [
                            -a
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            b
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                    ]
                )
                + Edge,
            )
        )
        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 1.2 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [-Edge, Edge],
                        [
                            -a
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            b
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                        [-Edge + t1 / _np.sqrt(2), Edge],
                        [
                            -a
                            + t1_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            b
                            + t1_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            + (t * _np.cos(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                            Base
                            + (t * _np.sin(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 2.1 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [Edge - t2 / _np.sqrt(2), Edge],
                        [
                            b
                            - t2_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            a
                            + t2_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base + (t * _np.sin(beta) / _np.sqrt(2)),
                            Base + (t * _np.cos(beta) / _np.sqrt(2)),
                        ],
                        [Edge, Edge],
                        [
                            b
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            a
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 2.2 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [Edge, Edge],
                        [
                            b
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            a
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                        [Edge, Edge - t2 / _np.sqrt(2)],
                        [
                            b
                            + t2_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            a
                            - t2_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            + (t * _np.sin(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                            Base
                            - (t * _np.cos(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 3.1 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        # [Edge, -Edge + t/(2*_np.sin(gamma+_np.pi/4))],
                        [Edge, -Edge + t3 / _np.sqrt(2)],
                        [
                            a
                            + t3_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            -b
                            + t3_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base + (t * _np.cos(beta) / _np.sqrt(2)),
                            Base - (t * _np.sin(beta) / _np.sqrt(2)),
                        ],
                        [Edge, -Edge],
                        [
                            a
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            -b
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 3.2 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [Edge, -Edge],
                        [
                            a
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            -b
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                        [Edge - t3 / _np.sqrt(2), -Edge],
                        [
                            a
                            - t3_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                            -b
                            - t3_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            - (t * _np.cos(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                            Base
                            - (t * _np.sin(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 4.1 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [-Edge + t4 / _np.sqrt(2), -Edge],
                        [
                            -b
                            + t4_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            -a
                            - t4_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base - (t * _np.sin(beta) / _np.sqrt(2)),
                            Base - (t * _np.cos(beta) / _np.sqrt(2)),
                        ],
                        [-Edge, -Edge],
                        [
                            -b
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            -a
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 4.2 spline
            splinepy.Bezier(
                degrees=[2, 1],
                control_points=_np.array(
                    [
                        [-Edge, -Edge],
                        [
                            -b
                            - ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            -a
                            + ((t / 2 / _np.sin(delta / 2)) - t / 2)
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                        [-Edge, -Edge + t4 / _np.sqrt(2)],
                        [
                            -b
                            - t4_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.sin(3 * _np.pi / 4 - delta / 2 - beta),
                            -a
                            + t4_ratio_middle_CP
                            * t
                            / (2 * _np.sin(delta / 2))
                            * _np.cos(3 * _np.pi / 4 - delta / 2 - beta),
                        ],
                        [
                            Base
                            - (t * _np.sin(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                            Base
                            + (t * _np.cos(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 5.1 middle spline
            splinepy.Bezier(
                degrees=[1, 1],
                control_points=_np.array(
                    [
                        [
                            Base
                            - (t * _np.sin(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                            Base
                            + (t * _np.cos(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                        ],
                        # #[Base - (t*_np.sin(beta)/_np.sqrt(2)) - long*_np.sin(_np.pi/4-beta), Base - (t*_np.cos(beta)/_np.sqrt(2)) + long*_np.cos(_np.pi/4-beta)],
                        [
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                        [
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                        [Base, Base],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 5.2 middle spline
            splinepy.Bezier(
                degrees=[1, 1],
                control_points=_np.array(
                    [
                        [
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                        [
                            Base - (t * _np.sin(beta) / _np.sqrt(2)),
                            Base - (t * _np.cos(beta) / _np.sqrt(2)),
                        ],
                        [Base, Base],
                        [
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 5.3middle spline
            splinepy.Bezier(
                degrees=[1, 1],
                control_points=_np.array(
                    [
                        [
                            Base
                            - (t * _np.cos(beta) / _np.sqrt(2))
                            + t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                        [Base, Base],
                        [
                            Base + (t * _np.sin(beta) / _np.sqrt(2)),
                            Base + (t * _np.cos(beta) / _np.sqrt(2)),
                        ],
                        [
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                    ]
                )
                + Edge,
            )
        )

        spline_list.append(
            # wäre auch mit splinepy.bezier möglich und erspart dadurch die Liste knot_vectors
            # 5.4 middle spline
            splinepy.Bezier(
                degrees=[1, 1],
                control_points=_np.array(
                    [
                        [Base, Base],
                        [
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                            Base
                            - (t * _np.sin(beta) / _np.sqrt(2))
                            - t / 2 * _np.sin(_np.pi / 4 - beta),
                        ],
                        [
                            Base
                            + (t * _np.sin(beta) / _np.sqrt(2))
                            + t / 2 * _np.sin(_np.pi / 4 - beta),
                            Base
                            + (t * _np.cos(beta) / _np.sqrt(2))
                            - t / 2 * _np.cos(_np.pi / 4 - beta),
                        ],
                        [
                            Base
                            + (t * _np.sin(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                            Base
                            - (t * _np.cos(_np.pi - beta - _np.pi / 2) / _np.sqrt(2)),
                        ],
                    ]
                )
                + Edge,
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

        return splinepy.Multipatch(extr_spline_list)

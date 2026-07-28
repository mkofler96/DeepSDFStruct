"""
Utility Functions
=================

This module provides general utility functions used throughout DeepSDFStruct,
including logging configuration and color scheme definitions.

Functions
---------
configure_logging
    Set up logging for the DeepSDFStruct package with customizable
    output format and destinations.
with_float32_lattice
    Run a callable with a lattice structure temporarily cast to float32.

Constants
---------
_TUWIEN_COLOR_SCHEME
    TU Wien corporate color scheme for consistent visualization styling.
"""

import logging

import torch

import DeepSDFStruct


def configure_logging(level=logging.INFO, logfile=None):
    """Configure logging for the DeepSDFStruct package.

    Sets up a logger with a standard format and optional file output.
    This is called automatically when DeepSDFStruct is imported.

    Parameters
    ----------
    level : int, default logging.INFO
        Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
    logfile : str, optional
        Path to log file. If provided, logs are written to both console
        and file. If None, logs only to console.

    Examples
    --------
    >>> from DeepSDFStruct.utils import configure_logging
    >>> import logging
    >>>
    >>> # Set debug level and log to file
    >>> configure_logging(level=logging.DEBUG, logfile='deepsdf.log')

    Notes
    -----
    The log format is: "HH:MM:SS message"
    All log messages are prefixed with a timestamp for easy debugging.
    """
    logger = logging.getLogger(DeepSDFStruct.__name__)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if not logger.hasHandlers():
        logger_handler = logging.StreamHandler()
        logger_handler.setFormatter(formatter)
        logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def with_float32_lattice(lattice_struct, bounds, fn):
    """Run ``fn(bounds_f32)`` with *lattice_struct* temporarily cast to float32.

    The DeepSDF decoder and FlexiCubes mesh extraction only run in float32,
    while a downstream shape optimizer may hold the lattice parameters in
    float64. This helper performs the cast locally around ``fn`` and restores
    the original dtypes afterwards, so callers never have to leave the lattice
    in a downgraded state.

    Parameters
    ----------
    lattice_struct : LatticeSDFStruct
        Structure whose parametrization parameters and ``bounds`` buffer are
        cast for the duration of the call.
    bounds : torch.Tensor
        Bounds tensor handed to ``fn`` as float32. Not modified in place.
    fn : Callable[[torch.Tensor], Any]
        Called once, with the float32 ``bounds``. Its return value is passed
        through.

    Returns
    -------
    Any
        Whatever ``fn`` returned.

    Examples
    --------
    >>> from DeepSDFStruct.utils import with_float32_lattice
    >>> from DeepSDFStruct.mesh import create_3D_mesh
    >>> mesh = with_float32_lattice(  # doctest: +SKIP
    ...     struct,
    ...     struct.bounds,
    ...     lambda b: create_3D_mesh(struct, 32, bounds=b, mesh_type="surface"),
    ... )
    """
    params = list(lattice_struct.parametrization.parameters())
    saved_params = [p.data for p in params]
    for p in params:
        p.data = p.data.float()

    saved_bounds = lattice_struct.bounds.data
    lattice_struct.bounds.data = lattice_struct.bounds.data.float()

    saved_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        return fn(bounds.float())
    finally:
        torch.set_default_dtype(saved_default_dtype)
        for p, s in zip(params, saved_params):
            p.data = s
        lattice_struct.bounds.data = saved_bounds


#: TU Wien corporate color scheme
#:
#: Dictionary mapping color names to RGB tuples (0-255 range).
#: Includes primary colors (blue, black, white) and secondary colors
#: (green, magenta, yellow, grey) with multiple shades of each.
#:
#: Useful for creating plots and visualizations with consistent branding.
_TUWIEN_COLOR_SCHEME = {
    "blue": (0, 102, 153),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "blue_1": (84, 133, 171),
    "blue_2": (114, 173, 213),
    "blue_3": (166, 213, 236),
    "blue_4": (223, 242, 253),
    "grey": (100, 99, 99),
    "grey_1": (157, 157, 156),
    "grey_2": (208, 208, 208),
    "grey_3": (237, 237, 237),
    "green": (0, 126, 113),
    "green_1": (106, 170, 165),
    "green_2": (162, 198, 194),
    "green_3": (233, 241, 240),
    "magenta": (186, 70, 130),
    "magenta_1": (205, 129, 168),
    "magenta_2": (223, 175, 202),
    "magenta_3": (245, 229, 239),
    "yellow": (225, 137, 34),
    "yellow_1": (238, 180, 115),
    "yellow_2": (245, 208, 168),
    "yellow_3": (153, 239, 225),
}

"""Tests for ``DeepSDFStruct.utils``."""

import logging

import pytest
import torch

import DeepSDFStruct
from DeepSDFStruct.utils import (
    _TUWIEN_COLOR_SCHEME,
    configure_logging,
    with_float32_lattice,
)


@pytest.fixture
def isolated_logger():
    """Hand out the package logger with its global state restored afterwards.

    ``configure_logging`` runs on import, and under pytest the root logger also
    carries handlers, so the handler-creation branch is only reachable from a
    logger that has been emptied and detached from its parent.
    """
    logger = logging.getLogger(DeepSDFStruct.__name__)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    logger.handlers = []
    logger.propagate = False
    yield logger

    for handler in logger.handlers:
        if handler not in saved_handlers:
            handler.close()
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


def test_configure_logging_adds_a_stream_handler(isolated_logger):
    configure_logging(level=logging.DEBUG)

    assert isolated_logger.level == logging.DEBUG
    stream_handlers = [
        h for h in isolated_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1
    assert stream_handlers[0].formatter is not None


def test_configure_logging_does_not_duplicate_stream_handlers(isolated_logger):
    configure_logging()
    configure_logging()
    configure_logging()

    # The hasHandlers() guard must keep repeat calls from stacking handlers,
    # which would print every record several times.
    assert len(isolated_logger.handlers) == 1


def test_configure_logging_sets_the_requested_level(isolated_logger):
    configure_logging(level=logging.WARNING)
    assert isolated_logger.level == logging.WARNING
    configure_logging(level=logging.INFO)
    assert isolated_logger.level == logging.INFO


def test_configure_logging_writes_to_logfile(isolated_logger, tmp_path):
    logfile = tmp_path / "deepsdf.log"
    configure_logging(level=logging.INFO, logfile=str(logfile))

    file_handlers = [
        h for h in isolated_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1

    isolated_logger.info("hello from the test")
    file_handlers[0].flush()

    contents = logfile.read_text(encoding="utf-8")
    assert "hello from the test" in contents
    # Format is "HH:MM:SS message".
    assert contents.split(" ")[0].count(":") == 2


def test_configure_logging_logfile_is_additive(isolated_logger, tmp_path):
    """A logfile must be added alongside the console handler, not replace it."""
    configure_logging(logfile=str(tmp_path / "a.log"))

    kinds = [type(h) for h in isolated_logger.handlers]
    assert logging.FileHandler in kinds
    assert any(k is logging.StreamHandler for k in kinds)


def test_tuwien_color_scheme_is_valid_rgb():
    assert _TUWIEN_COLOR_SCHEME["white"] == (255, 255, 255)
    assert _TUWIEN_COLOR_SCHEME["black"] == (0, 0, 0)

    for name, rgb in _TUWIEN_COLOR_SCHEME.items():
        assert len(rgb) == 3, name
        for channel in rgb:
            assert isinstance(channel, int), name
            assert 0 <= channel <= 255, name


class _FakeParametrization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cps = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))


class _FakeLattice:
    """Minimal stand-in exposing the attributes ``with_float32_lattice`` touches."""

    def __init__(self):
        self.parametrization = _FakeParametrization()
        self.bounds = torch.nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        )


def test_with_float32_lattice_casts_inside_and_restores_after():
    lattice = _FakeLattice()
    seen = {}

    def fn(bounds):
        seen["bounds_dtype"] = bounds.dtype
        seen["param_dtype"] = next(lattice.parametrization.parameters()).dtype
        seen["lattice_bounds_dtype"] = lattice.bounds.data.dtype
        seen["default_dtype"] = torch.get_default_dtype()
        return "sentinel"

    saved_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        result = with_float32_lattice(lattice, lattice.bounds, fn)
    finally:
        torch.set_default_dtype(saved_default)

    # Everything the decoder sees is float32 during the call...
    assert result == "sentinel"
    assert seen["bounds_dtype"] == torch.float32
    assert seen["param_dtype"] == torch.float32
    assert seen["lattice_bounds_dtype"] == torch.float32
    assert seen["default_dtype"] == torch.float32

    # ...and the float64 state is handed back untouched afterwards.
    assert next(lattice.parametrization.parameters()).dtype == torch.float64
    assert lattice.bounds.data.dtype == torch.float64


def test_with_float32_lattice_restores_state_when_fn_raises():
    lattice = _FakeLattice()

    def boom(_bounds):
        raise RuntimeError("mesh extraction failed")

    saved_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(RuntimeError, match="mesh extraction failed"):
            with_float32_lattice(lattice, lattice.bounds, boom)

        # The finally block must undo the cast, or the optimizer would silently
        # continue in float32 after a failed extraction.
        assert torch.get_default_dtype() == torch.float64
        assert next(lattice.parametrization.parameters()).dtype == torch.float64
        assert lattice.bounds.data.dtype == torch.float64
    finally:
        torch.set_default_dtype(saved_default)


def test_with_float32_lattice_does_not_modify_caller_bounds():
    lattice = _FakeLattice()
    bounds = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=torch.float64)

    with_float32_lattice(lattice, bounds, lambda b: None)

    assert bounds.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for ``DeepSDFStruct.export_onnx``."""

import onnx
import pytest
import torch

from DeepSDFStruct.export_onnx import export_ONNX
from DeepSDFStruct.pretrained_models import PretrainedModels, get_model


@pytest.fixture(scope="module", autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_ONNX(path)
    return path, onnx.load(path)


def test_export_onnx_produces_a_valid_model(exported_model):
    path, model = exported_model

    assert path.is_file()
    assert path.stat().st_size > 0
    # Raises if the graph is malformed.
    onnx.checker.check_model(model)


def test_export_onnx_input_matches_latent_plus_coordinates(exported_model):
    _, model = exported_model
    reference = get_model(PretrainedModels.Primitives)
    latent_dim = reference._trained_latent_vectors[0].shape[0]

    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert model.graph.input[0].name == "input"
    # The decoder is fed [latent_code, xyz].
    assert dims == [1, latent_dim + 3]

    out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert model.graph.output[0].name == "output"
    # One SDF value per input point.
    assert out_dims == [1, 1]


def test_export_onnx_writes_metadata(exported_model):
    _, model = exported_model
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    assert metadata["author"] == "Michael Kofler"
    assert metadata["trainingelementsize"] == "1"


def test_export_onnx_uses_requested_opset(exported_model):
    _, model = exported_model
    opsets = {imp.domain: imp.version for imp in model.opset_import}
    assert opsets[""] == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

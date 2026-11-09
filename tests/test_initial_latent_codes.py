import numpy as np
import pytest
import torch

from DeepSDFStruct.deep_sdf.training import initialize_latent_vectors


def embedding(num_scenes=4, latent_size=1, max_norm=1.0):
    return torch.nn.Embedding(num_scenes, latent_size, max_norm=max_norm)


def test_default_draw_matches_plain_normal_initialization():
    torch.manual_seed(3)
    expected = torch.empty(4, 2)
    torch.nn.init.normal_(expected, 0.0, 0.5 / np.sqrt(2))

    vectors = embedding(latent_size=2)
    torch.manual_seed(3)
    initialize_latent_vectors(vectors, {"CodeInitStdDev": 0.5}, "unused")

    assert torch.equal(vectors.weight.data, expected)


def test_supplied_codes_replace_random_codes(tmp_path):
    codes = np.array([[-0.9], [-0.1], [0.2], [0.7]])
    np.save(tmp_path / "codes.npy", codes)
    vectors = embedding()

    initialize_latent_vectors(vectors, {"InitialLatentCodes": "codes.npy"}, tmp_path)

    assert np.allclose(vectors.weight.detach().numpy(), codes)


def test_supplied_codes_leave_random_stream_unchanged(tmp_path):
    np.save(tmp_path / "codes.npy", np.zeros((4, 1)))

    torch.manual_seed(5)
    initialize_latent_vectors(embedding(), {}, tmp_path)
    without_option = torch.rand(1)

    torch.manual_seed(5)
    initialize_latent_vectors(
        embedding(), {"InitialLatentCodes": "codes.npy"}, tmp_path
    )
    with_option = torch.rand(1)

    assert torch.equal(without_option, with_option)


@pytest.mark.parametrize(
    "codes, message",
    [
        (np.zeros((3, 1)), "shape"),
        (np.array([[0.0], [np.nan], [0.0], [0.0]]), "nonfinite"),
        (np.array([[0.0], [1.5], [0.0], [0.0]]), "CodeBound"),
    ],
)
def test_invalid_supplied_codes_are_rejected(tmp_path, codes, message):
    np.save(tmp_path / "codes.npy", codes)

    with pytest.raises(ValueError, match=message):
        initialize_latent_vectors(
            embedding(), {"InitialLatentCodes": "codes.npy"}, tmp_path
        )

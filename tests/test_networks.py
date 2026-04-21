import torch

from DeepSDFStruct.deep_sdf.networks.hierarchical_positional_sdf_decoder import (
    HierachicalPositionalDeepSDFDecoder,
)
from DeepSDFStruct.deep_sdf.networks.hierarchical_deep_sdf_decoder import (
    HierachicalDeepSDFDecoder,
)


def test_hierarchical_positional_decoder_forward():
    batch = 4
    latent_size = 8
    geom_dim = 3
    dims = [32, 16, 8]
    # inject latent at layer 0 (before first linear)
    latent_in = [0]

    net = HierachicalPositionalDeepSDFDecoder(
        latent_size=latent_size,
        dims=dims,
        geom_dimension=geom_dim,
        latent_in=latent_in,
        positional_encoding=False,
        activation_fun="gelu",
    )
    net.eval()

    lat = torch.randn(batch, latent_size)
    xyz = torch.randn(batch, geom_dim)
    inp = torch.cat([lat, xyz], dim=1)

    with torch.no_grad():
        out = net(inp)

    assert out.shape == (batch, 1)


def test_hierarchical_deep_decoder_forward():
    batch = 5
    geom_dim = 3
    # one latent chunk of size 4 injected into layer 1
    latent_size = [4]
    latent_in = [1]
    dims = [32, 16]

    net = HierachicalDeepSDFDecoder(
        latent_size=latent_size,
        dims=dims,
        geom_dimension=geom_dim,
        latent_in=latent_in,
        activation_fun="silu",
    )
    net.eval()

    lat = torch.randn(batch, sum(latent_size))
    xyz = torch.randn(batch, geom_dim)
    inp = torch.cat([lat, xyz], dim=1)

    with torch.no_grad():
        out = net(inp)

    # hierarchical_deep_sdf_decoder returns a raw tensor (no explicit final layer),
    # ensure it returns per-sample outputs (batch, *) and is finite
    assert out.shape[0] == batch
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    test_hierarchical_deep_decoder_forward()
    test_hierarchical_positional_decoder_forward()

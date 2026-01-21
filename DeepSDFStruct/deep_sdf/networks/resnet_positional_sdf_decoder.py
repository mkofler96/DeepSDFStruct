import torch
import torch.nn as nn
import logging

logger = logging.getLogger("DeepSDFStruct")


class ResNetPositionalDeepSDFDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=None,
        norm_layers=None,
        latent_in=None,
        xyz_in_all=False,
        latent_dropout=False,
        weight_norm=True,
        use_tanh=None,
        geom_dimension=3,
        activation_fun="tanh",
        positional_encoding=False,
        pe_levels=6,  # number of frequency bands if PE enabled
        resnet=False,
        resnet_every=2,  # insert residual connection every N layers
    ):
        super().__init__()
        if use_tanh is not None:
            logger.warning("Use tanh is deprecated, use the activation_fun argument")
        self.geom_dimension = geom_dimension
        self.use_pe = positional_encoding
        self.pe_levels = pe_levels
        self.use_resnet = resnet
        self.resnet_every = resnet_every

        # Optional positional encoding increases xyz dimension
        input_geom_dim = geom_dimension
        if positional_encoding:
            input_geom_dim = geom_dimension * (2 * pe_levels + 1)

        # MLP layer dims
        dims = [latent_size + input_geom_dim] + dims + [1]
        self.num_layers = len(dims) - 1

        if activation_fun == "tanh":
            self.act = nn.Tanh()
        elif activation_fun == "relu":
            self.act = nn.ReLU()
        elif activation_fun == "leaky_relu":
            self.act = nn.LeakyReLU(0.2)
        elif activation_fun == "silu":
            self.act = nn.SiLU()
        elif activation_fun == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation_fun}")

        # Construct layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    # ----------- Positional Encoding -----------
    def positional_encoding(self, xyz):
        """
        Applies sinusoidal positional encoding to each of the xyz dims.
        Output shape: [B, 3*(2*L+1)]
        """
        enc = [xyz]
        for i in range(self.pe_levels):
            freq = 2**i
            enc.append(torch.sin(freq * xyz))
            enc.append(torch.cos(freq * xyz))
        return torch.cat(enc, dim=-1)

    # ---------------- Forward ------------------
    def forward(self, input):
        """
        input: [B, latent + xyz]
        """
        xyz = input[:, -self.geom_dimension :]
        latent = input[:, : -self.geom_dimension]

        if self.use_pe:
            xyz = self.positional_encoding(xyz)

        x = torch.cat([latent, xyz], dim=-1)

        residual = None
        for i, layer in enumerate(self.layers):
            x_in = x
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.act(x)

            if self.use_resnet and i % self.resnet_every == 0:

                if residual is None:
                    residual = x_in
                else:

                    if i < self.num_layers - 1:
                        if residual.shape == x.shape:
                            x = x + residual
                    residual = x_in

        return x

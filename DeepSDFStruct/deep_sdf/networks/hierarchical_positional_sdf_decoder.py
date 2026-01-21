import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

logger = logging.getLogger("DeepSDFStruct")

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
}


class HierachicalPositionalDeepSDFDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        geom_dimension,
        dropout=None,
        dropout_prob=0.0,
        latent_in=[],
        weight_norm=False,
        norm_layers=[],
        xyz_in_all=None,
        positional_encoding=False,
        pe_levels=6,
        split_latent=False,
        activation_fun="relu",
        latent_dropout=False,
        use_tanh=None,
    ):
        super(HierachicalPositionalDeepSDFDecoder, self).__init__()
        if use_tanh is not None:
            logger.warning("Use tanh is deprecated, use the activation_fun argument")

            # -----------------------------------------------------
        # Positional Encoding
        # -----------------------------------------------------
        self.use_pe = positional_encoding
        self.pe_levels = pe_levels

        if positional_encoding:
            # PE increases dimensionality from D -> D * (2*L + 1)
            self.xyz_dim = geom_dimension * (2 * pe_levels + 1)
        else:
            self.xyz_dim = geom_dimension

        self.geom_dimension = geom_dimension

        # -----------------------------------------------------
        # Base MLP structure
        # layer_dims[i] = input dimension of layer i
        # -----------------------------------------------------
        self.layer_dims = [
            self.xyz_dim
        ] + dims  # final layer out_dim is handled separately
        self.num_layers = len(dims)

        # -----------------------------------------------------
        # Latent injection logic
        # -----------------------------------------------------
        self.split_latent = split_latent
        self.latent_in = latent_in
        self.latent_size = latent_size
        self.latent_dropout = latent_dropout

        if self.split_latent:
            assert (
                latent_size % len(latent_in) == 0
            ), "latent_size must be divisible by number of latent injection layers"
            chunk_size = latent_size // len(latent_in)
            latent_chunk_sizes = [chunk_size] * len(latent_in)
        else:
            latent_chunk_sizes = [latent_size] * len(latent_in)

        self.latent_dims = {}  # map layer → latent_chunk_dim
        for size, layer_idx in zip(latent_chunk_sizes, latent_in):
            self.layer_dims[layer_idx] += size
            self.latent_dims[layer_idx] = size

        # -----------------------------------------------------
        # XYZ skip connection logic (xyz_in_all)
        # xyz is added BEFORE the linear layer, after activation,
        # but only for layers > 0
        # -----------------------------------------------------
        self.xyz_in_all = xyz_in_all
        if xyz_in_all:
            for layer in range(1, self.num_layers):
                self.layer_dims[layer] += self.xyz_dim

        # -----------------------------------------------------
        # Linear layer construction
        # -----------------------------------------------------
        self.weight_norm = weight_norm
        self.norm_layers = norm_layers

        for layer in range(self.num_layers):
            in_dim = self.layer_dims[layer]
            out_dim = dims[layer]

            linear = nn.Linear(in_dim, out_dim)
            if weight_norm and layer in norm_layers:
                linear = nn.utils.parametrizations.weight_norm(linear)

            setattr(self, f"lin{layer}", linear)

        # Output layer (final layer: size dims[-1] → 1)
        self.output_layer = nn.Linear(dims[-1], 1)

        # -----------------------------------------------------
        # Normalization + Dropout
        # -----------------------------------------------------
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layers = set(dropout)
        else:
            self.dropout_layers = set()

        # BatchNorm for chosen layers
        for layer in norm_layers:
            setattr(self, f"bn{layer}", nn.BatchNorm1d(dims[layer]))

        # Latent dropout
        if latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        # -----------------------------------------------------
        # Activation
        # -----------------------------------------------------
        if activation_fun not in activations:
            raise ValueError(
                f"Unsupported activation function {activation_fun}. "
                f"Choose from: {list(activations.keys())}"
            )
        self.activation = activations[activation_fun]

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

    # input: N x (L+3), or N x (L+geom_dimension)
    def forward(self, input: torch.Tensor):
        xyz = input[:, -self.geom_dimension :]
        latent_vecs = input[:, : -self.geom_dimension]
        if self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
        if self.split_latent:
            latent_chunks = torch.split(
                latent_vecs, int(self.latent_size / len(self.latent_in)), dim=1
            )
        else:
            latent_chunks = [latent_vecs for _ in self.latent_in]

        if self.use_pe:
            xyz = self.positional_encoding(xyz)
        x = xyz

        for layer in range(0, len(self.layer_dims) - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                latent_idx = self.latent_in.index(layer)
                latent_chunk = latent_chunks[latent_idx]
                x = torch.cat([x, latent_chunk], 1)

            if layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz])
            x = lin(x)

            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

            if layer <= (self.num_layers - 2):
                x = self.activation(x)

        return self.output_layer(x)

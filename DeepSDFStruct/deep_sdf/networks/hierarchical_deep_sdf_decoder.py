#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F

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


class HierachicalDeepSDFDecoder(nn.Module):
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
        activation_fun="relu",
        latent_dropout=False,
    ):
        super(HierachicalDeepSDFDecoder, self).__init__()

        self.layer_dims = [geom_dimension] + dims + [1]
        assert len(latent_size) == len(
            latent_in
        ), "latent shape must match latent in shape"
        # add latent dimensions to inputs
        self.latent_dims = {}
        for lat_size, index in zip(latent_size, latent_in):
            self.layer_dims[index] = self.layer_dims[index] + lat_size
            self.latent_dims[index] = lat_size

        self.num_layers = len(dims)
        self.geom_dimension = geom_dimension
        self.latent_size = latent_size
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.norm_layers = norm_layers

        for layer in range(0, len(self.layer_dims) - 1):
            if layer + 1 in latent_in:
                out_dim = self.layer_dims[layer + 1] - self.latent_dims[layer + 1]
            else:
                out_dim = self.layer_dims[layer + 1]

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.parametrizations.weight_norm(
                        nn.Linear(self.layer_dims[layer], out_dim)
                    ),
                )
            else:
                setattr(
                    self, "lin" + str(layer), nn.Linear(self.layer_dims[layer], out_dim)
                )

        if activation_fun not in activations:
            raise ValueError(
                f"Unsupported activation function '{activation_fun}'. "
                f"Available options are: {list(activations.keys())}"
            )

        self.activation = activations[activation_fun]

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3), or N x (L+geom_dimension)
    def forward(self, input):
        xyz = input[:, -self.geom_dimension :]
        latent_vecs = input[:, : -self.geom_dimension]
        latent_chunks = torch.split(latent_vecs, self.latent_size, dim=1)

        if self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
        x = xyz

        for layer in range(0, len(self.layer_dims) - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                latent_idx = self.latent_in.index(layer)
                latent_chunk = latent_chunks[latent_idx]
                x = torch.cat([x, latent_chunk], 1)
                # print(f"Latent chunk in layer {layer}: {latent_chunk}")
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

        return x

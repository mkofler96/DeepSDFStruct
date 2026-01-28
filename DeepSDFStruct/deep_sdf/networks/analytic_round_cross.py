"""
Analytic Round Cross Decoder Network
====================================

This module implements a specialized decoder architecture for round cross
lattice structures with analytical parametrization. The network incorporates
geometric priors specific to cross-shaped lattice unit cells.

The architecture is similar to the standard DeepSDF decoder but may include
specialized layers or constraints to better represent the round cross geometry
family, which is common in mechanical metamaterials.
"""

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch


class RoundCrossDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        geom_dimension,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(RoundCrossDecoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + geom_dimension] + dims + [1]

        self.num_layers = len(dims)
        self.geom_dimension = geom_dimension
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= geom_dimension

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.parametrizations.weight_norm(
                        nn.Linear(dims[layer], out_dim)
                    ),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3), or N x (L+geom_dimension)
    def forward(self, input):
        xyz = input[:, -self.geom_dimension :]
        r = input[:, 0]
        output = torch.linalg.norm(xyz, axis=1, ord=torch.inf)

        # add x cylinder
        cylinder = torch.sqrt(xyz[:, 1] ** 2 + xyz[:, 2] ** 2) - r
        output = torch.minimum(output, cylinder)
        # add y cylinder
        cylinder = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) - r
        output = torch.minimum(output, cylinder)
        # add z cylinder
        cylinder = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) - r
        output = torch.minimum(output, cylinder)

        return output.reshape(-1, 1)

#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # for F.pad

try:
    import pennylane as qml
except ImportError:
    raise ImportError(
        "pennylane is required for QuantumDeepSDFDecoder. "
        "Install with: uv sync --extra quantum"
    )


class QuantumDeepSDFDecoder(nn.Module):
    """
    Quantum Neural Network (QNN) drop-in replacement for DeepSDFDecoder.

    Maps (N, latent_size + geom_dimension) -> (N, 1) via a data-reuploading
    angle-encoded variational quantum circuit. The output is the Pauli-Z
    expectation value of qubit 0, which lies naturally in [-1, 1] matching
    the SDF target range — no classical readout layer needed.

    The input features are split into blocks of size n_qubits (zero-padded if
    needed). Each block is scaled by trainable per-feature weights before RY
    angle embedding, followed by a StronglyEntanglingLayers variational block.

    The full batch is evaluated in one call using PennyLane's native parameter
    broadcasting — the batch dimension of the input is broadcast internally,
    avoiding the torch.vmap/autograd.Function incompatibility.

    Parameters
    ----------
    latent_size : int
        Dimensionality of the latent code (default 16; combined with
        geom_dimension=3 gives 19 total features).
    geom_dimension : int
        Spatial dimension of xyz input (default 3).
    n_qubits : int
        Number of qubits (default 5). With 19 total features this gives
        ceil(19/5)=4 blocks → 20 slots → only 1 zero-padded slot,
        maximising encoding density at a tractable qubit count.
    n_variational_layers : int
        Number of StronglyEntanglingLayers per encoding block (default 1).
    n_repeats : int
        Number of times the full encode+variational sequence is repeated,
        each with independent weights (default 3). This is the data-reuploading
        depth of the circuit.
    **kwargs
        Absorbs unused classical NetworkSpecs keys: dims, dropout,
        dropout_prob, norm_layers, latent_in, weight_norm, xyz_in_all,
        use_tanh, latent_dropout.

    Notes
    -----
    - Exact statevector simulation: no shot noise, exact gradients.
    - Uses default.qubit with backprop (pure Python, compatible with native
      PennyLane broadcasting). lightning.qubit + adjoint was tested but slower
      in practice for this circuit size.
    - GPU acceleration provides no benefit at small qubit counts (< ~20).
    - DataParallel is not supported; set data_parallel=False in training config.
    - TorchScript / LibTorch export is not supported.
    """

    def __init__(
        self,
        latent_size: int = 16,
        geom_dimension: int = 3,
        n_qubits: int = 5,
        n_variational_layers: int = 1,
        n_repeats: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.geom_dimension = geom_dimension
        self.n_qubits = n_qubits
        total_features = latent_size + geom_dimension
        n_blocks = math.ceil(total_features / n_qubits)
        self.padded_dim = n_blocks * n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, enc_weights, var_weights):
            # inputs      : (N, padded_dim)
            # enc_weights : (n_repeats, n_blocks, n_qubits)
            # var_weights : (n_repeats, n_blocks, n_variational_layers, n_qubits, 3)
            # PennyLane broadcasts over the leading batch dim of inputs natively.
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for r in range(n_repeats):
                for i in range(n_blocks):
                    qml.AngleEmbedding(
                        enc_weights[r, i]
                        * inputs[:, i * n_qubits : (i + 1) * n_qubits],
                        wires=range(n_qubits),
                        rotation="X",
                    )
                    qml.StronglyEntanglingLayers(
                        var_weights[r, i], wires=range(n_qubits)
                    )
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

        # Both parameter tensors initialised in (-0.01π, +0.01π) — near-zero
        # init avoids barren plateaus by keeping gradients large at the start.
        def _near_zero(*shape):
            return (torch.rand(*shape) - 0.5) * 0.02 * math.pi

        self.enc_weights = nn.Parameter(_near_zero(n_repeats, n_blocks, n_qubits))
        self.q_weights = nn.Parameter(
            _near_zero(n_repeats, n_blocks, n_variational_layers, n_qubits, 3)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (N, latent_size + geom_dimension)
        pad = self.padded_dim - input.shape[1]
        x = F.pad(input, (0, pad)) if pad > 0 else input
        # x: (N, padded_dim) — evaluate all samples in one vectorised call
        q_out = self._circuit(x, self.enc_weights, self.q_weights)  # (N,), in [-1, 1]
        return q_out.float().unsqueeze(-1)  # (N, 1), in [-1, 1]

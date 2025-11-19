"""
DeepSDF Model Classes
=====================

This module provides the DeepSDFModel class, which wraps a trained decoder
network and latent vectors for evaluating learned implicit representations.

The model can be used to:
- Decode latent codes to SDF values at query points
- Export models for deployment (TorchScript/LibTorch)
- Integrate with gradient-based optimization

Classes
-------
DeepSDFModel
    Main interface for trained DeepSDF models, combining a decoder network
    with learned latent vectors.
"""

import torch


class DeepSDFModel:
    """Wrapper for trained DeepSDF decoder and latent vectors.
    
    This class provides a convenient interface for using trained DeepSDF models.
    It combines the decoder network (which maps latent+position to SDF values)
    with the trained latent vectors (which encode different shapes).
    
    The model supports both constant latent codes (one code for all query points)
    and spatially-varying codes (different codes per query point), enabling
    flexible shape representation.
    
    Parameters
    ----------
    decoder : torch.nn.Module
        The trained decoder network. Should accept input of shape
        (N, latent_dim + 3) and output SDF values of shape (N, 1).
    trained_latent_vectors : torch.Tensor
        Trained latent codes of shape (num_shapes, latent_dim).
        Each row encodes one learned shape.
    device : str or torch.device
        Device for computation ('cpu' or 'cuda').
        
    Attributes
    ----------
    _decoder : torch.nn.Module
        The decoder network.
    _trained_latent_vectors : torch.Tensor
        The latent code library.
    device : str or torch.device
        Computation device.
        
    Methods
    -------
    _decode_sdf(latent_vec, queries)
        Decode SDF values from a latent code and spatial queries.
    export_libtorch_executable(filename)
        Export model to TorchScript for C++ deployment.
        
    Examples
    --------
    >>> import torch
    >>> from DeepSDFStruct.deep_sdf.models import DeepSDFModel
    >>> 
    >>> # Assume we have a trained decoder and latent vectors
    >>> # decoder = ...
    >>> # latents = torch.randn(10, 256)  # 10 shapes, 256-dim codes
    >>> 
    >>> # Create model
    >>> model = DeepSDFModel(decoder, latents, device='cuda')
    >>> 
    >>> # Query first shape
    >>> points = torch.rand(1000, 3, device='cuda')
    >>> distances = model._decode_sdf(latents[0], points)
    >>> 
    >>> # Use with SDFfromDeepSDF
    >>> from DeepSDFStruct.SDF import SDFfromDeepSDF
    >>> sdf = SDFfromDeepSDF(model, latent_code=latents[0])
    >>> mesh = create_3D_mesh(sdf, N_base=64, mesh_type='surface')
    
    Notes
    -----
    The decoder architecture typically consists of multiple fully-connected
    layers with skip connections. See networks/ for architecture definitions.
    
    References
    ----------
    .. [1] Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S.
           (2019). DeepSDF: Learning continuous signed distance functions for
           shape representation. In CVPR.
    """
    
    def __init__(
        self, decoder: torch.nn.Module, trained_latent_vectors: torch.Tensor, device
    ):
        self._decoder = decoder
        self._trained_latent_vectors = trained_latent_vectors
        self.device = device

    def _decode_sdf(
        self, latent_vec: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """Decode SDF values from latent vector and xyz queries.
        
        Combines latent codes with spatial coordinates and passes through
        the decoder to obtain SDF values. Handles both per-query and
        constant latent vectors.
        
        Parameters
        ----------
        latent_vec : torch.Tensor
            Latent code(s). Two formats supported:
            - Shape (latent_dim,): Constant code used for all queries
            - Shape (num_samples, latent_dim): Per-query codes
        queries : torch.Tensor
            Query point coordinates of shape (num_samples, 3).
            
        Returns
        -------
        torch.Tensor
            SDF values of shape (num_samples, 1).
            
        Raises
        ------
        ValueError
            If latent_vec shape doesn't match expected dimensions.
            
        Notes
        -----
        If latent_vec is constant (1D), it's expanded to match the number
        of query points. If latent_vec is per-query (2D), it must have
        exactly one code per query point.
        """
        latent_dim = self._trained_latent_vectors[0].shape[0]
        num_samples = queries.shape[0]
        if latent_vec.ndim == 1:
            if latent_vec.shape[0] != latent_dim:
                raise ValueError(
                    f"Latent vector shape mismatch: {latent_vec.shape} does"
                    f"not align with latent dimension {latent_dim}."
                )
            latent_repeat = latent_vec.expand(-1, num_samples).T
        elif latent_vec.ndim == 2:
            if (latent_vec.shape[0] != num_samples) or (
                latent_vec.shape[1] != latent_dim
            ):
                raise ValueError(
                    f"Latent vector shape mismatch: {latent_vec.shape} does"
                    f" not align with {num_samples} queries."
                    f" Must be of shape ({num_samples}, {latent_dim})"
                )
            latent_repeat = latent_vec

        model_input = torch.cat([latent_repeat, queries], dim=1)
        return self._decoder(model_input)

    def export_libtorch_executable(self, filename: str):
        """
        Export the trained decoder model to a TorchScript file for use with LibTorch (C++).

        Args:
            filename (str): Path where the TorchScript model will be saved (e.g. "decoder.pt").

        Example:
            >>> model.export_libtorch_executable("decoder.pt")
            Example input:  tensor([[...]])
            Example Output: tensor([[...]])
            # The file "decoder.pt" is now ready for loading in LibTorch.
        """
        assert isinstance(
            self._trained_latent_vectors, torch.Tensor
        ), "trained_latent_vectors must be a tensor"
        assert (
            self._trained_latent_vectors.shape[0] > 0
        ), "trained_latent_vectors must contain at least one element"
        latent = self._trained_latent_vectors
        example_input = torch.cat(
            [latent[0], torch.tensor([0, 0, 0], device=self.device)]
        ).unsqueeze(0)

        print("Example input: ", example_input)
        print("Example Output: ", self._decoder(example_input))

        decoder_traced = torch.jit.trace(self._decoder, example_input)
        sm = torch.jit.script(decoder_traced)

        sm.save(filename)

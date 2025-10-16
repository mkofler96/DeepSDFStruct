import torch


class DeepSDFModel:
    def __init__(
        self, decoder: torch.nn.Module, trained_latent_vectors: torch.Tensor, device
    ):
        self._decoder = decoder
        self._trained_latent_vectors = trained_latent_vectors
        self.device = device

    def _decode_sdf(
        self, latent_vec: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode SDF values from latent vector and xyz queries.
        Handles both per-query and constant latent vectors.
        If latent vec is constant, the latent vector must be a flat tensor of
        length d_lat, otherwise [n_samples, d_lat]
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

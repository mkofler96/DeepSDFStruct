import torch


class DeepSDFModel:
    def __init__(
        self,
        decoder: torch.nn.Module,
        trained_latent_vectors: list[torch.Tensor],
        device,
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
        """
        num_samples = queries.shape[0]

        # todo: check something like this
        # raise ValueError(
        #     f"Latent vector shape mismatch: {latent_vec.shape} does not align with {num_samples} queries."
        # )

        if latent_vec.shape[0] == num_samples:
            latent_repeat = latent_vec
        else:
            latent_repeat = latent_vec.expand(-1, num_samples).T

        model_input = torch.cat([latent_repeat, queries], dim=1)
        return self._decoder(model_input)

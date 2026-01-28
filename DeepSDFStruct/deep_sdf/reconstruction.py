"""
Shape Reconstruction and Latent Code Optimization
=================================================

This module provides methods for reconstructing shapes from SDF samples
by optimizing latent codes. This is a key capability of DeepSDF models,
enabling fitting of new geometries not seen during training.

Key Functions
-------------

reconstruct_from_samples
    Optimize a latent code to match given SDF samples. Uses gradient
    descent to find the latent representation that best reproduces
    the target geometry.

The reconstruction process:
1. Start with initial latent code (often zeros or random)
2. Forward pass through decoder to get SDF predictions
3. Compute loss between predictions and target samples
4. Backpropagate and update latent code
5. Repeat until convergence

This enables:
- Fitting trained models to new geometries
- Shape interpolation by blending latent codes
- Shape editing by modifying latent codes
- Compression by storing only latent codes

Examples
--------
Reconstruct a shape from samples::

    from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
    from DeepSDFStruct.sampling import sample_sdf_near_surface

    # Sample target geometry
    target_samples = sample_sdf_near_surface(target_sdf, n_samples=100000)

    # Reconstruct using trained model
    reconstructed_sdf = reconstruct_from_samples(
        model,
        target_samples,
        num_iterations=1000,
        lr=5e-4
    )
"""

from DeepSDFStruct.SDF import SDFBase
from DeepSDFStruct.deep_sdf.training import ClampedL1Loss
from torch.utils.data import TensorDataset, DataLoader
from DeepSDFStruct.sampling import SampledSDF
from DeepSDFStruct.deep_sdf.plotting import plot_reconstruction_loss
from tqdm import trange
import torch


def reconstruct_from_samples(
    sdf: SDFBase,
    sdfSample: SampledSDF,
    num_iterations=1000,
    lr=5e-4,
    device="cpu",
    dtype=torch.float32,
    loss_fn="ClampedL1",
    batch_size=512,
    drop_last=True,
    loss_plot_path=None,
):

    optimizer = torch.optim.Adam(sdf.parameters(), lr=lr)

    verts_min = sdfSample.samples.min(axis=0)
    verts_max = sdfSample.samples.max(axis=0)

    print("Min/Max in PHYSICAL space:\n")
    for name, mn, mx in zip(["x", "y", "z"], verts_min.values, verts_max.values):
        print(f"{name}: min={mn:.6f}, max={mx:.6f}")

    # return verbose can be enabled to get more information from splinepy
    return_verbose = False
    queries_parameter_space = sdf.deformation_spline.spline.proximities(
        sdfSample.samples.detach().cpu().numpy(), return_verbose=return_verbose
    )
    if return_verbose:
        (
            queries_parameter_space,
            phys_coord,
            phys_diff,
            distance,
            convergence_norm,
            first_derivatives,
            second_derivatives,
        ) = queries_parameter_space
    queries_ps_torch = torch.tensor(queries_parameter_space, device=device, dtype=dtype)
    queries_min = queries_ps_torch.min(dim=0).values
    queries_max = queries_ps_torch.max(dim=0).values

    print("\nMin/Max in QUERY space:\n")
    for name, mn, mx in zip(
        ["x", "y", "z"], queries_min.tolist(), queries_max.tolist()
    ):
        print(f"{name}: min={mn:.6f}, max={mx:.6f}")

    gt_dist = sdfSample.distances

    pbar = trange(num_iterations, desc="Reconstructing SDF from mesh", leave=True)

    if loss_fn == "L1":
        Loss = torch.nn.L1Loss()
    elif loss_fn == "ClampedL1":
        Loss = ClampedL1Loss(clamp_val=0.1)
    elif loss_fn == "MSE":
        Loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function {loss_fn} not available.")

    dataset = TensorDataset(queries_ps_torch, gt_dist)
    if drop_last and (batch_size > len(dataset)):
        print(
            "Warning: drop_last was set to true, "
            f"but batch size ({batch_size}) is larger "
            f"than the size of the dataset ({len(dataset)})."
            " setting drop_last=False"
        )
        drop_last = False
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    loss_history = []
    for e in pbar:
        for querie_batch, gt_batch in dataloader:
            optimizer.zero_grad()
            pred_dist = sdf(querie_batch)
            loss = Loss(pred_dist, gt_batch)
            loss.backward()
            optimizer.step()
            loss_num = loss.detach().item()
            pbar.set_postfix({"loss": f"{loss_num:.5f}"})
            loss_history.append(loss_num)

    if loss_plot_path is not None:
        plot_reconstruction_loss(
            loss_history, iters_per_epoch=len(dataloader), filename=loss_plot_path
        )

        params = list(sdf.parameters())
        print(params)

        return params


def reconstruct_deepLS_from_samples(
    sdf: SDFBase,
    sdfSample: SampledSDF,
    num_iterations=1000,
    lr=5e-4,
    device="cpu",
    dtype=torch.float32,
    loss_fn="ClampedL1",
    batch_size=512,
    drop_last=True,
):

    optimizer = torch.optim.Adam(sdf.parameters(), lr=lr)

    gt_dist = sdfSample.distances

    pbar = trange(num_iterations, desc="Reconstructing SDF from mesh", leave=True)

    if loss_fn == "L1":
        Loss = torch.nn.L1Loss()
    elif loss_fn == "ClampedL1":
        Loss = ClampedL1Loss(clamp_val=0.1)
    elif loss_fn == "MSE":
        Loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function {loss_fn} not available.")

    dataset = TensorDataset(sdfSample.samples, gt_dist)
    if drop_last and (batch_size > len(dataset)):
        print(
            "Warning: drop_last was set to true, "
            f"but batch size ({batch_size}) is larger "
            f"than the size of the dataset ({len(dataset)})."
            " setting drop_last=False"
        )
        drop_last = False
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    for e in pbar:
        for querie_batch, gt_batch in dataloader:
            optimizer.zero_grad()
            pred_dist = sdf(querie_batch)
            loss = Loss(pred_dist, gt_batch)
            loss.backward()
            optimizer.step()
            loss_num = loss.detach().item()
            pbar.set_postfix({"loss": f"{loss_num:.5f}"})

    print("Reconstructed parameters:")
    params = list(sdf.parameters())
    print(params)
    return params

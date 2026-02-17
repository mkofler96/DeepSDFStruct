from DeepSDFStruct.SDF import SDFBase
from DeepSDFStruct.deep_sdf.training import ClampedL1Loss
from torch.utils.data import TensorDataset, DataLoader
from DeepSDFStruct.sampling import SampledSDF
from DeepSDFStruct.deep_sdf.plotting import plot_reconstruction_loss
from DeepSDFStruct.torch_spline import TorchSpline, TorchScaling
from tqdm import trange
import torch
import torch.nn as nn


def reconstruct_from_samples(
    sdf: SDFBase,
    sdfSample: SampledSDF,
    num_iterations=1000,
    lr=5e-4,
    loss_fn="ClampedL1",
    batch_size=512,
    drop_last=True,
    use_tanh_on_gt=False,
    loss_plot_path=None,
    optimizer_name="adam",
    deformation_function=None | TorchSpline | TorchScaling,
):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(sdf.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            sdf.parameters(),
            lr=lr,
            max_iter=20,  # inner Newton iterations
            history_size=100,  # curvature memory
            line_search_fn="strong_wolfe",
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not available.")

    verts_min = sdfSample.samples.min(axis=0)
    verts_max = sdfSample.samples.max(axis=0)

    print("Min/Max in PHYSICAL space:\n")
    for name, mn, mx in zip(["x", "y", "z"], verts_min.values, verts_max.values):
        print(f"{name}: min={mn:.6f}, max={mx:.6f}")

    if deformation_function is not None:
        queries_parameter_space = deformation_function.inverse_target_points(
            sdfSample.samples
        ).detach()
    else:
        queries_parameter_space = sdfSample.samples.detach()

    queries_min = queries_parameter_space.min(dim=0).values
    queries_max = queries_parameter_space.max(dim=0).values

    print("\nMin/Max in QUERY space:\n")
    for name, mn, mx in zip(
        ["x", "y", "z"], queries_min.tolist(), queries_max.tolist()
    ):
        print(f"{name}: min={mn:.6f}, max={mx:.6f}")

    gt_dist = sdfSample.distances
    if use_tanh_on_gt:
        gt_dist = torch.tanh(gt_dist)

    pbar = trange(num_iterations, desc="Reconstructing SDF from mesh", leave=True)

    if loss_fn == "L1":
        Loss = torch.nn.L1Loss()
    elif loss_fn == "ClampedL1":
        Loss = ClampedL1Loss(clamp_val=0.1)
    elif loss_fn == "MSE":
        Loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function {loss_fn} not available.")

    dataset = TensorDataset(queries_parameter_space, gt_dist)
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

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                pred_dist = sdf(querie_batch)
                loss = Loss(pred_dist, gt_batch)
                loss.backward()
                return loss

            if optimizer_name == "adam":
                loss = closure()
                optimizer.step()
            elif optimizer_name == "lbfgs":
                loss = optimizer.step(closure)
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

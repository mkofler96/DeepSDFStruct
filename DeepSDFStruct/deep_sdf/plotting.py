import json
import vedo
import numpy as np
import os
import gustaf as gus
import logging
import torch
import matplotlib.pyplot as plt

import deep_sdf.workspace as ws
import deep_sdf.mesh
import trimesh


def extract_paths(data, current_path=""):
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}/{key}" if current_path else key
            paths.extend(extract_paths(value, new_path))

    elif isinstance(data, list):
        for item in data:
            paths.extend(extract_paths(item, current_path))

    else:
        paths.append(f"{current_path}/{data}")

    return paths


def show_random_reconstruction(experiment_directory, epoch=None):
    if epoch is None:
        epoch = "latest"
    specs = ws.load_experiment_specifications(experiment_directory)
    npz_filenames = extract_paths(json.load(open(specs["TrainSplit"])))
    print(specs)
    id = np.random.choice(len(npz_filenames), 1, replace=False)
    model = deep_sdf.ws.load_trained_model(experiment_directory, str(epoch))
    latent_vecs = deep_sdf.ws.load_latent_vectors(experiment_directory, str(epoch))
    verts, faces = deep_sdf.mesh.create_mesh_microstructure(
        [1, 1, 1], model, deep_sdf.mesh.constantLatvec(latent_vecs[id[0]]), N=64
    )
    mesh = trimesh.Trimesh(verts, faces)
    return mesh.show()


def show_random_training_files(
    experiment_directory, n_files=4, epoch=None, only_show_halve=False
):
    specs = ws.load_experiment_specifications(experiment_directory)
    npz_filenames = extract_paths(json.load(open(specs["TrainSplit"])))
    ids = np.random.choice(len(npz_filenames), n_files, replace=False)
    plots = []
    for plt_id, id in enumerate(ids):
        current_plot = []
        plt = vedo.Plotter(axes=1)
        npz_filename = npz_filenames[id]
        full_filename = os.path.join(
            specs["DataSource"], "SdfSamples", npz_filename + ".npz"
        )
        points = np.load(full_filename)
        all_points = np.vstack([points["neg"], points["pos"]])
        if only_show_halve:
            mask = np.where(all_points[:, 2] < 0)
            all_points = all_points[mask]
        points = gus.vertices.Vertices(all_points[:, :3])
        points.vertex_data["sdf"] = all_points[:, -1]
        points.show_options["data"] = "sdf"
        points.show_options["cmap"] = "coolwarm"
        points.show_options["vmin"] = -0.1
        points.show_options["vmax"] = 0.1
        points.show_options["r"] = 10
        points.show_options["axes"] = True
        current_plot.append(points)

        # reconstruct the mesh
        if epoch is not None:
            ply_file = deep_sdf.mesh.create_mesh_from_latent(
                experiment_directory, epoch, id
            )
            try:
                mesh = gus.io.meshio.load(ply_file)
            except ValueError:
                logging.warning(f"Reconstruction for {npz_filename} not found")
                continue
            current_plot.append(mesh)
        plots.append(current_plot)
    gus.show(*plots)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_logs(experiment_directory, show_lr=False, ax=None):

    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))

    logging.info("latest epoch is {}".format(logs["epoch"]))

    num_iters = len(logs["loss"])
    iters_per_epoch = num_iters / logs["epoch"]

    logging.info("{} iters per epoch".format(iters_per_epoch))

    smoothed_loss_41 = running_mean(logs["loss"], 41)
    smoothed_loss_1601 = running_mean(logs["loss"], 1601)

    show_plt = False

    if show_lr:
        if ax is None:
            fig, ax = plt.subplots(1, 2)
            fig.tight_layout()
            show_plt = True
    else:
        if ax is None:
            fig, ax = plt.subplots()
            show_plt = True
        ax = [ax]

    ax[0].plot(
        np.arange(num_iters) / iters_per_epoch,
        logs["loss"],
        "#82c6eb",
        np.arange(20, num_iters - 20) / iters_per_epoch,
        smoothed_loss_41,
        "#2a9edd",
    )

    ax[0].set(xlabel="Epoch", ylabel="Loss")
    ax[0].legend(["Loss", "Loss (Running Mean)", "Loss (Running Mean 1601)"])

    if show_lr:
        combined_lrs = np.array(logs["learning_rate"])
        ax[1].plot(
            np.arange(combined_lrs.shape[0]),
            combined_lrs[:, 0],
            np.arange(combined_lrs.shape[0]),
            combined_lrs[:, 1],
        )
        ax[1].set(xlabel="Epoch", ylabel="Learning Rate")
        ax[1].legend(["Decoder", "Latent Vector"])
    # elif type == "time":
    #     ax.plot(logs["timing"], "#833eb7")
    #     ax.set(xlabel="Epoch", ylabel="Time per Epoch (s)", title="Timing")

    # elif type == "lat_mag":
    #     ax.plot(logs["latent_magnitude"])
    #     ax.set(xlabel="Epoch", ylabel="Magnitude", title="Latent Vector Magnitude")

    # elif type == "param_mag":
    #     for _name, mags in logs["param_magnitude"].items():
    #         ax.plot(mags)
    #     ax.set(xlabel="Epoch", ylabel="Magnitude", title="Parameter Magnitude")
    #     ax.legend(logs["param_magnitude"].keys())

    # else:
    #     raise Exception('unrecognized plot type "{}"'.format(type))
    for axis in ax:
        axis.grid()
    if show_plt:
        plt.show()

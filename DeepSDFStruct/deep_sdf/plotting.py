import numpy as np
import os
import logging
import torch
import matplotlib.pyplot as plt

import DeepSDFStruct.deep_sdf.workspace as ws


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


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_logs(experiment_directory, show_lr=False, ax=None, filename=None):

    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))

    logging.info("latest epoch is {}".format(logs["epoch"]))

    num_iters = len(logs["loss"])
    iters_per_epoch = num_iters / logs["epoch"]

    logging.info("{} iters per epoch".format(iters_per_epoch))

    smoothed_loss_41 = running_mean(logs["loss"], 41)

    show_plt = False

    if show_lr:
        if ax is None:
            fig, ax = plt.subplots(2, 1)
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
    ax[0].set_yscale("log")

    ax[0].set(xlabel="Epoch", ylabel="Loss")
    ax[0].legend(["Loss", "Loss (Running Mean)", "Loss (Running Mean 41)"])

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

    for axis in ax:
        axis.grid()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    elif show_plt:
        plt.show()

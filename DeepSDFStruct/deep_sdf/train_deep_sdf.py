#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import datetime
import random
import pathlib

import DeepSDFStruct.deep_sdf
import DeepSDFStruct.deep_sdf.workspace as ws
import DeepSDFStruct.deep_sdf.data

import numpy as np


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def train_deep_sdf(
    experiment_directory, data_source, continue_from=None, batch_split=1, device=None
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    logging.debug("running " + experiment_directory)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        device_name = torch.cuda.get_device_name()
    elif device == "cpu":
        device_name = "cpu"
    else:
        raise RuntimeError("Device must be either cpu or cuda")

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    reconstruction_split_file = specs["ReconstructionSplit"]
    if os.path.isfile(reconstruction_split_file):
        with open(reconstruction_split_file, "r") as f:
            reconstruction_split = json.load(f)
    else:
        reconstruction_split = None

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        if torch.cuda.is_available():
            lat_mat = torch.zeros(0).cuda()
        else:
            lat_mat = torch.zeros(0)
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    if torch.cuda.device_count() > 1:
        data_parallel = True
    else:
        data_parallel = False

    decoder = ws.init_decoder(specs, device, data_parallel)

    geom_dimension = decoder.geom_dimension
    logging.info(f"training on {device_name}")

    seed = get_spec_with_default(specs, "seed", 42)
    logging.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    train_split_file = pathlib.Path(data_source) / specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = DeepSDFStruct.deep_sdf.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        load_ram=True,
        geom_dimension=geom_dimension,
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(
        num_scenes, latent_size, max_norm=code_bound, device=device
    )
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    if do_code_regularization == "homogenization":
        loss_MSE = torch.nn.MSELoss(reduction="mean")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    start_train = time.time()
    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for sdf_data, properties, indices in sdf_loader:
            # Process the input data
            sdf_data = sdf_data.reshape(-1, geom_dimension + 1).to(device)
            properties = properties.to(device)
            indices = indices.to(device)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:geom_dimension]
            sdf_gt = sdf_data[:, geom_dimension].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0
            batch_reg_loss = 0.0
            optimizer_all.zero_grad()

            for i in range(batch_split):
                batch_lat_vecs = lat_vecs(indices[i])

                input = torch.cat([batch_lat_vecs, xyz[i]], dim=1)

                # NN optimization
                pred_sdf = decoder(input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                if torch.cuda.is_available():
                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                else:
                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i]) / num_sdf_samples

                if do_code_regularization == "homogenization":
                    unique_lat_vecs = lat_vecs(indices[i].unique())
                    if torch.cuda.is_available():
                        pred_properties = decoder.module.regressor(
                            unique_lat_vecs.cuda()
                        )
                        reg_loss = code_reg_lambda * loss_MSE(
                            pred_properties, properties.cuda()
                        )
                        chunk_loss = chunk_loss + reg_loss.cuda()
                        batch_reg_loss = batch_reg_loss + reg_loss.cuda()
                    else:
                        pred_properties = decoder.module.regressor(batch_lat_vecs)
                        reg_loss = code_reg_lambda * loss_MSE(
                            pred_properties, properties
                        )
                        chunk_loss = chunk_loss + reg_loss
                        batch_reg_loss = batch_reg_loss + reg_loss

                l2_size_loss = torch.sum(torch.norm(batch_lat_vecs, dim=1))
                reg_loss = (
                    code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                ) / num_sdf_samples
                if torch.cuda.is_available():
                    chunk_loss = chunk_loss + reg_loss.cuda()
                    batch_reg_loss = batch_reg_loss + reg_loss.cuda()
                else:
                    chunk_loss = chunk_loss + reg_loss
                    batch_reg_loss = batch_reg_loss + reg_loss

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)
            grad_clip = 1.0
            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()
        # logging.info("epoch {}...".format(epoch))
        tot_time = time.time() - start_train
        avg_time_per_epoch = tot_time / (epoch)
        estimated_remaining_time = avg_time_per_epoch * (num_epochs - (epoch))
        time_string = str(datetime.timedelta(seconds=round(estimated_remaining_time)))
        if epoch == num_epochs:
            total_time = str(datetime.timedelta(seconds=round(tot_time)))
            logging.info(
                f"Finished {epoch} ({epoch}/{num_epochs}) [{epoch/num_epochs*100:.2f}%] after {total_time}"
            )
        else:
            logging.info(
                f"Finished epoch {epoch:5g}/{num_epochs} | "
                f"with Reg.: {batch_reg_loss:.4f} "
                f"and Tot.: {batch_loss:.4f} "
                f"[{epoch/num_epochs*100:.2f}%] in {time_string} "
                f"({avg_time_per_epoch:.2f}s/epoch)"
            )
        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        default=None,
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    DeepSDFStruct.deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    DeepSDFStruct.deep_sdf.configure_logging(args)
    if args.continue_from is not None:
        continue_from = str(args.continue_from)
    else:
        continue_from = None
    train_deep_sdf(args.experiment_directory, continue_from, int(args.batch_split))

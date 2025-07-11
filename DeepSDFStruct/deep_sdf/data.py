#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import DeepSDFStruct.deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """ "Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


# and also convert to single precision
# usually one function should only contain one functionality, but here I think
# it makes sense, because the data is loaded at 3 different positions in the code
def remove_nans(tensor, geom_dimension):
    tensor_nan = torch.isnan(tensor[:, geom_dimension])
    return tensor[~tensor_nan, :].float()


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos.npy"]).float()
    neg_tensor = torch.from_numpy(npz["neg.npy"]).float()

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, geom_dimension, subsample=None):
    npz = np.load(filename)

    pos_tensor = remove_nans(torch.from_numpy(npz["pos.npy"]), geom_dimension)
    neg_tensor = remove_nans(torch.from_numpy(npz["neg.npy"]), geom_dimension)

    if subsample is None:
        return torch.cat([pos_tensor, neg_tensor], 0)

    half = int(subsample / 2)
    pos_len = len(pos_tensor)
    neg_len = len(neg_tensor)
    if pos_len < half:
        neg_len = 2 * half - pos_len
    elif neg_len < half:
        pos_len = 2 * half - neg_len
    else:
        pos_len = neg_len = half

    use_randperm = True
    if use_randperm:
        random_pos = torch.randperm(len(pos_tensor))[:pos_len]
        random_neg = torch.randperm(len(neg_tensor))[:neg_len]
    else:
        # split the sample into half
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    # if len(samples) < 50:
    # print("less than half")

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return torch.cat(data, 0)
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    if pos_size <= half:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    else:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        geom_dimension,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample
        self.geom_dimension = geom_dimension
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)
        self.filenames = []
        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        self.loaded_mat_properties = []
        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(
                    torch.from_numpy(npz["pos.npy"]), self.geom_dimension
                )
                neg_tensor = remove_nans(
                    torch.from_numpy(npz["neg.npy"]), self.geom_dimension
                )
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
                if "E" in npz.keys():
                    self.loaded_mat_properties.append(
                        torch.from_numpy(npz["E.npy"]).to(torch.float32)
                    )
                self.filenames.append(filename)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if len(self.loaded_mat_properties) != 0:
            mat_prop = self.loaded_mat_properties[idx]
        else:
            mat_prop = torch.tensor(0)
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                mat_prop,
                idx,
            )
        else:
            return (
                unpack_sdf_samples(filename, self.geom_dimension, self.subsample),
                mat_prop,
                idx,
            )

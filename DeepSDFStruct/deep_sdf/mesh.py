#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import splinepy as sp
import time
import torch
import pathlib
import os

from typing import TypedDict

import DeepSDFStruct.deep_sdf.utils
from DeepSDFStruct.flexicubes.flexicubes import FlexiCubes

logger = logging.getLogger(__name__)


def create_mesh(decoder, latent_vec, N=256, max_batch=32**3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.eval()
    flexi_cubes_constructor = FlexiCubes(device=device)

    samples, cube_idx = flexi_cubes_constructor.construct_voxel_grid(
        resolution=(N, N, N)
    )
    # transform samples from [-0.5, 0.5] to [-1, 1]
    samples = samples.to(device) * 2
    cube_idx = cube_idx.to(device)

    head = 0
    latent_vec = torch.tensor(latent_vec).to(device)
    num_samples = len(samples)
    sdf_values = torch.zeros(samples.shape[0]).to(device)
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(
            device
        )

        sdf_values[head : min(head + max_batch, num_samples)] = (
            DeepSDFStruct.deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    verts, faces, loss = flexi_cubes_constructor(
        voxelgrid_vertices=torch.tensor(samples[:, :3]).to(device),
        scalar_field=sdf_values.view(-1),
        cube_idx=cube_idx,
        resolution=(N, N, N),
        output_tetmesh=False,
    )
    return verts, faces


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    if not isinstance(voxel_size, list):
        voxel_size = [voxel_size] * 3
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=voxel_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logger.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logger.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


location_lookup = {
    "x0": (0, -1),
    "x1": (0, 1),
    "y0": (1, -1),
    "y1": (1, 1),
    "z0": (2, -1),
    "z1": (2, 1),
}


class CapType(TypedDict):
    cap: int
    measure: float


class CapBorderDict(TypedDict):
    x0: CapType = {"cap": -1, "measure": 0}
    x1: CapType = {"cap": -1, "measure": 0}
    y0: CapType = {"cap": -1, "measure": 0}
    y1: CapType = {"cap": -1, "measure": 0}
    z0: CapType = {"cap": -1, "measure": 0}
    z1: CapType = {"cap": -1, "measure": 0}


def create_mesh_microstructure(
    tiling,
    decoder,
    latent_vec_interpolation,
    filename=None,
    N=256,
    max_batch=32**3,
    offset=None,
    scale=None,
    cap_border_dict: CapBorderDict = None,
    save_ply_file=False,
    use_flexicubes=False,
    device=None,
    output_tetmesh=False,
    compute_derivatives=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cap_border_dict is None:
        cap_border_dict = CapBorderDict(
            x0={"cap": -1, "measure": 0.01},
            x1={"cap": -1, "measure": 0.01},
            y0={"cap": -1, "measure": 0.01},
            y1={"cap": -1, "measure": 0.01},
            z0={"cap": -1, "measure": 0.01},
            z1={"cap": -1, "measure": 0.01},
        )

    if isinstance(tiling, list):
        if len(tiling) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        tiling = np.array(tiling)
    elif isinstance(tiling, int):
        tiling = np.array([tiling, tiling, tiling])
    else:
        raise ValueError("Tiling must be a list or an integer")

    # add 1 on each side to slightly include the border
    if isinstance(N, list):
        if len(N) != 3:
            raise ValueError("Number of grid points must be a list of 3 integers")
        N = np.array(N) + 2
    elif isinstance(N, int):
        N = np.array([N, N, N]) + 2
    else:
        raise ValueError("Number of grid points must be a list or an integer")

    start = time.time()

    decoder.eval()

    if use_flexicubes:
        reconstructor = DeepSDFStruct.flexicubes(device=device)
        samples_orig, cube_idx = reconstructor.construct_voxel_grid(resolution=tuple(N))
        samples_orig = samples_orig.to(device)
        cube_idx = cube_idx.to(device)
        # transform samples from [-0.5, 0.5] to [-1.05, 1.05]
        samples_orig = samples_orig * 2.1
        N_tot = samples_orig.shape[0]
        N = N + 1
    else:
        N_tot = N[0] * N[1] * N[2]
        overall_index = torch.arange(0, N_tot, 1, out=torch.LongTensor())
        samples_orig = torch.zeros(N_tot, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples_orig[:, 2] = overall_index % N[2]
        samples_orig[:, 1] = (overall_index // N[2]) % N[1]
        samples_orig[:, 0] = ((overall_index // N[2]) // N[1]) % N[0]

        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_size_x = 2.0 / (N[0] - 1 - 2)
        voxel_size_y = 2.0 / (N[1] - 1 - 2)
        voxel_size_z = 2.0 / (N[2] - 1 - 2)
        voxel_size = [voxel_size_x, voxel_size_y, voxel_size_z]
        voxel_origin = [-1 - voxel_size_x, -1 - voxel_size_y, -1 - voxel_size_z]
        # transform first 3 columns
        # to be the x, y, z coordinate
        samples_orig[:, 0] = (samples_orig[:, 0] * voxel_size_x) + voxel_origin[0]
        samples_orig[:, 1] = (samples_orig[:, 1] * voxel_size_y) + voxel_origin[1]
        samples_orig[:, 2] = (samples_orig[:, 2] * voxel_size_z) + voxel_origin[2]

    # samples = [-1, 1]
    tx, ty, tz = tiling

    def transform(x, t):
        # transform x from [-1,1] to [0,1]
        x = (x + 1) / 2
        return 4 * torch.abs(t * x / 2 - torch.floor((t * x + 1) / 2)) - 1

    samples = torch.zeros(N_tot, 4)
    samples[:, 0] = transform(samples_orig[:, 0], tx)
    samples[:, 1] = transform(samples_orig[:, 1], ty)
    samples[:, 2] = transform(samples_orig[:, 2], tz)

    num_samples = N_tot

    samples.requires_grad = False

    head = 0
    inside_domain = torch.where(
        (samples_orig[:, 0] >= -1)
        & (samples_orig[:, 0] <= 1)
        & (samples_orig[:, 1] >= -1)
        & (samples_orig[:, 1] <= 1)
        & (samples_orig[:, 2] >= -1)
        & (samples_orig[:, 2] <= 1)
    )
    lat_vec_red = torch.zeros(
        (samples_orig.shape[0], latent_vec_interpolation.control_points[0].shape[0])
    )
    lat_vec_red[inside_domain] = torch.tensor(
        latent_vec_interpolation.evaluate(
            samples_orig[:, 0:3][inside_domain].cpu().numpy()
        ),
        dtype=torch.float32,
    )
    queries = torch.hstack(
        [torch.tensor(lat_vec_red).to(torch.float32), samples[:, 0:3]]
    )

    while head < num_samples:
        sample_subset = queries[head : min(head + max_batch, num_samples), :].to(device)

        queries[head : min(head + max_batch, num_samples), -1] = (
            DeepSDFStruct.deep_sdf.utils.decode_sdf(decoder, None, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch
    sample_time = time.time()
    print("sampling takes: %f" % (sample_time - start))
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N[0], N[1], N[2])
    sdf_values = queries[:, -1].data.cpu().numpy()
    samples_orig = samples_orig.cpu().numpy()
    for loc, cap_dict in cap_border_dict.items():
        cap, measure = cap_dict["cap"], cap_dict["measure"]
        dim, multiplier = location_lookup[loc]
        border_sdf = (samples_orig[:, dim] - multiplier * (1 - measure)) * -multiplier
        if cap == -1:
            sdf_values = np.maximum(sdf_values, -border_sdf)
        elif cap == 1:
            sdf_values = np.minimum(sdf_values, border_sdf)
        else:
            raise ValueError("Cap must be -1 or 1")
    end = time.time()

    # cap everything outside the unit cube

    for dim, measure in zip([0, 0, 1, 1, 2, 2], [-1, 1, -1, 1, -1, 1]):
        border_sdf = (samples_orig[:, dim] - measure) * -measure
        sdf_values = np.maximum(sdf_values, -border_sdf)

    sdf_values = sdf_values.reshape(N[0], N[1], N[2])
    sdf_values = torch.tensor(sdf_values).to(device)

    if save_ply_file:
        if filename is None:
            raise RuntimeError("No filename given")
        convert_sdf_samples_to_ply(
            sdf_values.cpu(), voxel_origin, voxel_size, filename + ".ply", offset, scale
        )
    else:
        if use_flexicubes:
            # DeepSDFStruct.flexicubes has the possibility to output tetmesh, but it's extremely slow
            # and often fails
            recon_from_latent = lambda l: reconstructor(
                voxelgrid_vertices=torch.tensor(samples_orig[:, :3]).to(device),
                scalar_field=l,
                cube_idx=cube_idx,
                resolution=tuple(N - 1),
                output_tetmesh=output_tetmesh,
            )
            verts, faces, loss = recon_from_latent(sdf_values.view(-1))
            if compute_derivatives:
                # this is wrong: this computes the jacobian with respect to the sdf values, not the latent vector
                # todo: refactor this whole function and compute the jacobian with respect to the latent vector
                jac = torch.autograd.functional.jacobian(
                    lambda l: recon_from_latent(l)[0],
                    sdf_values.view(-1),
                    strict=False,
                    vectorize=True,
                )
                tot_jac = []
                basis_eval = latent_vec_interpolation.basis(
                    np.clip(samples_orig[:, :3], -1, 1)
                )
                tot_jac = np.matmul(jac.detach().cpu().numpy(), basis_eval)
            verts = (verts + 1) / 2

            return verts, faces, tot_jac
        else:
            if not isinstance(voxel_size, list):
                voxel_size = [voxel_size] * 3
            verts, faces, normals, values = skimage.measure.marching_cubes(
                sdf_values.cpu().numpy(), level=0.0, spacing=voxel_size
            )
            # sci-kit measure assumes origin at (0,0,0)
            # input for SDF is -1-voxel_size_{x,y,z} to 1+voxel_size_{x,y,z}
            # scale factor 2 to get to 0 to 1
            verts = (verts - voxel_size) / 2
        return verts, faces


def create_mesh_microstructure_diff(
    tiling,
    decoder,
    latent_vec_interpolation,
    N=256,
    max_batch=32**3,
    offset=None,
    scale=None,
    cap_border_dict=CapBorderDict,
    device=None,
    output_tetmesh=False,
    compute_derivatives=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(tiling, list):
        if len(tiling) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        tiling = np.array(tiling)
    elif isinstance(tiling, int):
        tiling = np.array([tiling, tiling, tiling])
    else:
        raise ValueError("Tiling must be a list or an integer")

    # add 1 on each side to slightly include the border
    if isinstance(N, list):
        if len(N) != 3:
            raise ValueError("Number of grid points must be a list of 3 integers")
        N = np.array(N)
    elif isinstance(N, int):
        N = np.array([N, N, N])
    else:
        raise ValueError("Number of grid points must be a list or an integer")

    decoder.eval()
    flexi_cubes_constructor = DeepSDFStruct.flexicubes(device=device)
    samples, samples_orig, lat_vec_red, cube_idx = prepare_samples(
        flexi_cubes_constructor, device, N, tiling, latent_vec_interpolation
    )
    logger.debug(f"Querying {np.prod(N)} DeepSDF points")
    tstart = time.time()

    verts, faces = evaluate_network(
        lat_vec_red,
        samples,
        samples_orig,
        decoder,
        N,
        cap_border_dict,
        cube_idx,
        output_tetmesh,
        flexi_cubes_constructor,
    )
    tot_jac = []
    DeepSDFStruct.deep_sdf.utils.log_memory_usage()
    save_memory = True
    logger.debug("Computing DeepSDF derivatives")
    if compute_derivatives:
        # shape of SDF jacobian = dVerts/dLatent: (n_verts, dim_phys, N_eval_points, dim_latent)
        # shape of Spline Jacobian = dLatent/dControl: (N_eval_points, n_control_points)
        # shape of Total Jacobian = dVerts/dControl: (n_verts, dim_phys, n_control_points, dim_latent)

        if save_memory:
            # slow version
            # for i in range(jac.shape[3]):
            #     tot_jac.append(np.matmul(jac.detach().cpu().numpy()[:,:,:,i], basis_eval))
            # tot_jac = np.dstack(tot_jac)

            # small loop to loop over latent dimensions
            physical_dim = samples.shape[1] - 1
            latent_dim = lat_vec_red.shape[1]
            queries = np.clip(samples_orig[:, :3].detach().cpu().numpy(), -1, 1)
            n_control_points = latent_vec_interpolation.control_points.shape[0]
            basis, supports = latent_vec_interpolation.basis_and_support(queries)
            basis_eval = sp.utils.data.make_matrix(
                basis, supports, n_control_points, as_array=True
            )
            basis_eval_torch = torch.from_numpy(basis_eval).to(
                device, dtype=torch.float32
            )

            tot_jac = []
            for i_lat in range(latent_dim):
                cpt_jac = []
                for i_cpt in range(n_control_points):
                    # dLatent / dControl is the same for each latent dimension
                    dLatent_dControl = basis_eval_torch[:, i_cpt]

                    # function = lambda l: evaluate_network(l.reshape(-1,latent_dim))[0].reshape(-1)
                    def verts_from_latent(single_latent_entry):
                        modified_latent_vector = lat_vec_red
                        modified_latent_vector[:, i_lat] = single_latent_entry
                        return evaluate_network(
                            modified_latent_vector,
                            samples,
                            samples_orig,
                            decoder,
                            N,
                            cap_border_dict,
                            cube_idx,
                            output_tetmesh,
                            flexi_cubes_constructor,
                        )[0]

                    # shape of jacobian: [n_verts*physical_dim, n_samples*latent_dim]
                    # shape of dLatent_dControl: [n_samples*latent_dim]
                    # the commented code should output the same
                    # dVerts_dLatent = torch.autograd.functional.jacobian(function, lat_vec_red.reshape(-1))
                    # dVerts_dControl_matmul = torch.matmul(dVerts_dLatent, dLatent_dControl)
                    _, dVerts_dControl_i = torch.autograd.functional.jvp(
                        verts_from_latent, lat_vec_red[:, i_lat], v=dLatent_dControl
                    )
                    cpt_jac.append(dVerts_dControl_i.reshape(-1, physical_dim))
                tot_jac.append(cpt_jac)
                #                 # dLatent / dControl is the same for each latent dimension
                # dLatent_dControl = basis_eval_torch[:,:, i_cpt].reshape(-1)
                # function = lambda l: evaluate_network(l.reshape(-1,latent_dim))[0].reshape(-1)
                # # shape of jacobian: [n_verts*physical_dim, n_samples*latent_dim]
                # # shape of dLatent_dControl: [n_samples*latent_dim]
                # # the commented code should output the same
                # # dVerts_dLatent = torch.autograd.functional.jacobian(function, lat_vec_red.reshape(-1))
                # # dVerts_dControl_matmul = torch.matmul(dVerts_dLatent, dLatent_dControl)
                # _, dVerts_dControl_i = torch.autograd.functional.jvp(function, lat_vec_red.reshape(-1), v=dLatent_dControl)
                # tot_jac.append(dVerts_dControl_i.reshape(-1, physical_dim))
            torch.cuda.empty_cache()
            # DeepSDFStruct.deep_sdf.utils.log_memory_usage()
            tot_jac = torch.stack(
                [torch.stack(inner_jac, dim=2) for inner_jac in tot_jac], dim=3
            )
            tot_jac_cpu = tot_jac.detach().cpu().numpy()
        else:
            jac = torch.autograd.functional.jacobian(
                lambda l: evaluate_network(l)[0], lat_vec_red
            )
            torch.cuda.empty_cache()
            # DeepSDFStruct.deep_sdf.utils.log_memory_usage()
            basis_eval = latent_vec_interpolation.basis(
                np.clip(samples_orig[:, :3].detach().cpu().numpy(), -1, 1)
            )
            # slow version
            # for i in range(jac.shape[3]):
            #     tot_jac.append(np.matmul(jac.detach().cpu().numpy()[:,:,:,i], basis_eval))
            # tot_jac = np.dstack(tot_jac)
            jac_cpu = jac.detach().cpu().numpy()
            tot_jac_cpu = np.einsum("ijkl,km->ijml", jac_cpu, basis_eval)
    t_finish = time.time() - tstart
    logger.log(
        logging.DEBUG,
        f"Time for querying {np.prod(N)} points and computing derivatives: {t_finish}",
    )
    verts = (verts + 1) / 2
    max_memory = torch.cuda.max_memory_allocated(device=device)
    maxmem_human = DeepSDFStruct.deep_sdf.utils.format_memory_size(max_memory)
    logger.debug(f"Maximum memory used by {device}: {maxmem_human}")
    return verts, faces, tot_jac_cpu


def prepare_samples(
    flexi_cubes_constructor, device, N, tiling, latent_vec_interpolation
):
    # prepare samples

    samples_orig, cube_idx = flexi_cubes_constructor.construct_voxel_grid(
        resolution=tuple(N)
    )
    samples_orig = samples_orig.to(device)
    cube_idx = cube_idx.to(device)
    # transform samples from [-0.5, 0.5] to [-1.05, 1.05]
    samples_orig = samples_orig * 2.1
    N_tot = samples_orig.shape[0]
    N = N + 1

    tx, ty, tz = tiling

    def transform(x, t):
        # transform x from [-1,1] to [0,1]
        x = (x + 1) / 2
        return 4 * torch.abs(t * x / 2 - torch.floor((t * x + 1) / 2)) - 1

    samples = torch.zeros(N_tot, 4)
    samples[:, 0] = transform(samples_orig[:, 0], tx)
    samples[:, 1] = transform(samples_orig[:, 1], ty)
    samples[:, 2] = transform(samples_orig[:, 2], tz)

    samples.requires_grad = False

    inside_domain = torch.where(
        (samples_orig[:, 0] >= -1)
        & (samples_orig[:, 0] <= 1)
        & (samples_orig[:, 1] >= -1)
        & (samples_orig[:, 1] <= 1)
        & (samples_orig[:, 2] >= -1)
        & (samples_orig[:, 2] <= 1)
    )
    lat_vec_red = torch.zeros(
        (samples_orig.shape[0], latent_vec_interpolation.control_points[0].shape[0]),
        dtype=torch.float32,
    )
    lat_vec_red[inside_domain] = torch.tensor(
        latent_vec_interpolation.evaluate(
            samples_orig[:, 0:3][inside_domain].cpu().numpy()
        ),
        dtype=torch.float32,
    )
    lat_vec_red = lat_vec_red.to(device)
    samples = samples.to(device)
    return samples, samples_orig, lat_vec_red, cube_idx


def evaluate_network(
    lat_vec_red,
    samples,
    samples_orig,
    decoder,
    N,
    cap_border_dict,
    cube_idx,
    output_tetmesh,
    flexicubes_reconstructor,
):
    queries = torch.hstack([lat_vec_red, samples[:, 0:3]])
    start_time = time.time()
    sdf_values = DeepSDFStruct.deep_sdf.utils.decode_sdf(
        decoder, None, queries
    ).squeeze(1)
    sample_time = time.time()
    # logger.debug("sampling takes: %f" % (sample_time - start_time))
    for loc, cap_dict in cap_border_dict.items():
        cap, measure = cap_dict["cap"], cap_dict["measure"]
        dim, multiplier = location_lookup[loc]
        border_sdf = (samples_orig[:, dim] - multiplier * (1 - measure)) * -multiplier
        if cap == -1:
            sdf_values = torch.maximum(sdf_values, -border_sdf)
        elif cap == 1:
            sdf_values = torch.minimum(sdf_values, border_sdf)
        else:
            raise ValueError("Cap must be -1 or 1")

    # cap everything outside the unit cube

    for dim, measure in zip([0, 0, 1, 1, 2, 2], [-1, 1, -1, 1, -1, 1]):
        border_sdf = (samples_orig[:, dim] - measure) * -measure
        sdf_values = torch.maximum(sdf_values, -border_sdf)

    # sdf_values = sdf_values.reshape(N[0]+1, N[1]+1, N[2]+1)
    # sdf_values = torch.tensor(sdf_values).to(device)

    # DeepSDFStruct.flexicubes has the possibility to output tetmesh, but it's extremely slow
    # and often fails
    verts, faces, loss = DeepSDFStruct.flexicubes_reconstructor(
        voxelgrid_vertices=samples_orig[:, :3],
        scalar_field=sdf_values,
        cube_idx=cube_idx,
        resolution=tuple(N),
        output_tetmesh=output_tetmesh,
    )
    return verts, faces


def create_mesh_from_latent(experiment_directory, epoch, index, **kwargs):
    decoder = DeepSDFStruct.deep_sdf.ws.load_trained_model(
        experiment_directory, str(epoch)
    )
    latent_vecs = DeepSDFStruct.deep_sdf.ws.load_latent_vectors(
        experiment_directory, str(epoch)
    )
    class_name = "all"
    instance_name = f"{index}"
    dataset = "latent_recon"
    fname = DeepSDFStruct.deep_sdf.ws.get_reconstructed_mesh_filename(
        experiment_directory, epoch, dataset, class_name, instance_name
    )
    fname = pathlib.Path(fname)
    if not os.path.isdir(fname.parent):
        os.makedirs(fname.parent)
    create_mesh(decoder, latent_vecs[index], str(fname), **kwargs)
    return fname


def constantLatvec(value):
    return sp.BSpline([0, 0, 0], [[-1, 1], [-1, 1], [-1, 1]], [value])

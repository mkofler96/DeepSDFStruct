"""
FlexiSquares Implementation
===========================

This module contains the core implementation of the FlexiSquares algorithm,
a differentiable 2D mesh extraction method adapted from FlexiCubes.

FlexiSquares improves upon traditional Marching Squares by allowing gradient-
based optimization of the extracted contours, making it suitable for inverse
design and shape optimization problems.

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
Licensed under the Apache License, Version 2.0.
"""

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import logging


import DeepSDFStruct
from DeepSDFStruct.flexisquares.tables import num_vd_table, dmc_table, tet_table

logger = logging.getLogger(DeepSDFStruct.__name__)

__all__ = ["FlexiSquares"]


class FlexiSquares:
    """
    Implements the 2D variant of the flexicubes method, adapted as
    flexisquares, for extracting contour meshes from scalar fields.
    This class uses lookup tables and indexed operations to perform
    differentiable isocontour extraction from signed distance fields
    defined on 2D grids.

    Flexisquares is a differentiable variant of the Dual Marching
    Squares (DMS) algorithm. It improves geometric fidelity by
    optimizing surface representations using gradient-based methods.

    During initialization, the class loads precomputed lookup tables
    and converts them into PyTorch tensors on the specified device.

    Attributes:
        device (str): Computational device, usually "cuda" or "cpu".
        dmc_table (torch.Tensor): Dual Marching Squares (DMS) table
            encoding the edges associated with each dual vertex in 16
            possible Marching Squares configurations.
        num_vd_table (torch.Tensor): Table holding the number of dual
            vertices for each configuration.
        tet_table (torch.Tensor): Lookup table used during triangle
            generation inside the contour.
        cube_corners (torch.Tensor): Positions of the four corners of
            a unit square in 2D space.
        cube_corners_idx (torch.Tensor): Corner indices encoded as
            binary powers to compute case IDs for Marching Squares.
        cube_edges (torch.Tensor): Edge connections in a square,
            listed in pairs of corner indices.
    """

    def __init__(self, device="cuda", qef_reg_scale=1e-3, weight_scale=0.99):

        self.device = device
        self.dmc_table = torch.tensor(
            dmc_table, dtype=torch.long, device=device, requires_grad=False
        )
        self.num_vd_table = torch.tensor(
            num_vd_table, dtype=torch.long, device=device, requires_grad=False
        )

        self.tet_table = torch.tensor(
            tet_table, dtype=torch.long, device=device, requires_grad=False
        )
        self.quad_split_1 = torch.tensor(
            [0, 1, 2, 0, 2, 3], dtype=torch.long, device=device, requires_grad=False
        )
        self.quad_split_2 = torch.tensor(
            [0, 1, 3, 3, 1, 2], dtype=torch.long, device=device, requires_grad=False
        )
        self.quad_split_train = torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 0],
            dtype=torch.long,
            device=device,
            requires_grad=False,
        )

        self.cube_corners = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [1, 1]], device=device
        )
        # Each corner corresponds to a binary bit (for the 2^4 possible inside/outside cases):
        self.cube_corners_idx = torch.pow(2, torch.arange(4, requires_grad=False))

        # (0,1) ─ 2 ─ (1,1)
        # |         |
        # 0         1
        # |         |
        # (0,0) ─ 3 ─ (1,0)

        # (2) ─ 2 ─ (3)
        #  |         |
        #  0         1
        #  |         |
        # (0) ─ 3 ─ (1)

        # Edges are:
        # e0: bottom → top along left side → (0,0)-(0,1) → corners (0,2)
        # e1: bottom → top along right side → (1,0)-(1,1) → corners (1,3)
        # e2: left → right along bottom → (0,0)-(1,0) → corners (0,1)
        # e3: left → right along top → (0,1)-(1,1) → corners (2,3)

        self.cube_edges = torch.tensor(
            [0, 2, 1, 3, 2, 3, 0, 1], dtype=torch.long, device=device
        )
        # should be equivalently, reshaped as pairs:
        # self.cube_edges = torch.tensor(
        #     [[0, 2], [1, 3], [0, 1], [2, 3]], dtype=torch.long, device=device
        # )

        self.edge_dir_table = torch.tensor(
            [1, 1, 0, 0], dtype=torch.long, device=device
        )

        self.dir_faces_table = torch.tensor(
            [[[0, 1], [2, 3]], [[0, 2], [1, 3]]],  # x-dir edges  # y-dir edges
            dtype=torch.long,
            device=device,
        )
        # not sure which version is correct
        self.adj_pairs_2233 = torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=device
        )
        self.adj_pairs_3322 = torch.tensor(
            [0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.long, device=device
        )
        self.qef_reg_scale = qef_reg_scale
        self.weight_scale = weight_scale

    def construct_voxel_grid(
        self, resolution, bounds=None
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Generates a 2D grid of vertices and their square indices given
        the specified resolution.

        Args:
            resolution (int or list[int]): Resolution of the 2D grid.
                If an integer is provided, it is applied to both x and
                y dimensions. If a tuple/list of two integers is given,
                they specify (x_res, y_res).
            bounds (torch.Tensor, optional): 2×2 tensor defining the
                [min, max] bounds in x and y.
                Defaults to [[-0.05, -0.05], [1.05, 1.05]].

        Returns:
            (torch.Tensor, torch.Tensor): Tuple containing:
                - Vertices (N×2): Scaled vertex coordinates of the
                  2D grid.
                - Squares (F×4): Indices into `vertices` defining each
                  square's corner connectivity.
        """
        res = resolution
        base_cube_f = torch.arange(4).to(self.device)
        if isinstance(res, int):
            res = (res, res)
        voxel_grid_template = torch.ones(res, device=self.device)

        res = torch.tensor([res], device=self.device)
        coords = (
            torch.nonzero(voxel_grid_template).to(torch.get_default_dtype()) / res
        )  # N, 2
        verts = (self.cube_corners.unsqueeze(0) / res + coords.unsqueeze(1)).reshape(
            -1, 2
        )
        cubes = (
            base_cube_f.unsqueeze(0)
            + torch.arange(coords.shape[0], device=self.device).unsqueeze(1) * 4
        ).reshape(-1)

        verts_rounded = torch.round(verts * 10**5) / (10**5)
        verts_unique, inverse_indices = torch.unique(
            verts_rounded, dim=0, return_inverse=True
        )
        cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 4)

        if bounds is None:
            bounds = torch.tensor(
                [[-0.05, -0.05], [1.05, 1.05]],
                device=verts_unique.device,
                dtype=verts_unique.dtype,
            )
        else:
            bounds = torch.as_tensor(
                bounds, device=verts_unique.device, dtype=verts_unique.dtype
            )
            assert bounds.shape == (2, 2), "bounds must have shape [2, 2]"

        # Scale samples from [0, 1] to the given bounds
        # samples = samples * 1.1 + _torch.tensor([0.5, 0.5, 0.5], device=device)
        samples = verts_unique - 0.5
        verts_scaled = (
            bounds[0]
            + 0.5 * (bounds[1] - bounds[0])
            + (bounds[1] - bounds[0]) * samples
        )

        tolerance = 1e-6
        torch._assert(
            torch.all(
                verts_scaled.ge(bounds[0] - tolerance)
                & verts_scaled.le(bounds[1] + tolerance)
            ),
            "Samples are out of specified bounds",
        )
        return verts_scaled, cubes

    # changes mkofler: adapted function signature to match the kaolin package
    # function signature on kaolin package:
    # verts, faces, loss = DeepSDFStruct.flexicubes_reconstructor(voxelgrid_vertices=samples_orig[:, :3],
    # scalar_field=sdf_values,
    # cube_idx=cube_idx,
    # resolution=tuple(N),
    # output_tetmesh=output_tetmesh)
    def __call__(
        self,
        voxelgrid_vertices,
        scalar_field,
        cube_idx,
        beta_fx4=None,
        alpha_fx4=None,
        output_tetmesh=False,
        resolution=None,
        n_smoothing_iterations=5,
    ):
        r"""
        Extracts a differentiable 2D contour mesh from a scalar field using DeepSDFStruct.flexisquares.

        Converts discrete signed distance fields on a 2D grid into polygonal contours using a
        differentiable Dual Marching Squares process, as described in
        *Flexible Isosurface Extraction for Gradient-Based Mesh Optimization* (adapted to 2D).

        Args:
            voxelgrid_vertices (torch.Tensor): 2D coordinates of grid vertices (N×2).
            scalar_field (torch.Tensor): Signed distance field values at each vertex.
                Negative values are inside the contour.
            cube_idx (torch.LongTensor): Indices of 4 vertices per grid cell (F×4).
            resolution (int or tuple[int, int]): Resolution of the grid.
            beta_fx4 (torch.Tensor, optional): Edge weights adjusting dual vertex positioning.
            alpha_fx4 (torch.Tensor, optional): Corner weights adjusting interpolation.
            output_tetmesh (bool, optional): Retained for compatibility; in 2D, returns triangles only.


        Returns:
            (torch.Tensor, torch.LongTensor, torch.Tensor): Tuple of:
                - Vertices (V×2): Coordinates of contour vertices.
                - Faces (F×3): Triangle indices of the contour mesh.
                - Regularization loss (L_dev): Deviation loss per dual vertex.

        .. _Flexible Isosurface Extraction for Gradient-Based Mesh Optimization:
            https://research.nvidia.com/labs/toronto-ai/DeepSDFStruct.flexicubes/
        .. _Manifold Dual Contouring:
            https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf
        """
        x_nx2 = voxelgrid_vertices
        s_n = scalar_field
        cube_fx4 = cube_idx
        surf_cubes, occ_fx4 = self._identify_surf_cubes(s_n, cube_fx4)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                (
                    torch.zeros((0, 4), dtype=torch.long, device=self.device)
                    if output_tetmesh
                    else torch.zeros((0, 3), dtype=torch.long, device=self.device)
                ),
                torch.zeros((0), device=self.device),
            )
        beta_fx4, alpha_fx4 = self._normalize_weights(beta_fx4, alpha_fx4, surf_cubes)

        case_ids = self._get_case_id(occ_fx4, surf_cubes)

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            s_n, cube_fx4, surf_cubes
        )

        vd, L_dev, vd_idx_map = self._compute_vd(
            x_nx2, surf_edges, s_n, case_ids, beta_fx4, alpha_fx4, idx_map
        )
        boundary_vertices, boundary_lines, s_edges, edge_indices = (
            self._extract_zero_contour(
                s_n, surf_edges, vd, edge_counts, idx_map, vd_idx_map, surf_edges_mask
            )
        )
        if not output_tetmesh:
            return boundary_vertices, boundary_lines, L_dev
        else:
            self.check_open_mesh(boundary_lines)
            all_vertices, triangles = self._triangulate(
                x_nx2,
                s_n,
                cube_fx4,
                boundary_vertices,
                boundary_lines,
                surf_edges,
                s_edges,
                case_ids,
                edge_indices,
                surf_cubes,
            )
            vertices_smoothed = self._apply_laplacian_smoothing(
                all_vertices,
                triangles,
                boundary_lines,
                iterations=n_smoothing_iterations,
            )
            return vertices_smoothed, triangles, L_dev

    def _apply_laplacian_smoothing(
        self, vertices, triangles, boundary_vertices, iterations=5, lamb=0.5
    ):
        """
        Vectorized Laplacian smoothing using PyTorch sparse matrices.
        """
        logger.info(f"applying {iterations} iterations of laplacian smoothing")
        num_total = vertices.shape[0]
        device = vertices.device

        unique_boundary = torch.unique(boundary_vertices.long())
        is_interior = torch.ones(num_total, 1, device=device, dtype=torch.bool)
        is_interior[unique_boundary] = False

        i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]

        edges = torch.cat(
            [
                torch.stack([i, j], dim=1),
                torch.stack([j, i], dim=1),
                torch.stack([j, k], dim=1),
                torch.stack([k, j], dim=1),
                torch.stack([k, i], dim=1),
                torch.stack([i, k], dim=1),
            ],
            dim=0,
        )  # shape: (num_edges, 2)

        # Remove duplicate edges
        edges = torch.unique(edges, dim=0)

        src = edges[:, 0]  # neighbor index
        dst = edges[:, 1]  # vertex to accumulate into

        degree = torch.zeros(num_total, device=device, dtype=vertices.dtype)
        one = torch.ones_like(dst, dtype=vertices.dtype)
        degree.scatter_add_(0, dst, one)
        degree_inv = 1.0 / torch.clamp(degree, min=1.0)

        v_smoothed = vertices.clone()

        for _ in range(iterations):
            neighbor_sum = torch.zeros_like(v_smoothed)  # (N, 3)

            neighbor_sum.scatter_add_(
                0, dst.unsqueeze(1).expand(-1, v_smoothed.shape[1]), v_smoothed[src]
            )

            neighbor_avg = neighbor_sum * degree_inv.unsqueeze(1)

            v_update = (1 - lamb) * v_smoothed + lamb * neighbor_avg
            v_smoothed = torch.where(is_interior, v_update, v_smoothed)

        return v_smoothed

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """
        dist = torch.norm(
            ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1
        )
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(
            0, edge_group_to_vd, dist
        ) / vd_num_edges.squeeze(1).to(torch.get_default_dtype())
        mad = (
            dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)
        ).abs()
        return mad

    def _normalize_weights(self, beta_fx4, alpha_fx4, surf_cubes):
        """
        Normalizes the given weights to be non-negative. If input weights are None, it creates and returns a set of weights of ones.
        """
        n_cubes = surf_cubes.shape[0]

        if beta_fx4 is not None:
            beta_fx4 = torch.tanh(beta_fx4) * self.weight_scale + 1
        else:
            beta_fx4 = torch.ones((n_cubes, 4), device=self.device)

        if alpha_fx4 is not None:
            alpha_fx4 = torch.tanh(alpha_fx4) * self.weight_scale + 1
        else:
            alpha_fx4 = torch.ones((n_cubes, 4), device=self.device)

        return beta_fx4[surf_cubes], alpha_fx4[surf_cubes]

    @torch.no_grad()
    def _get_case_id(self, occ_fx4, surf_cubes):
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the
        supplementary material. It should be noted that this function assumes a regular grid.
        """
        case_ids = (
            occ_fx4[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)
        ).sum(-1)

        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, s_n, cube_fx4, surf_cubes):
        """
        Finds edges of active squares that intersect the contour by checking for opposite signs
        of the scalar field at edge endpoints.
        Assigns unique indices to surface-intersecting edges and produces a mapping
        from square edges to unique edge IDs.
        """
        occ_n = s_n < 0
        all_edges = cube_fx4[surf_cubes][:, self.cube_edges].reshape(-1, 2)

        unique_edges, _idx_map, counts = torch.unique(
            all_edges, dim=0, return_inverse=True, return_counts=True
        )

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = (
            torch.ones(
                (unique_edges.shape[0]), dtype=torch.long, device=cube_fx4.device
            )
            * -1
        )
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_fx4.device)
        # Shaped as [number of cubes x 4 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, s_n, cube_fx4):
        """
        Identifies grid cubes that intersect with the underlying surface by checking if the signs at
        all corners are not identical.
        """
        occ_n = s_n < 0
        occ_fx4 = occ_n[cube_fx4.reshape(-1)].reshape(-1, 4)
        _occ_sum = torch.sum(occ_fx4, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 4)
        return surf_cubes, occ_fx4

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = torch.cat(
            [
                torch.index_select(
                    input=edges_weight,
                    index=torch.tensor([1], device=self.device),
                    dim=edge_dim,
                ),
                -torch.index_select(
                    input=edges_weight,
                    index=torch.tensor([0], device=self.device),
                    dim=edge_dim,
                ),
            ],
            edge_dim,
        )
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _compute_vd(
        self, x_nx2, surf_edges, s_n, case_ids, beta_fx4, alpha_fx4, idx_map
    ):
        """
        Computes the location of dual vertices as described in Section 4.2
        """
        alpha_nx4x2 = torch.index_select(
            input=alpha_fx4, index=self.cube_edges, dim=1
        ).reshape(-1, 4, 2)
        surf_edges_x = torch.index_select(
            input=x_nx2, index=surf_edges.reshape(-1), dim=0
        ).reshape(-1, 2, 2)
        surf_edges_s = torch.index_select(
            input=s_n, index=surf_edges.reshape(-1), dim=0
        ).reshape(-1, 2, 1)
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)

        idx_map = idx_map.reshape(-1, 4)

        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges = (
            [],
            [],
            [],
            [],
        )

        num_vd = self.num_vd_table[case_ids]
        # This is the "Source of Truth" for ordering
        v_offsets = torch.cumsum(num_vd, dim=0) - num_vd
        total_num_vd = num_vd.sum().item()

        # total_num_vd = 0
        vd = torch.zeros((total_num_vd, 2), device=self.device)
        beta_sum = torch.zeros((total_num_vd, 1), device=self.device)
        vd_num_edges = torch.zeros((total_num_vd, 1), device=self.device)

        vd_idx_map = torch.zeros(
            (case_ids.shape[0], 4),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        all_edge_groups = []
        all_edge_to_vd = []
        all_edge_to_cube = []
        all_vd_num_edges = []
        for num in torch.unique(num_vd):
            cur_cubes_mask = num_vd == num
            num_selected = cur_cubes_mask.sum()
            batch_v_starts = v_offsets[cur_cubes_mask]
            # curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes_mask], :num].reshape(
                num_selected, -1
            )
            # Create global vertex IDs for this batch
            # local_ids: [0, 1, 0, 1...] for num=2
            local_ids = torch.arange(num, device=self.device).repeat(num_selected)
            # global_ids: [start0, start0, start1, start1...]
            global_starts = batch_v_starts.repeat_interleave(num)

            # This aligns the dual vertex ID with your triangulation offsets
            curr_edge_group_to_vd = (
                (global_starts + local_ids).reshape(-1, 1).repeat(1, 2).reshape(-1)
            )

            # Mapping cubes to their global indices
            curr_edge_group_to_cube = torch.arange(
                case_ids.shape[0], device=self.device
            )[cur_cubes_mask].repeat_interleave(num * 2)

            curr_mask = curr_edge_group.reshape(-1) != -1
            curr_vd_edge_counts = curr_mask.reshape(-1, 2).sum(-1, keepdims=True)
            all_vd_num_edges.append(curr_vd_edge_counts)
            all_edge_groups.append(curr_edge_group.reshape(-1)[curr_mask])
            all_edge_to_vd.append(curr_edge_group_to_vd[curr_mask])
            all_edge_to_cube.append(curr_edge_group_to_cube[curr_mask])

        edge_group = torch.cat(all_edge_groups)
        edge_group_to_vd = torch.cat(all_edge_to_vd)
        edge_group_to_cube = torch.cat(all_edge_to_cube)
        vd_num_edges = torch.cat(all_vd_num_edges)

        vd = torch.zeros((total_num_vd, 2), device=self.device)
        beta_sum = torch.zeros((total_num_vd, 1), device=self.device)

        idx_group = torch.gather(
            input=idx_map.reshape(-1), dim=0, index=edge_group_to_cube * 4 + edge_group
        )

        # remove the idx_group>0, should not be needed
        x_group = torch.index_select(
            input=surf_edges_x, index=idx_group.reshape(-1), dim=0
        ).reshape(-1, 2, 2)
        s_group = torch.index_select(
            input=surf_edges_s, index=idx_group.reshape(-1), dim=0
        ).reshape(-1, 2, 1)

        zero_crossing_group = torch.index_select(
            input=zero_crossing, index=idx_group.reshape(-1), dim=0
        ).reshape(-1, 2)

        alpha_group = torch.index_select(
            input=alpha_nx4x2.reshape(-1, 2),
            dim=0,
            index=edge_group_to_cube * 4 + edge_group,
        ).reshape(-1, 2, 1)
        ue_group = self._linear_interp(s_group * alpha_group, x_group)

        beta_group = torch.gather(
            input=beta_fx4.reshape(-1), dim=0, index=edge_group_to_cube * 4 + edge_group
        ).reshape(-1, 1)
        beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
        vd = (
            vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group)
            / beta_sum
        )
        L_dev = self._compute_reg_loss(
            vd, zero_crossing_group, edge_group_to_vd, vd_num_edges
        )

        # v_idx = torch.arange(vd.shape[0], device=self.device)  # + total_num_vd

        vd_idx_map = vd_idx_map.reshape(-1).scatter(
            dim=0, index=edge_group_to_cube * 4 + edge_group, src=edge_group_to_vd
        )
        # for vert in vd:
        #     ax.scatter(vert[0], vert[1], marker="x", color="lightblue")

        return vd, L_dev, vd_idx_map

    def _extract_zero_contour(
        self, s_n, surf_edges, vd, edge_counts, idx_map, vd_idx_map, surf_edges_mask
    ):
        """
        Connects two adjacent dual vertices around an active square to form a contour.
        """

        # for e_idx, e in enumerate(surf_edges):
        #     v0, v1 = x_nx2[e]
        #     color = "red" if edge_counts[surf_edges_mask][e_idx] == 1 else "green"
        #     mid = (v0 + v1) / 2
        #     ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color=color, linewidth=1.5)
        #     # ax.text(
        #     #     mid[0], mid[1], str(edge_counts[e_idx].item()), color=color, fontsize=7
        #     # )
        # fig.savefig("sdf.png", dpi=1000, bbox_inches="tight")
        with torch.no_grad():
            group_mask = (
                edge_counts == 2
            ) & surf_edges_mask  # surface edges shared by 2 cubes.
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 2)

            # Ensure all face directions point towards the positive SDF to maintain consistent winding.
            s_edges = s_n[
                surf_edges[edge_indices.reshape(-1, 2)[:, 0]].reshape(-1)
            ].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx[flip_mask] = quad_vd_idx[flip_mask][:, [0, 1]]
            quad_vd_idx[~flip_mask] = quad_vd_idx[~flip_mask][:, [1, 0]]

        return vd, quad_vd_idx, s_edges, edge_indices

    def _triangulate(
        self,
        x_nx2,
        s_n,
        square_fx4,
        surf_vertices,
        edges,
        surf_edges_global,
        s_edges,
        case_ids,
        edge_indices,
        surf_cubes,
    ):
        """
        Triangulates the interior surface to produce a triangular mesh, adopted from 3D as described in Section 4.5.
        """
        # occupancy field per query point
        occ_n = s_n < 0
        # occupancy field for each square
        occ_fx4 = occ_n[square_fx4.reshape(-1)].reshape(-1, 4)
        # sum of occupancy
        #   == 1: 1 corner inside
        #   == 2: 2 corners inside
        #   == 3: 3 corners inside
        #   == 4: 4 corners inside
        occ_sum = torch.sum(occ_fx4, -1)
        # get number of vd, most have 1, but the edge cases have 2
        # the fully filled and completely empty squares have none
        vd_counts = self.num_vd_table[case_ids]
        v_offsets = torch.cumsum(vd_counts, dim=0) - vd_counts
        tris_list = []
        """
        The first step is to connect all surface edges to the interior vertices
        """

        inside_verts = x_nx2[occ_n.reshape(-1)]
        mapping_inside_verts = (
            torch.ones((occ_n.shape[0]), dtype=torch.long, device=self.device) * -1
        )
        mapping_inside_verts[occ_n.reshape(-1)] = (
            torch.arange(occ_n.sum(), device=self.device) + surf_vertices.shape[0]
        )

        s_edges = s_n[surf_edges_global].reshape(-1, 2)  # signed distance at each end
        inside_mask = s_edges < 0
        inside_verts_idx = mapping_inside_verts[surf_edges_global[inside_mask]]

        tris_surface = torch.cat([edges, inside_verts_idx.unsqueeze(-1)], -1)
        tris_list.append(tris_surface)
        vertices = torch.cat([surf_vertices, inside_verts])
        # plot_triangles(
        #     "tris_inside_after_surf.png",
        #     triangles=tris_surface,
        #     vertices=vertices,
        #     x_nx2=x_nx2,
        #     s_n=s_n,
        # )

        # =====================================================================
        #
        # The next step is to find all quads that are fully inside (occsum==4)
        # and connect the center to the edges
        #
        # =====================================================================

        quads_occ4_center = (
            x_nx2[square_fx4[occ_sum == 4].reshape(-1)].reshape(-1, 4, 2).mean(1)
        )

        quad_occ4_center_x = x_nx2[square_fx4[occ_sum == 4]].mean(1)
        center_idx = (
            torch.arange(quad_occ4_center_x.shape[0], device=self.device)
            + surf_vertices.shape[0]
            + inside_verts.shape[0]
        )
        index_occ4 = torch.tensor([[0, 1], [1, 3], [3, 2], [2, 0]], device=self.device)
        quads_occ4 = square_fx4[occ_sum == 4]
        centers_occ4_x4 = center_idx.unsqueeze(1).expand(-1, 4).reshape(-1, 1)

        outside_edges_occ4 = mapping_inside_verts[quads_occ4[:, index_occ4]]
        # and connect it to the center node:
        # each triangle = [center, corner1, corner2]
        tris_inside_occ4 = torch.cat(
            [centers_occ4_x4, outside_edges_occ4.reshape(-1, 2)], dim=1
        )
        vertices = torch.cat([surf_vertices, inside_verts, quads_occ4_center])
        tris_list.append(tris_inside_occ4)
        # plot_triangles(
        #     "tris_inside_after_4.png",
        #     triangles=tris_inside_occ4,
        #     vertices=vertices,
        #     x_nx2=x_nx2,
        #     s_n=s_n,
        # )
        # =====================================================================
        #
        # The next step is to find all quads where three corners are inside
        #
        # =====================================================================

        surf_vert_ids_occ3 = v_offsets[occ_sum[surf_cubes] == 3]
        case_ids_occ3 = case_ids[occ_sum[surf_cubes] == 3]
        quads_occ3 = square_fx4[occ_sum == 3]
        index_occ3 = self.tet_table[case_ids_occ3]
        K_occ3 = index_occ3.shape[1]  # number of triangles per quad

        # Expand quads so we can gather per-row
        quads_expanded_occ3 = quads_occ3.unsqueeze(1).expand(-1, K_occ3, -1)

        outside_edges_occ3 = mapping_inside_verts[
            torch.gather(quads_expanded_occ3, 2, index_occ3)
        ]
        # and connect it to the center node:
        # each triangle = [center, corner1, corner2]
        centers = surf_vert_ids_occ3.unsqueeze(1).expand(-1, K_occ3)  # (Q, K)
        tris_inside_occ3 = torch.cat(
            [centers.reshape(-1, 1), outside_edges_occ3.reshape(-1, 2)], dim=1
        )
        tris_list.append(tris_inside_occ3)
        # plot_triangles(
        #     "tris_inside_after_3.png",
        #     triangles=tris_inside_occ3,
        #     vertices=vertices,
        #     x_nx2=x_nx2,
        #     s_n=s_n,
        # )
        # =====================================================================
        #
        # The next step is to find all quads where two corners are inside
        #             non edge cases (i.e. all except 6 and 9)
        # =====================================================================
        mask_occ2_standard = torch.logical_and(vd_counts == 1, occ_sum[surf_cubes] == 2)
        if mask_occ2_standard.any():
            surf_vert_ids_occ2_standard = v_offsets[mask_occ2_standard]
            case_ids_occ2_standard = case_ids[mask_occ2_standard]
            quads_occ2_standard = square_fx4[surf_cubes][mask_occ2_standard]  # (Q,4)
            quads_expanded_occ2_standard = quads_occ2_standard.unsqueeze(1).expand(
                -1, 2, -1
            )
            index_occ2_standard = self.tet_table[case_ids_occ2_standard]  # (Q, 2, 2)

            valid_mask_occ2_standard = index_occ2_standard[..., 0] != -1  # (Q, 2)

            outside_edges_occ2_standard = torch.gather(
                quads_expanded_occ2_standard[valid_mask_occ2_standard, :],
                1,
                index_occ2_standard[valid_mask_occ2_standard],
            )

            outside_edges_occ2_standard = mapping_inside_verts[
                outside_edges_occ2_standard
            ]

            centers_occ2_standard = surf_vert_ids_occ2_standard.unsqueeze(1)  # (Q, K)
            tris_inside_occ2_standard = torch.cat(
                [
                    centers_occ2_standard.reshape(-1, 1),
                    outside_edges_occ2_standard.reshape(-1, 2),
                ],
                dim=1,
            )
            tris_list.append(tris_inside_occ2_standard)
            # plot_triangles(
            #     "tris_inside_after_2_standard.png",
            #     triangles=tris_inside_occ2_standard,
            #     vertices=vertices,
            #     x_nx2=x_nx2,
            #     s_n=s_n,
            # )
        # =====================================================================
        #
        # The next step is to find all quads where two corners are inside
        #             edge cases (i.e. 6 and 9)
        # =====================================================================
        mask_occ2_edge = torch.logical_and(vd_counts == 2, occ_sum[surf_cubes] == 2)
        if mask_occ2_edge.any():
            center_1_odd2_edge = (v_offsets[mask_occ2_edge]).unsqueeze(1).repeat(1, 2)
            center_2_odd2_edge = center_1_odd2_edge + 1
            case_ids_occ2_edge = case_ids[mask_occ2_edge]
            # occ sum = on the global level, mask_occ2_edge = on surface level
            quads_occ2_edge = square_fx4[surf_cubes][mask_occ2_edge]  # (Q,4)
            quads_expanded_occ2_edge = quads_occ2_edge.unsqueeze(1).expand(-1, 2, -1)
            index_occ2_edge = self.tet_table[case_ids_occ2_edge]  # (Q, 2, 2)

            valid_mask_occ2_edge = index_occ2_edge[..., 0] != -1  # (Q, 2)

            outside_edges_occ2_edge = torch.gather(
                quads_expanded_occ2_edge[valid_mask_occ2_edge, :],
                1,
                index_occ2_edge[valid_mask_occ2_edge],
            )

            outside_edges_occ2_edge = mapping_inside_verts[outside_edges_occ2_edge]

            tris_inside_occ2_edge = torch.cat(
                [
                    center_1_odd2_edge.reshape(-1, 1),
                    center_2_odd2_edge.reshape(-1, 1),
                    outside_edges_occ2_edge.reshape(-1, 1),
                ],
                dim=1,
            )
            # plot_triangles(
            #     "tris_inside_after_2_edge.png",
            #     triangles=tris_inside_occ2_edge,
            #     vertices=vertices,
            #     x_nx2=x_nx2,
            #     s_n=s_n,
            # )

            tris_list.append(tris_inside_occ2_edge)
        tris = torch.cat(tris_list)
        # fix orientation
        p0 = vertices[tris[:, 0]]
        p1 = vertices[tris[:, 1]]
        p2 = vertices[tris[:, 2]]

        cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
            p1[:, 1] - p0[:, 1]
        ) * (p2[:, 0] - p0[:, 0])

        flip = cross < 0

        # swap columns 1 and 2 where needed
        tris_flipped = tris.clone()
        tris_flipped[flip, 1] = tris[flip, 2]
        tris_flipped[flip, 2] = tris[flip, 1]

        tris = tris_flipped

        return vertices, tris

    def check_open_mesh(self, edges, num_vertices=None):
        # edges shape: (Num_Edges, 2)
        flattened_edges = edges.reshape(-1)

        # Count occurrences of each vertex ID
        if num_vertices is None:
            num_vertices = flattened_edges.max() + 1

        counts = torch.bincount(flattened_edges, minlength=num_vertices)

        # A mesh is open if any vertex is used only once
        open_points = (counts == 1).sum()
        if open_points != 0:
            raise RuntimeError(
                "Unclosed mesh found. Check your bounds or SDF definition!"
            )


# def plot_triangles(filename, triangles, vertices, x_nx2, s_n):
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     ax.set_aspect(1)
#     for tri in triangles:
#         v0, v1, c = tri
#         p1 = vertices[v0]
#         p2 = vertices[v1]
#         p3 = vertices[c]
#         ax.plot(
#             [p1[0].item(), p2[0].item(), p3[0].item(), p1[0].item()],
#             [p1[1].item(), p2[1].item(), p3[1].item(), p1[1].item()],
#             "-k",
#         )
#         ax.text(
#             p1[0].item(),
#             p1[1].item(),
#             str(v0),
#             color="black",
#             fontsize=8,
#             ha="center",
#             va="center",
#         )
#         ax.text(
#             p2[0].item(),
#             p2[1].item(),
#             str(v1),
#             color="black",
#             fontsize=8,
#             ha="center",
#             va="center",
#         )
#         ax.text(
#             p3[0].item(),
#             p3[1].item(),
#             str(c),
#             color="black",
#             fontsize=8,
#             ha="center",
#             va="center",
#         )

#     ax.scatter(
#         x_nx2[:, 0].detach().cpu().numpy(),
#         x_nx2[:, 1].detach().cpu().numpy(),
#         c=s_n.detach().cpu().numpy(),
#         cmap="coolwarm",
#     )
#     fig.savefig(filename, dpi=1000)

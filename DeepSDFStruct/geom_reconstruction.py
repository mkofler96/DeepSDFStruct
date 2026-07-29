"""
Shape Reconstruction
====================

Fit a spatially varying field of DeepSDF latent codes to a target mesh.

The decoder of a trained DeepSDF model turns a single latent code into one
shape. To represent a whole part rather than a single microtile, this module
tiles that decoder over a domain and lets the latent code *vary in space*: the
codes are the control points of a B-spline, and fitting the shape means
optimizing those control points against SDF samples taken from a target mesh.
Because neighbouring tiles read nearby points of the same spline, the result is
a continuous structure -- a field of local shapes.

:class:`LocalShapesReconstructor` wraps that whole procedure::

    import torch, trimesh
    from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor

    recon = LocalShapesReconstructor(output_dir="output")
    torch.manual_seed(42)

    mesh_orig = trimesh.load_mesh("part.stl")
    assert mesh_orig.is_watertight, "target mesh must be watertight"

    struct, scaling, gt_sdf, params = recon.fit_mesh(
        mesh=mesh_orig, tiling=[32, 32, 32]
    )
    recon.export(struct, scaling)

``struct`` is a :class:`~DeepSDFStruct.lattice_structure.LatticeSDFStruct`
defined in parameter space; ``scaling`` maps it back onto the original mesh's
coordinates, and is what you hand to
:func:`~DeepSDFStruct.mesh.create_3D_mesh` as ``deformation_function``.

Functions
---------
build_parameter_spline
    Build the B-spline whose control points carry the latent codes.
sample_gt_sdf
    Draw uniform and near-surface SDF samples from a target mesh.

Classes
-------
LocalShapesReconstructor
    Fits a latent-code field to a mesh.
StructBuild
    Named tuple returned by :meth:`LocalShapesReconstructor.build_struct`.
FitResult
    Named tuple returned by :meth:`LocalShapesReconstructor.fit_mesh`.
"""

import logging
import pathlib
from typing import Any, Callable, NamedTuple

import numpy as np
import splinepy
import torch
import trimesh

import DeepSDFStruct
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.mesh import export_reconstructed_artifacts
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.pretrained_models import PretrainedModels, get_model
from DeepSDFStruct.SDF import SDFBase, SDFfromDeepSDF, SDFfromMesh
from DeepSDFStruct.SDF import normalize_mesh_to_unit_cube
from DeepSDFStruct.sampling import SampledSDF, random_sample_sdf, sample_mesh_surface
from DeepSDFStruct.torch_spline import TorchScaling

logger = logging.getLogger(DeepSDFStruct.__name__)

__all__ = [
    "LocalShapesReconstructor",
    "StructBuild",
    "FitResult",
    "build_parameter_spline",
    "sample_gt_sdf",
]


def build_parameter_spline(
    spline_degrees: list[int],
    tiling: list[int],
    latent_dim: int,
    bounds: np.ndarray | None = None,
) -> splinepy.BSpline:
    """Build the B-spline that carries one latent code per control point.

    Starts from a single clamped span per dimension and inserts ``n_box - 1``
    uniformly spaced interior knots, so the spline has exactly one knot span
    per microtile. Control points are zero-initialized; callers are expected to
    set them (see :meth:`LocalShapesReconstructor.build_struct`, which starts
    from the mean trained latent vector).

    Parameters
    ----------
    spline_degrees : list of 3 int
        Polynomial degree per spatial dimension. ``[1, 1, 1]`` gives trilinear
        interpolation between neighbouring latent codes.
    tiling : list of 3 int
        Number of knot spans (microtiles) per dimension.
    latent_dim : int
        Width of each control point, i.e. the model's latent code length.
    bounds : (2, 3) array-like, optional
        ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` spanned by the spline.
        Defaults to the unit cube ``[0, 1]^3``.

    Returns
    -------
    splinepy.BSpline
        Spline with ``prod(tiling + 1)`` control points of width ``latent_dim``
        for degree 1.
    """
    if bounds is not None:
        bounds = np.asarray(bounds)
        mins = bounds[0]
        maxs = bounds[1]
    else:
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([1.0, 1.0, 1.0])

    knot_vectors = [
        [mins[i]] * (spline_degrees[i] + 1) + [maxs[i]] * (spline_degrees[i] + 1)
        for i in range(3)
    ]

    n_ctrl_per_dim = [len(knot_vectors[i]) - spline_degrees[i] - 1 for i in range(3)]
    n_ctrl_total = int(np.prod(n_ctrl_per_dim))
    control_points = np.zeros((n_ctrl_total, latent_dim))

    param_spline_sp = splinepy.BSpline(spline_degrees, knot_vectors, control_points)

    for i_box, n_box in enumerate(tiling):
        knots = np.linspace(mins[i_box], maxs[i_box], n_box + 1)[1:-1]
        if len(knots) == 0:
            continue
        logger.debug(
            "Inserting %d knots at %s into spline dim %d", n_box - 1, knots, i_box
        )
        param_spline_sp.insert_knots(i_box, knots)

    return param_spline_sp


def sample_gt_sdf(
    gt_sdf: SDFBase,
    mesh: trimesh.Trimesh,
    bounds: torch.Tensor,
    *,
    n_uniform: int,
    n_surface: int,
    device="cpu",
    stds=(0.025, 0.0001),
    box_constrained: bool = False,
) -> SampledSDF:
    """Draw uniform and near-surface SDF samples from a target mesh.

    Combines a uniform fill of ``bounds`` (which teaches the fit where the
    material is *not*) with a dense band hugging the surface (which sharpens
    the zero level set).

    Parameters
    ----------
    gt_sdf : SDFBase
        Ground-truth SDF, normally ``SDFfromMesh(mesh, scale=False)``. Passed
        in rather than built here so the caller can reuse the same object for
        error metrics afterwards.
    mesh : trimesh.Trimesh
        Target mesh, already in the coordinate system of ``bounds``.
    bounds : torch.Tensor
        (2, 3) box for the uniform samples.
    n_uniform : int
        Number of uniform samples inside ``bounds``.
    n_surface : int
        Number of surface points; each is perturbed once per entry of ``stds``,
        so the surface band holds ``n_surface * len(stds)`` samples (or exactly
        ``n_surface`` when ``box_constrained`` is set).
    device : str or torch.device
        Device the samples are created on.
    stds : sequence of float
        Standard deviations of the Gaussian noise added to surface points --
        one coarse, one fine by convention.
    box_constrained : bool, default False
        If True, reject surface samples outside ``bounds`` and re-sample until
        ``n_surface`` accepted points are collected. Needed when the lattice
        domain is smaller than the mesh's own bounding box, as in a shape
        optimization whose design domain is a sub-box.

    Returns
    -------
    SampledSDF
        Concatenation of the uniform and surface samples.
    """
    uniform_samples = random_sample_sdf(
        gt_sdf, bounds, n_samples=n_uniform, sampling_strategy="uniform", device=device
    )
    surface_samples = sample_mesh_surface(
        gt_sdf, mesh, n_samples=n_surface, stds=list(stds), device=device
    )

    if not box_constrained:
        return uniform_samples + surface_samples

    # Rejection sampling: keep only surface samples inside bounds, topping up
    # in rounds since the acceptance rate is unknown a priori.
    bmin = bounds[0].to(surface_samples.samples.device)
    bmax = bounds[1].to(surface_samples.samples.device)

    inside = (
        (surface_samples.samples >= bmin) & (surface_samples.samples <= bmax)
    ).all(dim=1)
    n_accepted = int(inside.sum().item())
    acceptance_rate = (
        n_accepted / surface_samples.samples.shape[0] if n_accepted > 0 else 0.0
    )

    all_samples = [surface_samples.samples[inside]]
    all_distances = [surface_samples.distances[inside]]
    n_collected = n_accepted

    max_rounds = 10
    for _ in range(max_rounds):
        if n_collected >= n_surface:
            break
        n_needed = n_surface - n_collected
        oversample_factor = max(1.0 / max(acceptance_rate, 0.01), 2.0)
        n_to_sample = int(n_needed * oversample_factor * 1.5)

        extra = sample_mesh_surface(
            gt_sdf, mesh, n_samples=n_to_sample, stds=list(stds), device=device
        )
        inside_extra = ((extra.samples >= bmin) & (extra.samples <= bmax)).all(dim=1)
        all_samples.append(extra.samples[inside_extra])
        all_distances.append(extra.distances[inside_extra])

        n_new = int(inside_extra.sum().item())
        n_collected += n_new
        if n_new > 0:
            acceptance_rate = n_new / extra.samples.shape[0]

    final_samples = torch.cat(all_samples, dim=0)[:n_surface]
    final_distances = torch.cat(all_distances, dim=0)[:n_surface]
    logger.debug(
        "Box-constrained sampling (rejection): %d/%d surface samples inside "
        "bounds (acceptance rate ~%.1f%%)",
        final_samples.shape[0],
        n_surface,
        acceptance_rate * 100,
    )
    if final_samples.shape[0] < n_surface:
        logger.warning(
            "Only collected %d of %d requested surface samples inside bounds",
            final_samples.shape[0],
            n_surface,
        )

    return uniform_samples + SampledSDF(
        samples=final_samples, distances=final_distances
    )


class StructBuild(NamedTuple):
    """What :meth:`LocalShapesReconstructor.build_struct` produces.

    Attributes
    ----------
    struct : LatticeSDFStruct
        Lattice over the normalized mesh, latent codes initialized but not yet
        fitted.
    scaling : TorchScaling
        Parameter space -> original mesh coordinates.
    mesh_norm : trimesh.Trimesh
        The target mesh normalized into parameter space.
    bounds : torch.Tensor
        (2, 3) bounds of ``struct``, taken from ``mesh_norm``.
    param_spline : SplineParametrization
        The latent-code spline; its control points are the free variables.
    param_spline_sp : splinepy.BSpline
        The underlying splinepy spline, for knot-grid exports.
    scale : float
        Inverse of the normalization scale, as consumed by ``TorchScaling``.
    shift : numpy.ndarray
        Center of the original mesh's bounding box.
    """

    struct: LatticeSDFStruct
    scaling: TorchScaling
    mesh_norm: trimesh.Trimesh
    bounds: torch.Tensor
    param_spline: SplineParametrization
    param_spline_sp: splinepy.BSpline
    scale: float
    shift: np.ndarray


class FitResult(NamedTuple):
    """Outcome of :meth:`LocalShapesReconstructor.fit_mesh`.

    Deliberately just the four headline outputs, so the common case unpacks
    directly::

        struct, scaling, gt_sdf, params = recon.fit_mesh(...)

    Fit diagnostics are not carried here. For the loss curve, pass
    ``loss_plot_path`` / ``loss_csv_path`` to write them out, or drive the
    steps yourself (:meth:`~LocalShapesReconstructor.build_struct`,
    :func:`sample_gt_sdf`, :meth:`~LocalShapesReconstructor.fit_samples`) --
    ``fit_samples`` returns ``loss_history``, ``final_loss`` and ``num_steps``.

    Attributes
    ----------
    struct : LatticeSDFStruct
        The fitted structure, in parameter space, with its parametrization
        already set to the optimized control points. ``struct.bounds`` holds
        the evaluation bounds.
    scaling : TorchScaling
        Parameter space -> original mesh coordinates. Hand this to
        :func:`~DeepSDFStruct.mesh.create_3D_mesh` as ``deformation_function``.
    gt_sdf : SDFBase
        Ground-truth SDF of the normalized target mesh, reusable for error
        metrics.
    params : list of torch.Tensor
        Optimized parameters, i.e. ``[control_points]`` of shape
        ``(n_control_points, latent_dim)``.
    """

    struct: LatticeSDFStruct
    scaling: TorchScaling
    gt_sdf: SDFBase
    params: list[torch.Tensor]


class LocalShapesReconstructor:
    """Fits a field of DeepSDF latent codes to a target mesh.

    Parameters
    ----------
    model : str, PretrainedModels or DeepSDFModel, optional
        The decoder providing the microtile. Accepts a
        :class:`~DeepSDFStruct.pretrained_models.PretrainedModels` member, a
        path to a checkpoint directory, or an already loaded
        :class:`~DeepSDFStruct.deep_sdf.models.DeepSDFModel`. Defaults to the
        bundled primitives decoder.
    checkpoint : str, default "latest"
        Checkpoint name inside the model directory. Ignored when ``model`` is
        an already loaded model.
    device : str or torch.device, optional
        Compute device. Defaults to CUDA when available.
    output_dir : path-like or None, default ``"tests/tmp_outputs"``
        Where every file this reconstructor writes goes: the loss curve and CSV
        of :meth:`fit_mesh`, and whatever :meth:`export` produces. Relative
        paths are resolved against the working directory. The directory is
        created on first write, not here. Pass ``None`` to keep the
        reconstructor from writing anything unless an explicit path is given.

        The default points at the repository's gitignored scratch directory, so
        a run started from a checkout does not leave artifacts in the repository
        root. It is relative to the working directory, so callers running from
        outside a checkout should pass an explicit path.

    Examples
    --------
    >>> recon = LocalShapesReconstructor()  # doctest: +SKIP
    >>> struct, scaling, gt_sdf, params = recon.fit_mesh(  # doctest: +SKIP
    ...     mesh=mesh, tiling=[8, 8, 8]
    ... )
    >>> recon.export(struct, scaling)  # doctest: +SKIP

    Notes
    -----
    :meth:`fit_mesh` is the one-call path. When you need to interleave work --
    exporting intermediate fields, or reusing one structure across several
    fits -- use :meth:`build_struct`, :func:`sample_gt_sdf` and
    :meth:`fit_samples` separately; ``fit_mesh`` is just their composition.
    """

    def __init__(
        self,
        model: str | PretrainedModels | DeepSDFModel = PretrainedModels.Primitives,
        checkpoint: str = "latest",
        device=None,
        output_dir: str | pathlib.Path | None = "tests/tmp_outputs",
    ):
        if isinstance(model, DeepSDFModel):
            self.model = model
        else:
            if device is not None and "cuda" in str(device):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"device={device!r} was requested, but "
                        "torch.cuda.is_available() is False."
                    )
            self.model = get_model(model, checkpoint=checkpoint, device=device)

        self.device = self.model.device
        self.microtile = SDFfromDeepSDF(self.model)
        self.latent_dim = int(self.model._trained_latent_vectors[0].shape[0])
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None

    def ensure_output_dir(self) -> pathlib.Path:
        """Create :attr:`output_dir` if needed and return it.

        Raises
        ------
        ValueError
            If the reconstructor was built with ``output_dir=None``, i.e. was
            explicitly told not to write files.
        """
        if self.output_dir is None:
            raise ValueError(
                "This LocalShapesReconstructor was created with output_dir=None, "
                "so it has no directory to write to. Pass output_dir=... to the "
                "constructor, or an explicit path to the method that writes."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def build_struct(
        self,
        mesh: trimesh.Trimesh,
        tiling: list[int],
        *,
        spline_degree=(1, 1, 1),
        shrink_factor: float = 1.0,
    ):
        """Normalize *mesh* and build the latent-code spline and structure.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Target mesh. Copied before normalization, so the caller's mesh is
            left untouched.
        tiling : list of 3 int
            Microtiles per dimension.
        spline_degree : sequence of 3 int, default (1, 1, 1)
            Degree of the latent-code spline per dimension.
        shrink_factor : float, default 1.0
            Passed to
            :func:`~DeepSDFStruct.SDF.normalize_mesh_to_unit_cube`. Values
            below 1 leave a margin between the mesh and the domain border.

        Returns
        -------
        StructBuild
            Named tuple with the structure and everything derived alongside it.
        """
        mesh_norm, scale, shift = normalize_mesh_to_unit_cube(
            mesh.copy(), shrink_factor=shrink_factor
        )

        bounds = torch.tensor(mesh_norm.bounds, device=self.device, dtype=torch.float32)
        scaling = TorchScaling(
            scale_factors=scale, translation=shift, bounds=bounds, device=self.device
        )

        param_spline_sp = build_parameter_spline(
            spline_degrees=list(spline_degree),
            tiling=list(tiling),
            latent_dim=self.latent_dim,
            bounds=bounds.detach().cpu().numpy(),
        )
        param_spline = SplineParametrization(param_spline_sp, device=self.device)

        # Initialize every control point to the mean of the trained latent
        # vectors. The decoder is only well-conditioned inside the learned
        # latent manifold; starting from ~0 lands in a flat/degenerate region
        # where the fit stalls (pronounced at small latent dims).
        control_points = param_spline.torch_spline.control_points
        trained_codes = torch.stack(list(self.model._trained_latent_vectors), dim=0)
        mean_code = trained_codes.mean(dim=0).to(
            device=control_points.device, dtype=control_points.dtype
        )
        param_spline.set_param(mean_code.expand(control_points.shape))

        struct = LatticeSDFStruct(
            tiling=list(tiling),
            microtile=self.microtile,
            parametrization=param_spline,
            bounds=bounds,
        )
        return StructBuild(
            struct=struct,
            scaling=scaling,
            mesh_norm=mesh_norm,
            bounds=bounds,
            param_spline=param_spline,
            param_spline_sp=param_spline_sp,
            scale=scale,
            shift=shift,
        )

    @staticmethod
    def fit_samples(
        struct: LatticeSDFStruct,
        samples: SampledSDF,
        *,
        num_iterations: int = 10,
        lr: float = 5e-3,
        batch_size: int = 4096,
        code_reg_lambda: float = 0.0,
        code_bound: float | None = 1.0,
        grad_clip: float | None = 1.0,
        eikonal_lambda: float = 0.0,
        loss_plot_path=None,
        loss_csv_path=None,
        step_callback: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Optimize *struct*'s latent codes against *samples*.

        Static: everything needed is already inside ``struct``, so this can be
        called as ``LocalShapesReconstructor.fit_samples(struct, samples, ...)``
        to refit a lattice without loading a model again.

        Parameters
        ----------
        struct : LatticeSDFStruct
            Structure to fit; its parametrization parameters are optimized
            in place.
        samples : SampledSDF
            Target samples, in the same (parameter) space as ``struct``.
        num_iterations : int, default 10
            Number of epochs over ``samples``. One epoch is
            ``len(samples) // batch_size`` optimizer steps, so with the default
            sample counts of :meth:`fit_mesh` this is already a few thousand
            steps.
        lr : float, default 5e-3
            Adam learning rate.
        batch_size : int, default 4096
            Samples per optimizer step.
        code_reg_lambda : float, default 0.0
            Weight of the L2 penalty on evaluated latent codes.
        code_bound : float or None, default 1.0
            If set, control points are clamped to ``[-code_bound, code_bound]``
            after every step, keeping them inside the range the decoder was
            trained on. ``None`` disables the clamp.
        grad_clip : float or None, default 1.0
            If set, gradient-norm clipping threshold. ``None`` disables it.
        eikonal_lambda : float, default 0.0
            Weight of the near-surface Eikonal penalty ``(|grad SDF| - 1)^2``.
        loss_plot_path, loss_csv_path : path-like, optional
            Where to write the loss curve and its raw values.
        step_callback : callable, optional
            Called as ``step_callback(epoch, batch_idx, n_batches)`` after
            every optimizer step, for progress exports.

        Returns
        -------
        dict
            Keys ``params``, ``loss_history``, ``final_loss``, ``num_steps``.
        """
        return reconstruct_from_samples(
            struct,
            samples,
            num_iterations=num_iterations,
            lr=lr,
            loss_fn="ClampedL1",
            batch_size=batch_size,
            use_tanh_on_gt=False,
            loss_plot_path=loss_plot_path,
            loss_csv_path=loss_csv_path,
            optimizer_name="adam",
            # The samples are already in parameter space, so no mapping back.
            deformation_function=None,
            code_reg_lambda=code_reg_lambda,
            code_bound=code_bound,
            grad_clip=grad_clip,
            eikonal_lambda=eikonal_lambda,
            step_callback=step_callback,
        )

    def fit_mesh(
        self,
        mesh: trimesh.Trimesh,
        tiling: list[int],
        *,
        num_iterations: int = 10,
        lr: float = 5e-3,
        batch_size: int = 4096,
        n_uniform: int = int(1e5),
        n_surface: int = int(5e5),
        spline_degree=(1, 1, 1),
        shrink_factor: float = 1.0,
        samples_surface_stds=(0.005, 0.0001),
        box_constrained: bool = False,
        code_reg_lambda: float = 0.0,
        code_bound: float | None = 1.0,
        grad_clip: float | None = 1.0,
        eikonal_lambda: float = 0.0,
        loss_plot_path=None,
        loss_csv_path=None,
        step_callback: Callable[[int, int, int], None] | None = None,
    ) -> FitResult:
        """Fit a latent-code field to *mesh*, end to end.

        Normalizes the mesh into parameter space, builds the structure, samples
        the ground-truth SDF, runs the fit, and writes the optimized codes back
        into the structure.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Target mesh. Should be watertight, otherwise the sign of the
            ground-truth SDF is not well defined.
        tiling : list of 3 int
            Microtiles per dimension.
        num_iterations, lr, batch_size : int, float, int
            Fitting hyperparameters, see :meth:`fit_samples`.
        n_uniform, n_surface : int
            Sample counts, see :func:`sample_gt_sdf`.
        spline_degree, shrink_factor
            Structure construction, see :meth:`build_struct`.
        samples_surface_stds, box_constrained
            Sampling behaviour, see :func:`sample_gt_sdf`.
        code_reg_lambda, code_bound, grad_clip, eikonal_lambda
            Regularization, see :meth:`fit_samples`.
        loss_plot_path, loss_csv_path : path-like, optional
            Where the loss curve and its raw values go. Both default to
            ``reconstruction_loss.png`` / ``.csv`` inside
            :attr:`output_dir`; with ``output_dir=None`` they are not written.
        step_callback
            Diagnostics, see :meth:`fit_samples`.

        Returns
        -------
        FitResult
            Unpacks as ``struct, scaling, gt_sdf, params``. Fit diagnostics are
            not included -- see :class:`FitResult` for how to get them.
        """
        if self.output_dir is not None and (
            loss_plot_path is None or loss_csv_path is None
        ):
            out_dir = self.ensure_output_dir()
            if loss_plot_path is None:
                loss_plot_path = out_dir / "reconstruction_loss.png"
            if loss_csv_path is None:
                loss_csv_path = out_dir / "reconstruction_loss.csv"

        built = self.build_struct(
            mesh, tiling, spline_degree=spline_degree, shrink_factor=shrink_factor
        )
        struct, scaling, mesh_norm, bounds = (
            built.struct,
            built.scaling,
            built.mesh_norm,
            built.bounds,
        )

        gt_sdf = SDFfromMesh(mesh_norm, scale=False)
        samples = sample_gt_sdf(
            gt_sdf,
            mesh_norm,
            bounds,
            n_uniform=n_uniform,
            n_surface=n_surface,
            device=self.device,
            stds=samples_surface_stds,
            box_constrained=box_constrained,
        )

        result = self.fit_samples(
            struct,
            samples,
            num_iterations=num_iterations,
            lr=lr,
            batch_size=batch_size,
            code_reg_lambda=code_reg_lambda,
            code_bound=code_bound,
            grad_clip=grad_clip,
            eikonal_lambda=eikonal_lambda,
            loss_plot_path=loss_plot_path,
            loss_csv_path=loss_csv_path,
            step_callback=step_callback,
        )

        struct.parametrization.set_param(result["params"][0])
        logger.info(
            "Fitted %d control points in %d steps, final loss %.6e",
            result["params"][0].shape[0],
            result["num_steps"],
            result["final_loss"],
        )

        return FitResult(
            struct=struct, scaling=scaling, gt_sdf=gt_sdf, params=result["params"]
        )

    def export(
        self,
        struct: LatticeSDFStruct,
        scaling: TorchScaling | None = None,
        *,
        mesh_resolution: int = 32,
        bounds=None,
        output_dir: str | pathlib.Path | None = None,
        **kwargs,
    ) -> pathlib.Path:
        """Write *struct* to disk as an SDF grid plus surface meshes.

        Thin wrapper around
        :func:`~DeepSDFStruct.mesh.export_reconstructed_artifacts` that fills in
        what the reconstructor already knows: the output directory (created if
        missing), the device, and -- unless overridden -- the structure's own
        bounds.

        Parameters
        ----------
        struct : LatticeSDFStruct
            Structure to export, normally the one :meth:`fit_mesh` returned.
        scaling : TorchScaling, optional
            Parameter-to-physical-space map. Without it only the
            parameter-space mesh is written.
        mesh_resolution : int, default 32
            FlexiCubes grid resolution per dimension.
        bounds : torch.Tensor, optional
            Evaluation bounds. Defaults to ``struct.bounds``.
        output_dir : path-like, optional
            Destination, overriding :attr:`output_dir` for this call.
        **kwargs
            Forwarded to
            :func:`~DeepSDFStruct.mesh.export_reconstructed_artifacts`, e.g.
            ``sdf_grid_N`` or the file-name arguments.

        Returns
        -------
        pathlib.Path
            Path of the physical-space mesh, or of the parameter-space mesh
            when no ``scaling`` was given.
        """
        if output_dir is None:
            out_dir = self.ensure_output_dir()
        else:
            out_dir = pathlib.Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        return export_reconstructed_artifacts(
            struct,
            out_dir,
            mesh_resolution=mesh_resolution,
            bounds=struct.bounds if bounds is None else bounds,
            device=self.device,
            scaling=scaling,
            **kwargs,
        )

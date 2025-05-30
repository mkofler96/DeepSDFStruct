from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct, constantLatvec
import splinepy
import torch

model = get_model(PretrainedModels.RoundCross)
sdf = SDFfromDeepSDF(model)

sdf.set_latent_vec(torch.tensor([0.3]))
sdf.plot_slice()


deformation_spline = splinepy.helpme.create.box(2, 1, 1)
lattice_struct = LatticeSDFStruct(
    tiling=(6, 3, 1),
    deformation_spline=deformation_spline,
    microtile=sdf,
    parametrization_spline=constantLatvec([0.5]),
)

lattice_struct.plot_slice(origin=(0, 0, 0.5))

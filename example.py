from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
import torch

model = get_model(PretrainedModels.RoundCross)
sdf = SDFfromDeepSDF(model)
inputs = torch.tensor([[0, 0, 0], [0, 0.1, 0]])
print(sdf(inputs))

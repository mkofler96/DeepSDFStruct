import onnx
import torch
import json
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels


def export_ONNX(onnx_filename):
    model = get_model(PretrainedModels.Primitives)

    torch.onnx.export(
        model._decoder,
        torch.randn(
            1, model._trained_latent_vectors[0].shape[0] + 3, device=model.device
        ),
        onnx_filename,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        verify=True,
        dynamo=False,
    )
    onnx_model = onnx.load(onnx_filename)

    meta = {"author": "Michael Kofler", "trainingelementsize": "1"}

    for key, value in meta.items():
        meta_prop = onnx_model.metadata_props.add()
        meta_prop.key = key
        meta_prop.value = value

    onnx.save(onnx_model, onnx_filename)


if __name__ == "__main__":
    export_ONNX("model.onnx")

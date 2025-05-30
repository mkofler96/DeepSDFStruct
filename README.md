# DeepSDFStruct

A differentiable framework for generating and deforming 3D microstructured materials using Signed Distance Functions (SDFs) and spline-based lattices.

## 📦 Installation

You can install `DeepSDFStruct` directly from GitHub using `pip`:

```bash
pip install git+https://github.com/mkofler96/DeepSDFStruct.git
```

### ⚠️ Troubleshooting

If you encounter issues during installation, contact Michael Kofler.

---

## 🚀 Quick Start

Here's a simple example using a pretrained DeepSDF model, lattice structure generation, and deformation:

```python
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct, constantLatvec
import splinepy
import torch

# Load a pretrained DeepSDF model
model = get_model(PretrainedModels.RoundCross)
sdf = SDFfromDeepSDF(model)

# Set the latent vector and visualize a slice of the SDF
sdf.set_latent_vec(torch.tensor([0.3]))
sdf.plot_slice()

# Define a spline-based deformation field
deformation_spline = splinepy.helpme.create.box(2, 1, 1)

# Create the lattice structure with deformation and microtile
lattice_struct = LatticeSDFStruct(
    tiling=(6, 3, 1),
    deformation_spline=deformation_spline,
    microtile=sdf,
    parametrization_spline=constantLatvec([0.5]),
)

# Visualize a slice of the final lattice structure
lattice_struct.plot_slice(origin=(0, 0, 0.5))
```

---

Since all SDFs are callable, the signed distance can be obtained by calling e.g.
```
lattice_struct(torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.float32))
```

## 🔗 Repository

GitHub: [https://github.com/mkofler96/DeepSDFStruct](https://github.com/mkofler96/DeepSDFStruct)

---

## 📄 License
This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file or visit [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0) for more information.

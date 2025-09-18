# DeepSDFStruct

A differentiable framework for generating and deforming 3D microstructured materials using Signed Distance Functions (SDFs) and spline-based lattices.
## Coverage
[![Test Python Package](https://github.com/mkofler96/DeepSDFStruct/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/mkofler96/DeepSDFStruct/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/mkofler96/DeepSDFStruct/badge.svg?branch=main)](https://coveralls.io/github/mkofler96/DeepSDFStruct?branch=main)
## 📦 Installation

You can install `DeepSDFStruct` directly from GitHub using `pip`:

```bash
pip install git+https://github.com/mkofler96/DeepSDFStruct.git
```
To add it to your uv project run:
```
uv add git+https://github.com/mkofler96/DeepSDFStruct.git
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
```python
lattice_struct(torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.float32))
```
Furthermore, a differentiable surface mesh can be extracted and exported
```python
from DeepSDFStruct.mesh import create_3D_surface_mesh, export_surface_mesh_vtk
surf_mesh, derivative = create_3D_surface_mesh(
    lattice_struct, 30, differentiate=True
)
export_surface_mesh_vtk(
    surf_mesh.vertices, surf_mesh.faces, "mesh_with_derivative.vtk", derivative
)
```
The libraries gustaf and meshio can be used to export the mesh to differnt formats.
```python
import gustaf as gus
faces = surf_mesh.to_gus()
gus.io.meshio.export("faces.inp", faces)
gus.io.meshio.export("faces.obj", faces)
```
Finally, there is also a functionality to create a volumetric mesh from the generated surface mesh.
```python
from DeepSDFStruct.mesh import tetrahedralize_surface
volumes, _ = tetrahedralize_surface(faces)
gus.io.mfem.export("volumes.mfem", volumes)
```

## Training a Model
A model can be trained by using the `train_deep_sdf` function that takes as input the experiment directory and the data directory.
```python
from DeepSDFStruct.deep_sdf.train_deep_sdf import train_deep_sdf
train_deep_sdf("DeepSDFStruct/trained_models/test_experiment", data_dir)
```
Example data can be downloaded from huggingface
```python
from DeepSDFStruct.deep_sdf.train_deep_sdf import train_deep_sdf
from huggingface_hub import snapshot_download
data_dir = snapshot_download(
    "mkofler/lattice_structure_unit_cells", repo_type="dataset"
)
train_deep_sdf("DeepSDFStruct/trained_models/test_experiment", data_dir)
```
Note that this data contains the preprocessed and sampled data. 
The data that is used for training a DeepSDF model needs to be in the form of `.npz` files that contain the negative and positive points as an [x,y,z] array.
This can be achieved by using numpy's export function:
```python
np.savez(file_name, neg=neg_points, pos=pos_points)
```
## Generation of Training Data
Different types of geometry representations can be used for training. One possibility is to use the python library splinepy:
```python
import splinepy
import numpy as np
from DeepSDFStruct.sampling import SDFSampler
from DeepSDFStruct.splinepy_unitcells.chi_3D import Chi3D
from DeepSDFStruct.splinepy_unitcells.cross_lattice import CrossLattice

outdir = "./training_data"
splitdir = "./training_data/splits"
dataset_name = "microstructure"

sdf_sampler = SDFSampler(outdir, splitdir, dataset_name)

t_start = 0.1 * np.sqrt(2) / 2
t_end = 0.15 * np.sqrt(2) / 2
crosslattice_tiles = []
for t in np.linspace(t_start, t_end, 3):
    tile, _ = CrossLattice().create_tile(np.array([[t]]), make3D=True)
    crosslattice_tiles.append(splinepy.Multipatch(tile))

chi = Chi3D()
chi_tiles = []

for phi in np.linspace(0, -np.pi / 6, 2):
    for x2 in np.linspace(-0.1, 0.2, 2):
        t = 0.1
        x1 = 0.2
        r = 0.5 * t
        tile, _ = chi.create_tile(np.array([[phi, t, x1, x2, r]] * 5))
        chi_tiles.append(splinepy.Multipatch(tile))

sdf_sampler.add_class(chi_tiles, class_name="Chi3D_center")
sdf_sampler.add_class(crosslattice_tiles, class_name="CrossLattice")

sdf_sampler.process_geometries(
    sampling_strategy="uniform", n_faces=100, compute_mechanical_properties=False
)

sdf_sampler.write_json("chi_and_cross.json")
```
## 🔗 Repository

GitHub: [https://github.com/mkofler96/DeepSDFStruct](https://github.com/mkofler96/DeepSDFStruct)

The documentation can be found at: [mkofler96.github.io/DeepSDFStruct/](mkofler96.github.io/DeepSDFStruct/)

---

## 📄 License
This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file or visit [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0) for more information.

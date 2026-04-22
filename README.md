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
This section shows how to use the main features of this library. Further details can be found in [example.ipynb](example.ipynb).

Here's a simple example using a pretrained DeepSDF model, lattice structure generation, and deformation:

```python
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.torch_spline import TorchSpline
import splinepy
import torch
```

### Load a pretrained DeepSDF model

```python
model = get_model(PretrainedModels.AnalyticRoundCross)
sdf = SDFfromDeepSDF(model)
```

### Set the latent vector and visualize a slice of the SDF

```python
sdf.set_latent_vec(torch.tensor([0.3]))
_ = sdf.plot_slice(origin=(0, 0, 0))
```

![Notebook output 6](docs/readme_images/example_output_01.png)

### Define a spline-based deformation field

```python
import numpy as np

height = 1.0
surface_cps = np.array(
	[
		[0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0],
		[2.0, 1.0, 0.0],
		[3.0, 1.0, 0.0],
		[0.0, 0.0 + height, 0.0],
		[1.0, 0.0 + height, 0.0],
		[2.0, 1.0 + height, 0.0],
		[3.0, 1.0 + height, 0.0],
		[0.0, 0.0, 1.0],
		[1.0, 0.0, 1.0],
		[2.0, 1.0, 1.0],
		[3.0, 1.0, 1.0],
		[0.0, 0.0 + height, 1.0],
		[1.0, 0.0 + height, 1.0],
		[2.0, 1.0 + height, 1.0],
		[3.0, 1.0 + height, 1.0],
	]
)
spline = splinepy.Bezier(degrees=[3, 1, 1], control_points=surface_cps).bspline
deformation_spline = TorchSpline(spline)
```

### Create the lattice structure with deformation and microtile

```python
lattice_struct = LatticeSDFStruct(
	tiling=(6, 2, 1),
	microtile=sdf,
	parametrization=Constant([0.5], device=model.device),
)
```

### Visualize a slice of the final lattice structure

```python
_ = lattice_struct.plot_slice(origin=(0, 0, 0.5), deformation_function=deformation_spline)
```

![Notebook output 12](docs/readme_images/example_output_02.png)

Since all SDFs are callable, the signed distance can be obtained by calling e.g.

```python
lattice_struct(torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.float32))
```

The `CappedBorderSDF` class helps to avoid non-watertight meshs, by capping the borders.

```python
from DeepSDFStruct.SDF import CappedBorderSDF
capped_lattice = CappedBorderSDF(lattice_struct)
_ = capped_lattice.plot_slice(origin=(0, 0, 0.5), deformation_function=deformation_spline)
```

![Notebook output 16](docs/readme_images/example_output_03.png)

Furthermore, a differentiable surface mesh can be extracted and exporte

```python
from DeepSDFStruct.mesh import create_3D_mesh, export_surface_mesh

surf_mesh, derivative = create_3D_mesh(
	capped_lattice, 30, differentiate=True, mesh_type="surface"
)
export_surface_mesh(
	"mesh_with_derivative.vtk",surf_mesh, derivative
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

### Training a Model
A model can be trained by using the `train_deep_sdf` function that takes as input the experiment directory and the data directory.`train_deep_sdf("DeepSDFStruct/trained_models/test_experiment", data_dir)`

```python
from DeepSDFStruct.deep_sdf.training import train_deep_sdf
```

Example data can be downloaded from huggingface

```python
from huggingface_hub import snapshot_download
data_dir = snapshot_download(
	"mkofler/lattice_structure_unit_cells", repo_type="dataset"
)
train_deep_sdf("DeepSDFStruct/trained_models/test_experiment", data_dir)
```

![Notebook output 26](docs/readme_images/example_output_04.png)

Note that this data contains the preprocessed and sampled data.
The data that is used for training a DeepSDF model needs to be in the form of `.npz` files that contain the negative and positive points as an [x,y,z] array.
This can be achieved by using numpy's export function:
```python
np.savez(file_name, neg=neg_points, pos=pos_points)
```

### Generation of Training Data
Different types of geometry representations can be used for training. One possibility is to use the python library splinepy:

```python
import splinepy
import numpy as np
from DeepSDFStruct.sampling import SDFSampler
from DeepSDFStruct.splinepy_unitcells.chi_3D import Chi3D
from DeepSDFStruct.splinepy_unitcells.cross_lattice import CrossLattice

outdir = "./training_data"
splitdir = "./training_data/splits"
dataset_name = "example_dataset"

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
	sampling_strategy="uniform", n_faces=100
)

sdf_sampler.write_json("chi_and_cross.json")
```

For the full documentation, visit [mkofler96.github.io/DeepSDFStruct/](https://mkofler96.github.io/DeepSDFStruct/).

## 🎨 SDF Primitives and Operations

DeepSDFStruct provides a comprehensive set of geometric primitives and SDF operations for creating complex geometries. Below are visual examples of the available primitives and operations.

### 3D Primitives

<table>
    <tbody>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/sphere.png" alt="Sphere"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/box.png" alt="Box"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/cylinder.png" alt="Cylinder"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/cone.png" alt="Cone"></td>
        </tr>
        <tr>
            <td align="center"><b>Sphere:</b> Sphere with configurable center and radius</td>
            <td align="center"><b>Box:</b> Axis-aligned 3D box with configurable extents</td>
            <td align="center"><b>Cylinder:</b> Finite cylinder with customizable height and radius</td>
            <td align="center"><b>Cone:</b> Finite cone with base radius and height</td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/torus.png" alt="Torus"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/rounded_box.png" alt="Rounded Box"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/wireframe_box.png" alt="Wireframe Box"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/capsule.png" alt="Capsule"></td>
        </tr>
        <tr>
            <td align="center"><b>Torus:</b> Torus with major and minor radii</td>
            <td align="center"><b>Rounded Box:</b> Box with rounded corners</td>
            <td align="center"><b>Wireframe Box:</b> Box wireframe structure</td>
            <td align="center"><b>Capsule:</b> Line segment with spherical ends</td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/ellipsoid.png" alt="Ellipsoid"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/pyramid.png" alt="Pyramid"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/tetrahedron.png" alt="Tetrahedron"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/octahedron.png" alt="Octahedron"></td>
        </tr>
        <tr>
            <td align="center"><b>Ellipsoid:</b> Ellipsoid with different radii in each axis</td>
            <td align="center"><b>Pyramid:</b> Square pyramid with configurable height</td>
            <td align="center"><b>Tetrahedron:</b> Regular tetrahedron</td>
            <td align="center"><b>Octahedron:</b> Regular octahedron</td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/dodecahedron.png" alt="Dodecahedron"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/icosahedron.png" alt="Icosahedron"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/capped_cylinder.png" alt="Capped Cylinder"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/capped_cone.png" alt="Capped Cone"></td>
        </tr>
        <tr>
            <td align="center"><b>Dodecahedron:</b> Regular dodecahedron</td>
            <td align="center"><b>Icosahedron:</b> Regular icosahedron</td>
            <td align="center"><b>Capped Cylinder:</b> Cylinder with exact end caps</td>
            <td align="center"><b>Capped Cone:</b> Cone with exact end caps</td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/rounded_cylinder.png" alt="Rounded Cylinder"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/rounded_cone.png" alt="Rounded Cone"></td>
            <td style="width: 25%;"><img src="benchmarks_sdf_showcase/primitives/3D/corner_spheres.png" alt="Corner Spheres"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/3D/cross_ms.png" alt="Cross M"></td>
        </tr>
        <tr>
            <td align="center"><b>Rounded Cylinder:</b> Cylinder with smooth rounded transitions</td>
            <td align="center"><b>Rounded Cone:</b> Cone with smooth rounded transitions</td>
            <td align="center"><b>Corner Spheres:</b> Cube with spherical cutouts at corners</td>
            <td align="center"><b>Cross M:</b> Cross-shaped structure</td>
        </tr>
    </tbody>
</table>

<details>
<summary><b>Click to see 3D primitive code examples</b></summary>

```python
from DeepSDFStruct.sdf_primitives import (
    SphereSDF, BoxSDF, CylinderSDF, ConeSDF, TorusSDF,
    RoundedBoxSDF, WireframeBoxSDF, CapsuleSDF, EllipsoidSDF,
    PyramidSDF, TetrahedronSDF, OctahedronSDF,
    DodecahedronSDF, IcosahedronSDF, CappedCylinderSDF,
    CappedConeSDF, RoundedCylinderSDF, RoundedConeSDF,
    CornerSpheresSDF, CrossMsSDF
)

# Sphere
sphere = SphereSDF(center=[0, 0, 0], radius=1.0)

# Box
box = BoxSDF(center=[0, 0, 0], extents=[1.0, 1.0, 1.0])

# Cylinder
cylinder = CylinderSDF(point=[0, 0, -0.5], axis=[0, 0, 1], radius=0.4, height=1.0)

# Cone
cone = ConeSDF(apexpoint=[0, 0, -0.5], axis=[0, 0, 1], radius=0.4, height=1.0)

# Torus
torus = TorusSDF(center=[0, 0, 0], axis=[0, 0, 1], major_radius=1.0, minor_radius=0.3)

# Rounded Box
rounded_box = RoundedBoxSDF(center=[0, 0, 0], extents=[1.0, 1.0, 1.0], radius=0.15)

# Wireframe Box
wireframe_box = WireframeBoxSDF(center=[0, 0, 0], extents=[1.0, 1.0, 1.0], thickness=0.08)

# Capsule
capsule = CapsuleSDF(point_a=[0, 0, -0.5], point_b=[0, 0, 0.5], radius=0.25)

# Ellipsoid
ellipsoid = EllipsoidSDF(center=[0, 0, 0], extents=[0.6, 0.8, 0.5])

# Pyramid
pyramid = PyramidSDF(height=1.0)

# Platonic Solids
tetrahedron = TetrahedronSDF(r=0.8)
octahedron = OctahedronSDF(r=0.8)
dodecahedron = DodecahedronSDF(r=0.8)
icosahedron = IcosahedronSDF(r=0.8)

# With caps and rounded transitions
capped_cylinder = CappedCylinderSDF(point_a=[0, 0, -0.5], point_b=[0, 0, 0.5], radius=0.3)
rounded_cylinder = RoundedCylinderSDF(ra=0.3, rb=0.1, h=1.0)
capped_cone = CappedConeSDF(point_a=[0, 0, -0.5], point_b=[0, 0, 0.5], ra=0.1, rb=0.3)
rounded_cone = RoundedConeSDF(r1=0.1, r2=0.3, h=1.0)

# Special structures
corner_spheres = CornerSpheresSDF(radius=0.15, limit=0.8)
cross_ms = CrossMsSDF(radius=0.15)
```

</details>

### 2D Primitives

<table>
    <tbody>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/2D/circle.png" alt="Circle"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/2D/rectangle.png" alt="Rectangle"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/2D/rounded_rectangle.png" alt="Rounded Rectangle"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/2D/equilateral_triangle.png" alt="Equilateral Triangle"></td>
        </tr>
        <tr>
            <td align="center"><b>Circle:</b> Circle with configurable center and radius</td>
            <td align="center"><b>Rectangle:</b> Axis-aligned 2D rectangle</td>
            <td align="center"><b>Rounded Rectangle:</b> Rectangle with rounded corners</td>
            <td align="center"><b>Equilateral Triangle:</b> Regular equilateral triangle</td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/2D/hexagon.png" alt="Hexagon"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/primitives/2D/polygon_pentagon.png" alt="Polygon (Pentagon)"></td>
        </tr>
        <tr>
            <td align="center"><b>Hexagon:</b> Regular hexagon</td>
            <td align="center"><b>Polygon:</b> Custom polygon (demonstrated with pentagon)</td>
        </tr>
    </tbody>
</table>

<details>
<summary><b>Click to see 2D primitive code examples</b></summary>

```python
from DeepSDFStruct.sdf_primitives import (
    CircleSDF, RectangleSDF, RoundedRectangleSDF,
    EquilateralTriangleSDF, HexagonSDF, PolygonSDF
)

# Circle
circle = CircleSDF(center=[0, 0], radius=1.0)

# Rectangle
rectangle = RectangleSDF(center=[0, 0], extents=[2.0, 1.5])

# Rounded Rectangle
rounded_rectangle = RoundedRectangleSDF(center=[0, 0], extents=[2.0, 1.5], radius=0.2)

# Equilateral Triangle
triangle = EquilateralTriangleSDF(size=1.5)

# Hexagon
hexagon = HexagonSDF(size=1.5)

# Custom Polygon (Pentagon)
pentagon = PolygonSDF([
    [0, 1],
    [0.95, 0.31],
    [0.59, -0.81],
    [-0.59, -0.81],
    [-0.95, 0.31],
])
```

</details>

### SDF Operations

<table>
    <tbody>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/boolean/union_sphere_box.png" alt="Union"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/boolean/difference_sphere_box.png" alt="Difference"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/transformations/twist_torus.png" alt="Twist"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/transformations/dilate_sphere.png" alt="Dilate"></td>
        </tr>
        <tr>
            <td align="center"><b>Union:</b> Union of sphere and box</td>
            <td align="center"><b>Difference:</b> Sphere with box subtracted</td>
            <td align="center"><b>Twist:</b> Torus twisted by 90 degrees around Z-axis</td>
            <td align="center"><b>Dilate:</b> Sphere expanded by uniform distance</td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/transformations/shell_sphere.png" alt="Shell"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/transformations/repeat_sphere.png" alt="Repeat"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/transformations/mirror_sphere.png" alt="Mirror"></td>
            <td style="width: 25%;"><img src="benchmarks/sdf_showcase/operations/transformations/circular_array_sphere.png" alt="Circular Array"></td>
        </tr>
        <tr>
            <td align="center"><b>Shell:</b> Sphere with hollow shell</td>
            <td align="center"><b>Repeat:</b> Sphere repeated in 3D grid pattern</td>
            <td align="center"><b>Mirror:</b> Sphere reflected across YZ plane</td>
            <td align="center"><b>Circular Array:</b> Sphere replicated 6 times in circular pattern</td>
        </tr>
        <tr>
            <td colspan="2" style="width: 50%;"><img src="benchmarks/sdf_showcase/operations/transformations/revolve_circle.png" alt="Revolve"></td>
        </tr>
        <tr>
            <td colspan="2" align="center"><b>Revolve:</b> 2D circle revolved around Z-axis creates torus</td>
        </tr>
    </tbody>
</table>

<details>
<summary><b>Click to see boolean operation code examples</b></summary>

```python
from DeepSDFStruct.sdf_primitives import SphereSDF, BoxSDF
from DeepSDFStruct.SDF import UnionSDF, DifferenceSDF

sphere = SphereSDF(center=[-0.3, 0, 0], radius=0.5)
box = BoxSDF(center=[0.3, 0, 0], extents=[0.8, 0.8, 0.8])

# Union (can also use + operator)
union = UnionSDF(sphere, box)

# Difference
difference = DifferenceSDF(sphere, box)
```

</details>

<details>
<summary><b>Click to see transformation operation code examples</b></summary>

```python
import numpy as np
from DeepSDFStruct.sdf_primitives import SphereSDF, TorusSDF, CircleSDF
from DeepSDFStruct.sdf_operations import TwistSDF, DilateSDF, ShellSDF, RepeatSDF, MirrorSDF, CircularArraySDF, RevolveSDF

# Twisted Torus
base_torus = TorusSDF(center=[0, 0, 0], axis=[0, 0, 1], major_radius=1.0, minor_radius=0.2)
twisted_torus = TwistSDF(base_torus, k=np.pi/2)

# Dilated Sphere
sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
dilated_sphere = DilateSDF(sphere, r=0.15)

# Shell
shell_sphere = ShellSDF(sphere, thickness=0.1)

# Repeat (creates infinite grid)
small_sphere = SphereSDF(center=[0, 0, 0], radius=0.2)
repeated_sphere = RepeatSDF(small_sphere, spacing=[0.6, 0.6, 0.6])

# Mirror
off_center_sphere = SphereSDF(center=[0.5, 0, 0], radius=0.3)
mirrored_sphere = MirrorSDF(off_center_sphere, plane_point=[0, 0, 0], plane_normal=[1, 0, 0])

# Circular Array
arrayed_sphere = CircularArraySDF(off_center_sphere, count=6, radius=0.8)

# Revolve (2D to 3D)
circle_2d = CircleSDF(center=[0.8, 0], radius=0.15)
revolved_shape = RevolveSDF(circle_2d, axis='z')
```

</details>

All SDFs can be evaluated at query points:
```python
points = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float32)
distances = sphere(points)  # Returns signed distances
```

*These images are automatically generated from the actual SDF implementations. Run `uv run python benchmarks/generate_sdf_showcase.py` to regenerate all showcases.*

## 🔗 Repository

GitHub: [https://github.com/mkofler96/DeepSDFStruct](https://github.com/mkofler96/DeepSDFStruct)

---

## 📄 License
This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file or visit [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0) for more information.

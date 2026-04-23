"""
SDF Primitive and Operation Showcase
====================================

This script generates visual examples of all SDF primitives and operations
implemented in DeepSDFStruct, creating screenshots suitable for documentation.

The workflow:
1. Create an SDF (primitive or operation)
2. Generate a 3D mesh using create_3D_mesh()
3. Convert to gustaf format with .to_gus()
4. Create screenshots using vedo
5. Save to organized directory structure

Usage:
    uv run python benchmarks/generate_sdf_showcase.py

Output:
    benchmarks/sdf_showcase/
        ├── primitives/
        │   ├── 3D/
        │   │   ├── sphere.png
        │   │   ├── box.png
        │   │   ├── ...
        │   └── 2D/
        │       ├── circle.png
        │       ├── ...
        └── operations/
            ├── boolean/
            └── transformations/
"""

import sys
import os
import pathlib
import numpy as np
import torch
import vedo
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Add parent directory to import DeepSDFStruct
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from DeepSDFStruct.mesh import create_3D_mesh, create_2D_mesh
from DeepSDFStruct.sdf_primitives import (
    # 3D Primitives
    SphereSDF,
    BoxSDF,
    CylinderSDF,
    ConeSDF,
    TorusSDF,
    PlaneSDF,
    RoundedBoxSDF,
    WireframeBoxSDF,
    CapsuleSDF,
    EllipsoidSDF,
    PyramidSDF,
    CornerSpheresSDF,
    CrossMsSDF,
    RoundedCylinderSDF,
    CappedConeSDF,
    RoundedConeSDF,
    TetrahedronSDF,
    OctahedronSDF,
    DodecahedronSDF,
    IcosahedronSDF,
    # 2D Primitives
    CircleSDF,
    RectangleSDF,
    LineSDF,
    RoundedRectangleSDF,
    EquilateralTriangleSDF,
    HexagonSDF,
    PolygonSDF,
)
from DeepSDFStruct.sdf_operations import (
    ElongateSDF,
    TwistSDF,
    BendLinearSDF,
    BendRadialSDF,
    DilateSDF,
    ErodeSDF,
    ShellSDF,
    RepeatSDF,
    MirrorSDF,
    CircularArraySDF,
    RevolveSDF,
)
from DeepSDFStruct.SDF import UnionSDF, DifferenceSDF

# ==================== Configuration ====================

# Output directory structure
OUTPUT_DIR = pathlib.Path("benchmarks/sdf_showcase")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Camera settings for consistent screenshots
CAMERA_SETTINGS = dict(
    position=(2.5, 2.5, 2.5), focal_point=(0, 0, 0), viewup=(0, 0, 1)
)


# ==================== Helper Functions ====================


def create_screenshot(
    sdf,
    filename: str,
    resolution: int = 64,
    title: str = "",
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    mesh_color: str = "#6B9BD2",
) -> bool:
    """
    Generate a screenshot from an SDF.

    Parameters
    ----------
    sdf : SDFBase
        The SDF to visualize
    filename : str
        Output filename (will be saved as .png)
    resolution : int, optional
        Mesh resolution (default: 64)
    title : str, optional
        Title for the visualization (shown as 2D text overlay)
    background_color : tuple, optional
        RGB background color (default: white)
    mesh_color : str, optional
        Hex color for the mesh (default: nice blue)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Create 3D mesh from SDF
        mesh, _ = create_3D_mesh(
            sdf, N_base=resolution, mesh_type="surface", device="cpu"
        )

        # Convert to gustaf format
        gus_mesh = mesh.to_gus()

        # Create vedo mesh
        vedo_mesh = vedo.Mesh(
            [gus_mesh.vertices.astype(np.float64), gus_mesh.faces.astype(np.int64)]
        )
        vedo_mesh.color(mesh_color).lighting("glossy")

        # Setup plotter
        plt = vedo.Plotter(interactive=False)
        plt.offscreen = True  # Don't open window
        plt.background_color = background_color

        # Create the scene
        plt.show(vedo_mesh, camera=vedo.camera_from_dict(CAMERA_SETTINGS))

        # Add title if provided
        if title:
            title_text = vedo.Text2D(
                title, pos="top-left", font="Arial", s=0.05, c="black"
            )
            plt.show(title_text)

        # Save screenshot
        plt.screenshot(filename, scale=2.0)  # Higher resolution
        plt.close()

        print(f"  ✓ Created: {filename}")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {filename} - {e}")
        return False


def create_2d_screenshot(
    sdf,
    filename: str,
    resolution: int = 200,
    title: str = "",
    bounds: Optional[Tuple[float, float, float, float]] = None,
    query_offset: Optional[Tuple[float, float]] = None,
    interior_color: str = "#6B9BD2",
    exterior_color: str = "#FFFFFF",
) -> bool:
    """
    Generate a screenshot from a 2D SDF using matplotlib contour.

    Parameters
    ----------
    sdf : SDFBase (2D)
        The 2D SDF to visualize
    filename : str
        Output filename (will be saved as .png)
    resolution : int, optional
        Grid resolution for sampling (default: 200)
    title : str, optional
        Title for the visualization
    bounds : tuple, optional
        Plot bounds (xmin, xmax, ymin, ymax). If None, uses SDF domain bounds.
    query_offset : tuple, optional
        Offset applied to sampling coordinates before SDF evaluation.
    interior_color : str, optional
        Hex color for interior region (default: nice blue)
    exterior_color : str, optional
        Hex color for exterior region (default: white)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get bounds if not provided
        if bounds is None:
            sdf_bounds = sdf._get_domain_bounds()
            bounds = (
                sdf_bounds[0, 0].item(),
                sdf_bounds[1, 0].item(),
                sdf_bounds[0, 1].item(),
                sdf_bounds[1, 1].item(),
            )

        # Create grid for sampling
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[2], bounds[3], resolution)
        X, Y = np.meshgrid(x, y)

        # Sample SDF values
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        if query_offset is not None:
            points_for_eval = points - np.asarray(query_offset, dtype=np.float32)
        else:
            points_for_eval = points
        sdf_values = (
            sdf(torch.tensor(points_for_eval, dtype=torch.float32))
            .detach()
            .numpy()
            .reshape(X.shape)
        )

        # Create plot
        figsize = (6, 6) if not title else (6, 7)
        if title:
            return_list = [1]
        else:
            return_list = []

        fig, axes = plt.subplots(figsize=figsize)

        # Create filled contour plot (interior vs exterior)
        # SDF < 0 is interior, SDF > 0 is exterior
        contour = axes.contourf(
            X,
            Y,
            sdf_values,
            levels=[-100, 0, 100],
            colors=[interior_color, exterior_color],
        )

        # Add contour line at SDF = 0 (boundary)
        axes.contour(X, Y, sdf_values, levels=[0], colors="black", linewidths=2)

        # Set plot properties
        axes.set_aspect("equal")
        axes.set_xlim(bounds[0], bounds[1])
        axes.set_ylim(bounds[2], bounds[3])
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        if title:
            axes.set_title(title)

        # Remove ticks for cleaner look
        axes.set_xticks([])
        axes.set_yticks([])

        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  ✓ Created: {filename}")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {filename} - {e}")
        import traceback

        traceback.print_exc()
        return False


# ==================== SDF Definitions ====================


def create_sdf_primitives():
    """Create all SDF primitives and generate screenshots."""

    # ======== 3D Primitives ========
    output_dir = OUTPUT_DIR / "primitives" / "3D"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== 3D Primitives ===")

    # Sphere
    print("Sphere")
    sphere = SphereSDF(center=[0, 0, 0], radius=1.0)
    create_screenshot(sphere, output_dir / "sphere.png", resolution=64)

    # Box
    print("Box")
    box = BoxSDF(center=[0, 0, 0], extents=[1.5, 1.0, 0.8])
    create_screenshot(box, output_dir / "box.png", resolution=64)

    # Cone

    # Cone
    print("Cone")
    cone = ConeSDF(apexpoint=[0, 0, -0.5], axis=[0, 0, 1], radius=0.4, height=1.0)
    create_screenshot(cone, output_dir / "cone.png", resolution=64)

    # Torus
    print("Torus")
    torus = TorusSDF(
        center=[0, 0, 0], axis=[0, 0, 1], major_radius=0.5, minor_radius=0.15
    )
    create_screenshot(torus, output_dir / "torus.png", resolution=64)

    # Rounded Box
    print("Rounded Box")
    rounded_box = RoundedBoxSDF(center=[0, 0, 0], extents=[1.0, 1.0, 1.0], radius=0.15)
    create_screenshot(rounded_box, output_dir / "rounded_box.png", resolution=64)

    # Wireframe Box
    print("Wireframe Box")
    wireframe_box = WireframeBoxSDF(
        center=[0, 0, 0], extents=[1.0, 1.0, 1.0], thickness=0.08
    )
    create_screenshot(wireframe_box, output_dir / "wireframe_box.png", resolution=64)

    # Capsule
    print("Capsule")
    capsule = CapsuleSDF(point_a=[0, 0, -0.5], point_b=[0, 0, 0.5], radius=0.25)
    create_screenshot(capsule, output_dir / "capsule.png", resolution=64)

    # Ellipsoid
    print("Ellipsoid")
    ellipsoid = EllipsoidSDF(center=[0, 0, 0], extents=[0.6, 0.8, 0.5])
    create_screenshot(ellipsoid, output_dir / "ellipsoid.png", resolution=64)

    # Pyramid
    print("Pyramid")
    pyramid = PyramidSDF(height=1.0)
    create_screenshot(pyramid, output_dir / "pyramid.png", resolution=64)

    # Corner Spheres
    print("Corner Spheres")
    corner_spheres = CornerSpheresSDF(radius=0.15, limit=0.8)
    create_screenshot(corner_spheres, output_dir / "corner_spheres.png", resolution=64)

    # Cross M
    print("Cross M")
    cross_ms = CrossMsSDF(radius=0.15)
    create_screenshot(cross_ms, output_dir / "cross_ms.png", resolution=64)

    # Cylinder
    print("Cylinder")
    cylinder = CylinderSDF(point_a=[0, 0, -0.5], point_b=[0, 0, 0.5], radius=0.3)
    create_screenshot(cylinder, output_dir / "cylinder.png", resolution=64)

    # Rounded Cylinder
    print("Rounded Cylinder")
    rounded_cyl = RoundedCylinderSDF(ra=0.3, rb=0.1, h=1.0)
    create_screenshot(rounded_cyl, output_dir / "rounded_cylinder.png", resolution=64)

    # Capped Cone
    print("Capped Cone")
    capped_cone = CappedConeSDF(
        point_a=[0, 0, -0.5], point_b=[0, 0, 0.5], ra=0.1, rb=0.3
    )
    create_screenshot(capped_cone, output_dir / "capped_cone.png", resolution=64)

    # Rounded Cone
    print("Rounded Cone")
    rounded_cone = RoundedConeSDF(r1=0.1, r2=0.3, h=1.0)
    create_screenshot(rounded_cone, output_dir / "rounded_cone.png", resolution=64)

    # Platonic Solids
    print("Tetrahedron")
    tetra = TetrahedronSDF(r=0.8)
    create_screenshot(tetra, output_dir / "tetrahedron.png", resolution=64)

    print("Octahedron")
    octa = OctahedronSDF(r=0.8)
    create_screenshot(octa, output_dir / "octahedron.png", resolution=64)

    print("Dodecahedron")
    dodeca = DodecahedronSDF(r=0.8)
    create_screenshot(dodeca, output_dir / "dodecahedron.png", resolution=64)

    print("Icosahedron")
    icosa = IcosahedronSDF(r=0.8)
    create_screenshot(icosa, output_dir / "icosahedron.png", resolution=64)

    # ======== 2D Primitives ========
    output_dir = OUTPUT_DIR / "primitives" / "2D"
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_bounds = (0.0, 1.0, 0.0, 1.0)

    print("\\n=== 2D Primitives ===")

    # Circle
    print("Circle")
    circle = CircleSDF(center=[0.5, 0.5], radius=0.35)
    create_2d_screenshot(
        circle, output_dir / "circle.png", resolution=200, bounds=unit_bounds
    )

    # Rectangle
    print("Rectangle")
    rectangle = RectangleSDF(center=[0.5, 0.5], extents=[0.7, 0.55])
    create_2d_screenshot(
        rectangle, output_dir / "rectangle.png", resolution=200, bounds=unit_bounds
    )

    # Rounded Rectangle
    print("Rounded Rectangle")
    rounded_rect = RoundedRectangleSDF(
        center=[0.5, 0.5], extents=[0.72, 0.56], radius=0.08
    )
    create_2d_screenshot(
        rounded_rect,
        output_dir / "rounded_rectangle.png",
        resolution=200,
        bounds=unit_bounds,
    )

    # Equilateral Triangle
    print("Equilateral Triangle")
    triangle = EquilateralTriangleSDF(size=0.32)
    create_2d_screenshot(
        triangle,
        output_dir / "equilateral_triangle.png",
        resolution=200,
        bounds=unit_bounds,
        query_offset=(0.5, 0.5),
    )

    # Hexagon
    print("Hexagon")
    hexagon = HexagonSDF(size=0.28)
    create_2d_screenshot(
        hexagon,
        output_dir / "hexagon.png",
        resolution=200,
        bounds=unit_bounds,
        query_offset=(0.5, 0.5),
    )

    # Polygon (custom shape)
    print("Polygon (Pentagon)")
    pentagon = PolygonSDF(
        [
            [0.5, 0.86],  # Top
            [0.82, 0.62],  # Top right
            [0.7, 0.24],  # Bottom right
            [0.3, 0.24],  # Bottom left
            [0.18, 0.62],  # Top left
        ]
    )
    create_2d_screenshot(
        pentagon,
        output_dir / "polygon_pentagon.png",
        resolution=200,
        bounds=unit_bounds,
    )

    # ======== Boolean Operations ========
    output_dir = OUTPUT_DIR / "operations" / "boolean"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Boolean Operations ===")

    # Union
    print("Sphere + Box Union")
    sphere = SphereSDF(center=[0.0, 0, 0], radius=0.5)
    box = BoxSDF(center=[0.0, 0, 0], extents=[2.0, 0.3, 0.3])
    union = sphere + box  # UnionSDF is the + operator
    create_screenshot(
        union, output_dir / "union_sphere_box.png", resolution=64, title="Union"
    )

    # Difference
    print("Sphere - Box Difference")
    diff = DifferenceSDF(sphere, box)
    create_screenshot(
        diff,
        output_dir / "difference_sphere_box.png",
        resolution=64,
        title="Difference",
    )

    # ======== Transformation Operations ========
    output_dir = OUTPUT_DIR / "operations" / "transformations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Transformation Operations ===")

    # Twist
    print("Twisted Torus")
    base_torus = TorusSDF(
        center=[0, 0, 0], axis=[0, 0, 1], major_radius=1.0, minor_radius=0.2
    )
    twist_params = np.pi / 2  # 90 degree twist
    twisted_torus = TwistSDF(base_torus, k=twist_params)
    create_screenshot(
        twisted_torus, output_dir / "twist_torus.png", resolution=64, title="Twist"
    )

    # Dilate
    print("Dilated Sphere")
    base_sphere = SphereSDF(center=[0, 0, 0], radius=0.5)
    dilated_sphere = DilateSDF(base_sphere, r=0.15)
    create_screenshot(
        dilated_sphere, output_dir / "dilate_sphere.png", resolution=64, title="Dilated"
    )

    # Shell
    print("Sphere Shell")
    shell_sphere = ShellSDF(base_sphere, thickness=0.1)
    create_screenshot(
        shell_sphere, output_dir / "shell_sphere.png", resolution=64, title="Shell"
    )

    # Repeat
    print("Repeated Spheres (2D array)")
    small_sphere = SphereSDF(center=[0, 0, 0], radius=0.2)
    repeated_sphere = RepeatSDF(small_sphere, spacing=[0.6, 0.6, 0.6])
    create_screenshot(
        repeated_sphere,
        output_dir / "repeat_sphere.png",
        resolution=64,
        title="Repeated (2x2x2)",
    )

    # Mirror
    print("Mirrored Sphere")
    off_center_sphere = SphereSDF(center=[0.5, 0, 0], radius=0.3)
    mirrored_sphere = MirrorSDF(
        off_center_sphere, plane_point=[0, 0, 0], plane_normal=[1, 0, 0]
    )
    create_screenshot(
        mirrored_sphere,
        output_dir / "mirror_sphere.png",
        resolution=64,
        title="Mirrored",
    )

    # Circular Array
    print("Circular Array of Spheres")
    arrayed_sphere = CircularArraySDF(off_center_sphere, count=6, radius=0.8)
    create_screenshot(
        arrayed_sphere,
        output_dir / "circular_array_sphere.png",
        resolution=64,
        title="Circular Array (6x)",
    )

    # Revolve (2D to 3D)
    print("Revolved Circle")
    circle_2d = CircleSDF(center=[0.8, 0], radius=0.15)
    revolved_shape = RevolveSDF(
        circle_2d, axis=torch.tensor([0, 0, 1], dtype=torch.float32)
    )
    create_screenshot(
        revolved_shape,
        output_dir / "revolve_circle.png",
        resolution=64,
        title="Revolved (Torus)",
    )


# ==================== Main Script ====================


def main():
    """Generate all SDF showcase images."""

    print("=" * 60)
    print("SDF Primitive and Operation Showcase Generator")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")

    try:
        create_sdf_primitives()

        print("\n" + "=" * 60)
        print("✓ Showcase generation complete!")
        print("=" * 60)
        print(f"\nGenerated screenshots saved to: {OUTPUT_DIR.absolute()}")
        print("\nThese images can be used for documentation and README.")

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

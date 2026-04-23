# SDF Feature Tracking

This document tracks the implementation status and test coverage of SDF primitives and operations, inspired by the features available in `https://github.com/nobodyinperson/sdf`.

## 3D Primitives

| Primitive | Implemented | Tested | Notes |
| :--- | :---: | :---: | :--- |
| Sphere | ✅ | | `SphereSDF` |
| Box | ✅ | | `BoxSDF` |
| Rounded Box | ✅ | ⬜️ | `RoundedBoxSDF` |
| Wireframe Box | ✅ | ⬜️ | `WireframeBoxSDF` |
| Torus | ✅ | | `TorusSDF` |
| Capsule | ✅ | ⬜️ | `CapsuleSDF` |
| Cylinder | ✅ | | `CylinderSDF` |
| Capped Cylinder | ⬜️ | ⬜️ | |
| Rounded Cylinder | ⬜️ | ⬜️ | |
| Cone | ✅ | | `ConeSDF` |
| Capped Cone | ⬜️ | ⬜️ | |
| Rounded Cone | ⬜️ | ⬜️ | |
| Plane | ✅ | | `PlaneSDF` |
| Slab | ⬜️ | ⬜️ | Can be done with `Intersection` of `PlaneSDF`s |
| Ellipsoid | ✅ | ⬜️ | `EllipsoidSDF` |
| Pyramid | ✅ | ⬜️ | `PyramidSDF` |
| Tetrahedron | ⬜️ | ⬜️ | |
| Octahedron | ⬜️ | ⬜️ | |
| Dodecahedron | ⬜️ | ⬜️ | |
| Icosahedron | ⬜️ | ⬜️ | |

## 2D Primitives

| Primitive | Implemented | Tested | Notes |
| :--- | :---: | :---: | :--- |
| Circle | ✅ | | `CircleSDF` |
| Rectangle | ✅ | | `RectangleSDF` |
| Rounded Rectangle | ⬜️ | ⬜️ | |
| Line | ⬜️ | ⬜️ | |
| Equilateral Triangle | ⬜️ | ⬜️ | |
| Hexagon | ⬜️ | ⬜️ | |
| Polygon | ⬜️ | ⬜️ | |

## Operations

| Operation | Implemented | Tested | Notes |
| :--- | :---: | :---: | :--- |
| **Boolean** | | | |
| Union | ✅ | | `UnionSDF` or `+` operator |
| Difference | ✅ | | `DifferenceSDF` or `-` operator |
| Intersection | ✅ | | `&` operator (via `__add__` logic) |
| Negate | ✅ | | `NegatedCallable` |
| **Transformations** | | | |
| Translate | ✅ | | Part of `TransformedSDF` |
| Rotate | ✅ | | Part of `TransformedSDF` |
| Scale | ✅ | | Part of `TransformedSDF` |
| **Alterations** | | | |
| Elongate | ⬜️ | ⬜️ | |
| Twist | ⬜️ | ⬜️ | |
| Bend | ⬜️ | ⬜️ | |
| Shell / Dilate / Erode | ⬜️ | ⬜️ | |
| **2D -> 3D** | | | |
| Extrude | ✅ | ⬜️ | `ExtrudeSDF` |
| Revolve | ⬜️ | ⬜️ | |
| **Other** | | | |
| Repeat | ⬜️ | ⬜️ | |
| Mirror | ⬜️ | ⬜️ | |
| Circular Array | ⬜️ | ⬜️ | |

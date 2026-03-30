import time
import trimesh
import torch
import pandas as pd
from DeepSDFStruct.SDF import SDFfromMesh


def run_benchmark():
    mesh_configs = {
        "Sphere (Low)": trimesh.creation.icosphere(subdivisions=2),
        "Sphere (Med)": trimesh.creation.icosphere(subdivisions=4),
        "Bunny (High)": trimesh.load("tests/data/stanford_bunny.stl"),
    }

    query_sizes = [10**4, 10**5, 10**6]
    backends = ["igl", "trimesh"]
    results = []

    print(f"{'Mesh':<15} | {'Points':<10} | {'Backend':<10} | {'Time (s)':<10}")
    print("-" * 55)

    for name, mesh in mesh_configs.items():
        for n_points in query_sizes:
            queries = torch.randn((n_points, 3))

            for backend in backends:
                sdf_gen = SDFfromMesh(mesh, backend=backend, scale=True)

                start_time = time.perf_counter()
                _ = sdf_gen(queries)
                end_time = time.perf_counter()

                elapsed = end_time - start_time
                results.append(
                    {
                        "Mesh": name,
                        "Faces": len(mesh.faces),
                        "Points": n_points,
                        "Backend": backend,
                        "Time": elapsed,
                    }
                )
                print(f"{name:<15} | {n_points:<10} | {backend:<10} | {elapsed:.4f}")

    return pd.DataFrame(results)


df = run_benchmark()
summary = df.pivot_table(index=["Mesh", "Points"], columns="Backend", values="Time")
print("\nSpeedup (IGL vs Trimesh):")
summary["Speedup (x)"] = summary["trimesh"] / summary["igl"]
print(summary)

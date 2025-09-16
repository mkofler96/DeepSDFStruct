import numpy as np
from DeepSDFStruct.sampling import SDFSampler
from DeepSDFStruct.splinepy_unitcells.chi_3D import Chi3D
from DeepSDFStruct.splinepy_unitcells.cross_lattice import CrossLattice
import splinepy


def test_generate_dataset():
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


if __name__ == "__main__":
    test_generate_dataset()

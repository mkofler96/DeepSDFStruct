import json
import os

from DeepSDFStruct.deep_sdf.training import save_latent_code_data_map
import DeepSDFStruct.deep_sdf.workspace as ws


def test_save_latent_code_data_map(tmp_path):
    experiment_directory = str(tmp_path / "experiment")
    data_source = str(tmp_path / "data_source")
    npz_filenames = [
        "dataset_a/class_1/instance_1.npz",
        "dataset_b/class_2/instance_2.npz",
    ]

    save_latent_code_data_map(experiment_directory, data_source, npz_filenames)

    mapping_filename = ws.get_latent_code_data_map_filename(experiment_directory)
    assert os.path.isfile(mapping_filename)

    with open(mapping_filename, "r", encoding="utf-8") as f:
        latent_code_data_map = json.load(f)

    assert latent_code_data_map["data_source"] == data_source
    assert latent_code_data_map["sdf_samples_subdir"] == ws.sdf_samples_subdir
    assert latent_code_data_map["latent_codes"] == [
        {
            "latent_index": 0,
            "relative_npz_filename": npz_filenames[0],
            "npz_filename": os.path.join(
                data_source, ws.sdf_samples_subdir, npz_filenames[0]
            ),
        },
        {
            "latent_index": 1,
            "relative_npz_filename": npz_filenames[1],
            "npz_filename": os.path.join(
                data_source, ws.sdf_samples_subdir, npz_filenames[1]
            ),
        },
    ]
